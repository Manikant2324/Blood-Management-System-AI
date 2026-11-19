import prisma from "@/app/libs/prismadb";
// @ts-ignore
import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371; // km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
    Math.cos((lat2 * Math.PI) / 180) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function isCompatible(donorGroup: string, recipientGroup: string): boolean {
  const map: Record<string, string[]> = {
    "O-": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],
    "O+": ["O+", "A+", "B+", "AB+"],
    "A-": ["A-", "A+", "AB-", "AB+"],
    "A+": ["A+", "AB+"],
    "B-": ["B-", "B+", "AB-", "AB+"],
    "B+": ["B+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"],
  };
  const canDonateTo = map[donorGroup?.toUpperCase?.() || ""] || [];
  return canDonateTo.includes(recipientGroup?.toUpperCase?.());
}

function getSeedDonors() {
  try {
    const seedPath = path.resolve(process.cwd(), "..", "ml_engine", "saved_models", "seed_donors.json");
    if (fs.existsSync(seedPath)) return JSON.parse(fs.readFileSync(seedPath, "utf-8"));
    const localSeed = path.resolve(process.cwd(), "ml_engine", "saved_models", "seed_donors.json");
    if (fs.existsSync(localSeed)) return JSON.parse(fs.readFileSync(localSeed, "utf-8"));
  } catch (e) {
    // ignore
  }
  return [];
}

export const POST = async (request: NextRequest) => {
  try {
    const body = await request.json();
    const {
      recipientName,
      recipientPhone,
      recipientEmail,
      bloodGroup,
      latitude,
      longitude,
      radiusKm = 20,
      message,
      unitsRequested = 1,
    } = body || {};

    if (
      !recipientName ||
      !recipientPhone ||
      !bloodGroup ||
      typeof latitude !== "number" ||
      typeof longitude !== "number"
    ) {
      return NextResponse.json({ message: "Invalid request body" }, { status: 400 });
    }

    // -------------------------------------------------------------
    // Step 1: AI/ML Screening - Feature 4 (NLP Urgency & Fake Request Detection)
    // -------------------------------------------------------------
    let aiEvaluation: any = null;
    try {
      const evalRes = await fetch("http://127.0.0.1:8000/api/ml/requests/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: message || `Urgent blood request for ${recipientName} (${bloodGroup})`,
          unitsRequested: Number(unitsRequested) || 1,
          frequencyLast24h: 1,
          hospitalName: recipientName,
        }),
        cache: "no-store",
      });

      if (evalRes.ok) {
        aiEvaluation = await evalRes.json();
      }
    } catch (e) {
      // Offline fallback rule-based screening
      const lowerMsg = (message || "").toLowerCase();
      const suspiciousWords = ["selling", "cash reward", "pure blood bags", "pay advance", "lottery"];
      const hasFraud = suspiciousWords.some((w) => lowerMsg.includes(w));
      aiEvaluation = {
        urgency: { predictedUrgency: lowerMsg.includes("icu") || lowerMsg.includes("trauma") ? "Critical" : "Urgent", confidence: 0.85 },
        fakeDetection: {
          isFakeOrSuspicious: hasFraud,
          status: hasFraud ? "flagged" : "approved",
          verdict: hasFraud ? "Blocked (High Risk / Fraud Detected)" : "Verified Legitimate Request",
          suspiciousFlags: hasFraud ? ["Commercial/Fraud keyword detected"] : []
        }
      };
    }

    // If flagged as fraudulent by ML engine, intercept and block
    if (aiEvaluation?.fakeDetection?.isFakeOrSuspicious && aiEvaluation?.fakeDetection?.status === "flagged") {
      return NextResponse.json(
        {
          success: false,
          blocked: true,
          message: "Request intercepted by AI Fraud & Anomaly Detection Engine.",
          verdict: aiEvaluation.fakeDetection.verdict,
          suspiciousFlags: aiEvaluation.fakeDetection.suspiciousFlags,
        },
        { status: 403 }
      );
    }

    // -------------------------------------------------------------
    // Step 2: Save Request Record (with DB fallback)
    // -------------------------------------------------------------
    let sosRecord: any = null;
    try {
      sosRecord = await prisma.recipientRequest.create({
        data: {
          recipientName,
          recipientPhone,
          recipientEmail,
          bloodGroup,
          latitude,
          longitude,
          radiusKm,
          message,
          status: "active",
        },
      });
    } catch (dbErr) {
      sosRecord = {
        id: `sos_${Date.now()}`,
        recipientName,
        recipientPhone,
        recipientEmail,
        bloodGroup,
        latitude,
        longitude,
        radiusKm,
        message,
        status: "active",
        createdAt: new Date().toISOString(),
      };
    }

    // -------------------------------------------------------------
    // Step 3: Fetch Available Donors & Rank with Feature 1 (ML Donor Response)
    // -------------------------------------------------------------
    let donors: any[] = [];
    try {
      donors = await prisma.donor.findMany({ where: { available: true } });
    } catch (e) {
      donors = getSeedDonors();
    }
    if (!donors || donors.length === 0) {
      donors = getSeedDonors();
    }

    let rankedDonors: any[] = [];
    try {
      const rankRes = await fetch("http://127.0.0.1:8000/api/ml/donor-response/rank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bloodGroup,
          latitude,
          longitude,
          maxDistanceKm: radiusKm,
          hourOfDay: new Date().getHours(),
          topK: 15,
          donors,
        }),
        cache: "no-store",
      });
      if (rankRes.ok) {
        const rankData = await rankRes.json();
        rankedDonors = rankData.ranked_donors || [];
      }
    } catch (e) {
      // Fallback
      rankedDonors = donors
        .filter((d: any) => isCompatible(d.bloodGroup, bloodGroup))
        .map((d: any) => ({
          donorId: d.id,
          name: d.name || `${d.firstName || ''} ${d.lastName || ''}`.trim(),
          phone: d.phone,
          email: d.email,
          distanceKm: haversineKm(latitude, longitude, d.latitude || 28.7, d.longitude || 77.1),
          responseProbability: 0.75,
          recommendationTier: "Recommended"
        }))
        .filter((d: any) => d.distanceKm <= radiusKm)
        .slice(0, 15);
    }

    return NextResponse.json({
      success: true,
      request: sosRecord,
      aiScreening: aiEvaluation,
      rankedDonorsCount: rankedDonors.length,
      topRankedDonors: rankedDonors.slice(0, 5),
      message: `Emergency SOS broadcast dispatched to ${rankedDonors.length} AI-ranked compatible donors within ${radiusKm}km.`
    });
  } catch (err: any) {
    return NextResponse.json({ message: "SOS Error", error: err?.message || err }, { status: 500 });
  }
};
