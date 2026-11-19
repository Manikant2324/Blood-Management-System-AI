import prisma from "@/app/libs/prismadb";
import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

type MatchRequestBody = {
  bloodGroup: string;
  latitude: number;
  longitude: number;
  maxDistanceKm?: number;
  hourOfDay?: number;
  topK?: number;
  weights?: {
    distance?: number;
    availability?: number;
    recency?: number;
  };
};

function haversineKm(lat1: number, lon1: number, lat2?: number | null, lon2?: number | null): number | null {
  if (lat2 == null || lon2 == null) return null;
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

// Basic ABO/Rh compatibility rules
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

// Fallback seed donors if database is empty or offline
function getSeedDonors() {
  try {
    const seedPath = path.resolve(process.cwd(), "..", "ml_engine", "saved_models", "seed_donors.json");
    if (fs.existsSync(seedPath)) {
      return JSON.parse(fs.readFileSync(seedPath, "utf-8"));
    }
    const localSeed = path.resolve(process.cwd(), "ml_engine", "saved_models", "seed_donors.json");
    if (fs.existsSync(localSeed)) {
      return JSON.parse(fs.readFileSync(localSeed, "utf-8"));
    }
  } catch (e) {
    // ignore
  }
  return [];
}

export const POST = async (request: NextRequest) => {
  try {
    const body = (await request.json()) as MatchRequestBody;
    const {
      bloodGroup,
      latitude,
      longitude,
      maxDistanceKm = 50,
      hourOfDay = new Date().getHours(),
      topK = 10,
    } = body;

    if (
      typeof latitude !== "number" ||
      typeof longitude !== "number" ||
      !bloodGroup
    ) {
      return NextResponse.json(
        { message: "Invalid request body: latitude, longitude, and bloodGroup are required." },
        { status: 400 }
      );
    }

    // 1. Fetch available donors from database or fallback to seed donors
    let donors: any[] = [];
    try {
      donors = await prisma.donor.findMany({
        where: {
          bloodGroup: { not: undefined },
          available: true,
          latitude: { not: null },
          longitude: { not: null },
        },
        take: 300,
      });
    } catch (dbErr) {
      // Graceful fallback to seed donors if MongoDB is not reachable
      donors = getSeedDonors();
    }

    if (!donors || donors.length === 0) {
      donors = getSeedDonors();
    }

    // 2. Attempt to rank donors using the AI/ML Microservice
    try {
      const mlApiRes = await fetch("http://127.0.0.1:8000/api/ml/donor-response/rank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bloodGroup,
          latitude,
          longitude,
          maxDistanceKm,
          hourOfDay,
          topK,
          donors,
        }),
        cache: "no-store",
      });

      if (mlApiRes.ok) {
        const mlData = await mlApiRes.json();
        return NextResponse.json({
          success: true,
          model: "AI/ML Logistic Regression & Gradient Boosting",
          evaluation: "Recall@K & Precision@K",
          count: mlData.count,
          total_compatible: mlData.total_compatible,
          donors: mlData.ranked_donors,
        });
      }
    } catch (mlErr) {
      // Fall through to algorithmic backup ranking if ML API is not currently active
    }

    // 3. Fallback Heuristic Ranking if ML Service is offline
    const candidates = donors
      .filter((d) => isCompatible(d.bloodGroup, bloodGroup))
      .map((d) => {
        const distanceKm = haversineKm(latitude, longitude, d.latitude, d.longitude);
        return { donor: d, distanceKm } as const;
      })
      .filter((x) => x.distanceKm != null && (x.distanceKm as number) <= maxDistanceKm);

    const now = Date.now();
    const scored = candidates.map((c) => {
      const lastDon = c.donor.lastDonation || c.donor.last_donation;
      const daysSince = lastDon ? Math.max(0, (now - new Date(lastDon).getTime()) / (1000 * 60 * 60 * 24)) : (c.donor.daysSinceLastDonation ?? 999);
      const pastResp = c.donor.pastResponseRate ?? 0.6;
      
      // Proximity & recency score
      const distScore = Math.max(0, 1 - (c.distanceKm as number) / maxDistanceKm);
      const recencyScore = daysSince >= 90 ? 1.0 : Math.max(0.1, daysSince / 90.0);
      const compositeScore = 0.5 * distScore + 0.3 * pastResp + 0.2 * recencyScore;

      return {
        donorId: c.donor.id,
        name: c.donor.name || `${c.donor.firstName || ''} ${c.donor.lastName || ''}`.trim(),
        bloodGroup: c.donor.bloodGroup,
        phone: c.donor.phone,
        email: c.donor.email,
        city: c.donor.city,
        distanceKm: Number((c.distanceKm as number).toFixed(2)),
        daysSinceLastDonation: Math.round(daysSince),
        pastResponseRate: pastResp,
        responseProbability: Number(compositeScore.toFixed(4)),
        eligibilityStatus: daysSince >= 90 ? "Eligible" : `Cooldown (${Math.round(90 - daysSince)}d left)`,
        recommendationTier: compositeScore >= 0.65 ? "Highly Recommended" : "Recommended"
      };
    });

    scored.sort((a, b) => b.responseProbability - a.responseProbability);
    const topMatches = scored.slice(0, topK);

    return NextResponse.json({
      success: true,
      model: "Heuristic Fallback Ranking",
      count: topMatches.length,
      donors: topMatches,
    });
  } catch (err: any) {
    return NextResponse.json({ message: "MATCH Error", error: err?.message || err }, { status: 500 });
  }
};
