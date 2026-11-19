import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const ML_ENGINE_URL = process.env.ML_ENGINE_URL || "http://127.0.0.1:8000";

function getLocalMetrics() {
  try {
    const p1 = path.resolve(process.cwd(), "..", "ml_engine", "saved_models", "metrics.json");
    if (fs.existsSync(p1)) return JSON.parse(fs.readFileSync(p1, "utf-8"));
    const p2 = path.resolve(process.cwd(), "ml_engine", "saved_models", "metrics.json");
    if (fs.existsSync(p2)) return JSON.parse(fs.readFileSync(p2, "utf-8"));
  } catch (e) {
    // ignore
  }
  return null;
}

export async function GET(
  request: NextRequest,
  { params }: { params: { slug: string[] } }
) {
  const pathSuffix = params.slug.join("/");
  const targetUrl = `${ML_ENGINE_URL}/api/ml/${pathSuffix}${request.nextUrl.search}`;

  try {
    const upstreamRes = await fetch(targetUrl, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
    });

    if (upstreamRes.ok) {
      const data = await upstreamRes.json();
      return NextResponse.json(data);
    }
  } catch (err) {
    // Fallback if ML microservice is not running
  }

  // Handle local metrics fallback
  if (pathSuffix === "metrics") {
    const local = getLocalMetrics();
    if (local) {
      return NextResponse.json({ success: true, metrics: local, fallback: true });
    }
  }

  return NextResponse.json(
    {
      error: "ML Engine service unavailable",
      tip: "Ensure the Python ML microservice is running via `python ml_engine/api.py` or `uvicorn ml_engine.api:app --port 8000`",
    },
    { status: 503 }
  );
}

export async function POST(
  request: NextRequest,
  { params }: { params: { slug: string[] } }
) {
  const pathSuffix = params.slug.join("/");
  const targetUrl = `${ML_ENGINE_URL}/api/ml/${pathSuffix}`;

  try {
    const body = await request.json();
    const upstreamRes = await fetch(targetUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    if (upstreamRes.ok) {
      const data = await upstreamRes.json();
      return NextResponse.json(data);
    }
  } catch (err) {
    // Fallback
  }

  return NextResponse.json(
    {
      error: "ML Engine service unavailable",
      tip: "Ensure the Python ML microservice is running on port 8000",
    },
    { status: 503 }
  );
}
