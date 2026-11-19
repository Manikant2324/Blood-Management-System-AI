import prisma from "@/app/libs/prismadb"
import { NextRequest, NextResponse } from "next/server"
import fs from "fs"
import path from "path"

function getFallbackSeedDonors() {
  try {
    const p1 = path.resolve(process.cwd(), "..", "ml_engine", "saved_models", "seed_donors.json");
    if (fs.existsSync(p1)) return JSON.parse(fs.readFileSync(p1, "utf-8"));
    const p2 = path.resolve(process.cwd(), "ml_engine", "saved_models", "seed_donors.json");
    if (fs.existsSync(p2)) return JSON.parse(fs.readFileSync(p2, "utf-8"));
  } catch (e) {
    // ignore
  }
  return [];
}

export const POST = async (request: NextRequest, response: NextResponse) => {
    try {
        const body = await request.json();
        const { firstName, lastName, phone, email, bloodGroup, age, address, state, city, gender, latitude, longitude, lastDonation, available, role } = body;

        const newDonor = await prisma.donor.create({
            data: {
                firstName,
                lastName,
                phone,
                email,
                bloodGroup,
                age,
                address,
                state,
                city,
                gender,
                latitude,
                longitude,
                lastDonation: lastDonation ? new Date(lastDonation) : undefined,
                available: typeof available === 'boolean' ? available : undefined,
                role: role || undefined
            }
        });
        return NextResponse.json(newDonor);
    } catch (err) {
        return NextResponse.json({ message: "POST Error", err }, { status: 500 })
    }
}

export const GET = async (request: NextRequest) => {
    try {
        const { searchParams } = new URL(request.url);
        const page = parseInt(searchParams.get("page") || "1");
        const limit = parseInt(searchParams.get("limit") || "20");
        const bloodGroup = searchParams.get("bloodGroup");
        const skip = (page - 1) * limit;

        const where = bloodGroup 
          ? { bloodGroup, available: true }
          : { available: true };

        let Donors: any[] = [];
        let total = 0;

        try {
          const res = await Promise.all([
            prisma.donor.findMany({
              where,
              skip,
              take: limit,
              orderBy: { id: 'desc' },
              select: {
                id: true,
                firstName: true,
                lastName: true,
                phone: true,
                email: true,
                bloodGroup: true,
                age: true,
                address: true,
                state: true,
                city: true,
                gender: true,
              },
            }),
            prisma.donor.count({ where }),
          ]);
          Donors = res[0];
          total = res[1];
        } catch (dbErr) {
          // Graceful fallback to seed donors when MongoDB is not connected
          const allSeeds = getFallbackSeedDonors();
          const filtered = bloodGroup 
            ? allSeeds.filter((d: any) => d.bloodGroup === bloodGroup && d.available)
            : allSeeds.filter((d: any) => d.available);
          total = filtered.length;
          Donors = filtered.slice(skip, skip + limit).map((d: any) => ({
            id: d.id,
            firstName: d.name ? d.name.split(" ")[0] : (d.firstName || "Donor"),
            lastName: d.name ? d.name.split(" ").slice(1).join(" ") : (d.lastName || ""),
            phone: d.phone,
            email: d.email,
            bloodGroup: d.bloodGroup,
            age: d.age,
            address: d.city,
            state: "State",
            city: d.city,
            gender: d.gender,
          }));
        }

        const response = NextResponse.json({
            message: "Donors fetched successfully",
            success: true,
            Donors,
            pagination: {
              page,
              limit,
              total,
              totalPages: Math.ceil(total / limit) || 1,
            },
          }, {
            headers: {
              'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=120',
            },
          });      
          return response;
    } catch (err) {
        return NextResponse.json({ message: "GET Error", err }, { status: 500 })
    }
}