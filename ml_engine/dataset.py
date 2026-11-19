"""
dataset.py
Realistic dataset generator for the Blood Donation Prediction and Management System.
Generates comprehensive historical and operational data across all 6 ML task domains:
1. Donor Response to Emergency Alerts
2. Blood Demand Time-Series (by blood group & city)
3. Donor Appointment Attendance (No-Show)
4. Request Text & Anomaly Data (Urgency & Fake Detection)
5. Donor Retention & Churn Records
6. Donor Behavioral Profile Vectors (for Segmentation)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

INDIAN_CITIES = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "pop_factor": 1.5},
    "Delhi": {"lat": 28.7041, "lon": 77.1025, "pop_factor": 1.4},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "pop_factor": 1.2},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "pop_factor": 1.0},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "pop_factor": 0.9},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "pop_factor": 1.1},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "pop_factor": 0.8},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "pop_factor": 0.8},
}

BLOOD_GROUPS = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
# Typical Indian demographic distribution (normalized)
_bg_w = np.array([0.37, 0.03, 0.22, 0.02, 0.31, 0.02, 0.07, 0.01], dtype=float)
BLOOD_GROUP_WEIGHTS = list(_bg_w / _bg_w.sum())

FIRST_NAMES = [
    "Aarav", "Aditi", "Amit", "Ananya", "Arjun", "Deepak", "Divya", "Gaurav", 
    "Ishaan", "Kavita", "Manish", "Neha", "Pooja", "Rahul", "Riya", "Rohan", 
    "Sachin", "Sneha", "Sunil", "Tanvi", "Varun", "Vikas", "Priya", "Sanjay"
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Reddy", "Nair", "Iyer", "Mehta", "Singh",
    "Kumar", "Gupta", "Joshi", "Deshmukh", "Choudhury", "Bose", "Rao", "Mishra"
]

def generate_donors(n=600):
    """Generate realistic donor pool."""
    donors = []
    city_names = list(INDIAN_CITIES.keys())
    
    for i in range(n):
        city = random.choice(city_names)
        city_coords = INDIAN_CITIES[city]
        
        # Add random scatter around city center (~5-15 km radius)
        lat_scatter = np.random.normal(0, 0.08)
        lon_scatter = np.random.normal(0, 0.08)
        
        bg = np.random.choice(BLOOD_GROUPS, p=BLOOD_GROUP_WEIGHTS)
        age = int(np.random.normal(32, 10))
        age = max(18, min(65, age))
        
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        
        # Days since last donation (90 days minimum cooldown, some never donated)
        never_donated = random.random() < 0.15
        if never_donated:
            days_since = 999
            total_donations = 0
            last_date = None
        else:
            days_since = int(np.random.exponential(120) + 45)
            total_donations = int(np.random.poisson(3) + 1)
            last_date = (datetime.now() - timedelta(days=days_since)).isoformat()
            
        past_response_rate = float(np.clip(np.random.beta(5, 3), 0.1, 1.0)) if total_donations > 0 else 0.5
        
        donors.append({
            "id": f"donor_{i+1:04d}",
            "name": f"{first} {last}",
            "email": f"{first.lower()}.{last.lower()}{i+1}@example.com",
            "phone": f"+91 98{np.random.randint(10000000, 99999999)}",
            "bloodGroup": bg,
            "age": age,
            "gender": random.choice(["Male", "Female"]),
            "city": city,
            "latitude": float(city_coords["lat"] + lat_scatter),
            "longitude": float(city_coords["lon"] + lon_scatter),
            "daysSinceLastDonation": days_since,
            "lastDonation": last_date,
            "totalDonations": total_donations,
            "pastResponseRate": round(past_response_rate, 3),
            "available": random.random() > 0.1
        })
    return pd.DataFrame(donors)

def generate_response_training_data(n_samples=2500):
    """
    Generate historical alert-response instances for Feature 1 (Donor Response Prediction).
    Features: distance (km), days_since_last_donation, past_response_rate, time_of_day_hour.
    Target: responded (0 or 1).
    """
    records = []
    for _ in range(n_samples):
        # Distance to alert location: mostly within 20km, occasional farther
        distance_km = float(np.random.exponential(7.0) + 0.5)
        
        # Days since last donation
        days_since = float(np.random.choice([
            np.random.uniform(10, 80),   # Ineligible or recent: lower response
            np.random.uniform(90, 200),  # Prime eligible: higher response
            np.random.uniform(200, 500)  # Lapsed: moderate response
        ], p=[0.25, 0.55, 0.20]))
        
        # Past response rate (0 to 1)
        past_response_rate = float(np.clip(np.random.beta(4, 3), 0.05, 0.98))
        
        # Hour of the alert (0-23)
        # Night alerts (22-06) have lower response; daytime/evening higher
        p_hours = np.array([
            0.02, 0.01, 0.01, 0.01, 0.01, 0.02, # 00-05
            0.04, 0.06, 0.08, 0.08, 0.07, 0.07, # 06-11
            0.06, 0.06, 0.07, 0.07, 0.07, 0.06, # 12-17
            0.05, 0.04, 0.03, 0.02, 0.02, 0.01  # 18-23
        ], dtype=float)
        p_hours /= p_hours.sum()
        hour = int(np.random.choice(range(24), p=p_hours))
        
        # Ground truth probabilistic latent response generator
        # 1. Distance penalty (steep drop after 15km)
        dist_effect = -0.12 * distance_km
        
        # 2. Eligibility benefit (peak when days_since >= 90)
        if days_since < 90:
            recency_effect = -2.0  # Physically cannot donate yet
        elif 90 <= days_since <= 210:
            recency_effect = 1.2   # Ideal window
        else:
            recency_effect = 0.3   # Slightly lapsed
            
        # 3. Habitual responsiveness
        response_rate_effect = 3.0 * (past_response_rate - 0.5)
        
        # 4. Time of day effect (sleeping hours 22-06 penalty)
        if 22 <= hour or hour <= 6:
            tod_effect = -1.5
        elif 9 <= hour <= 18:
            tod_effect = 0.8
        else:
            tod_effect = 0.0
            
        logit = -0.5 + dist_effect + recency_effect + response_rate_effect + tod_effect + np.random.normal(0, 0.5)
        prob = 1.0 / (1.0 + np.exp(-logit))
        responded = 1 if np.random.rand() < prob else 0
        
        records.append({
            "distance_km": round(distance_km, 2),
            "days_since_last_donation": round(days_since, 1),
            "past_response_rate": round(past_response_rate, 3),
            "time_of_day_hour": hour,
            "responded": responded,
            "true_prob": round(prob, 4)
        })
    return pd.DataFrame(records)

def generate_demand_time_series(days=500):
    """
    Generate realistic multi-city, multi-blood-group daily blood demand time series.
    Includes day-of-week seasonality (higher mid-week surgeries), annual season (monsoon dengue spikes, summer blood shortages),
    and city population scalers.
    """
    start_date = datetime.now() - timedelta(days=days)
    records = []
    
    cities = list(INDIAN_CITIES.keys())
    
    for d in range(days):
        current_date = start_date + timedelta(days=d)
        day_of_week = current_date.weekday() # 0 = Monday, 6 = Sunday
        day_of_year = current_date.timetuple().tm_yday
        month = current_date.month
        
        # Weekly pattern: Elective surgeries peak Tue-Thu, drop on weekends
        weekly_factor = 1.2 if day_of_week in [1, 2, 3] else (0.75 if day_of_week == 6 else 1.0)
        
        # Seasonal pattern: Dengue / monsoon season (Jul-Oct, months 7-10) raises demand
        seasonal_factor = 1.35 if month in [7, 8, 9, 10] else (1.15 if month in [5, 6] else 0.95)
        
        # Base annual upward trend (4% growth / year)
        trend_factor = 1.0 + 0.04 * (d / 365.0)
        
        for city in cities:
            city_pop = INDIAN_CITIES[city]["pop_factor"]
            
            for bg, bg_w in zip(BLOOD_GROUPS, BLOOD_GROUP_WEIGHTS):
                # Rare types (O-, AB-) have lower absolute volume, but higher volatility
                base_units = 18.0 * city_pop * (bg_w / 0.3) * weekly_factor * seasonal_factor * trend_factor
                
                # Add Poisson / Gaussian noise
                units_demanded = max(1, int(np.random.poisson(base_units) + np.random.normal(0, 1.2)))
                
                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "city": city,
                    "blood_group": bg,
                    "day_of_week": day_of_week,
                    "day_of_year": day_of_year,
                    "month": month,
                    "units_demanded": units_demanded
                })
                
    return pd.DataFrame(records)

def generate_noshow_training_data(n_samples=2000):
    """
    Generate historical donation appointment attendance data for Feature 3 (No-Show Prediction).
    Features:
      - total_past_donations: int
      - past_noshow_rate: float (0.0 to 1.0)
      - urgency: ['Emergency', 'Urgent', 'Routine']
      - distance_km: float
      - lead_time_days: int (days between booking and appointment)
      - reminder_sent: 0 or 1
      - appointment_hour: 8 to 19
    Target:
      - no_show: 1 (did not show up) or 0 (attended)
    """
    records = []
    urgency_map = {"Emergency": 2, "Urgent": 1, "Routine": 0}
    
    for _ in range(n_samples):
        total_donations = int(np.random.exponential(4))
        past_noshow_rate = float(np.random.beta(2, 6)) if total_donations > 0 else 0.3
        urgency = random.choice(["Emergency", "Urgent", "Routine"])
        distance_km = float(np.random.exponential(9.0) + 1.0)
        lead_time_days = int(np.random.exponential(4) + 1)
        reminder_sent = 1 if random.random() < 0.75 else 0
        slot_hour = random.randint(8, 18)
        
        # Latent logit for no-show
        # Factors that INCREASE no-show: high past no-show, long lead time, far distance, routine (low urgency), no reminder
        logit = -1.2 \
            + 2.2 * past_noshow_rate \
            - 0.15 * min(total_donations, 10) \
            + 0.08 * distance_km \
            + 0.12 * min(lead_time_days, 14) \
            - 0.9 * urgency_map[urgency] \
            - 0.8 * reminder_sent \
            + (0.4 if (slot_hour < 9 or slot_hour > 17) else 0.0) \
            + np.random.normal(0, 0.4)
            
        prob = 1.0 / (1.0 + np.exp(-logit))
        no_show = 1 if np.random.rand() < prob else 0
        
        records.append({
            "total_past_donations": total_donations,
            "past_noshow_rate": round(past_noshow_rate, 3),
            "urgency": urgency,
            "distance_km": round(distance_km, 2),
            "lead_time_days": lead_time_days,
            "reminder_sent": reminder_sent,
            "slot_hour": slot_hour,
            "no_show": no_show
        })
    return pd.DataFrame(records)

def generate_request_nlp_data(n_samples=1800):
    """
    Generate realistic blood requests for Feature 4:
    - Urgency: 'Critical', 'Urgent', 'Routine'
    - Fraud/Fake indicators: 'Legitimate', 'Suspicious', 'Fake'
    Features:
      - text: request description
      - frequency_last_24h: integer requests from same phone/IP
      - units_requested: requested units
      - is_anomalous_burst: boolean
      - is_fake: 0 or 1
      - urgency_label: 0 (Routine), 1 (Urgent), 2 (Critical)
    """
    critical_templates = [
        "Patient {name} in severe ICU condition after road accident, requires {units} units of {bg} immediately at {hosp} Trauma Center. Extremely urgent!",
        "Emergency heart bypass surgery underway for 54yo at {hosp}. Active massive hemorrhage, urgently require {units} units {bg}. Please reach blood bank ASAP.",
        "Postpartum hemorrhage in delivery room at {hosp}. Hemoglobin dropped to 4.2. Need {units} units {bg} right now! Life threatening emergency.",
        "Severe dengue patient with platelet count critical at 12,000, multi-organ bleeding risk at {hosp}. Need {units} units {bg} within 2 hours.",
        "CRITICAL CODE RED: Multiple casualty trauma at {hosp}. Immediate requirement for {units} bags of {bg} blood. Contact duty doctor {phone}.",
        "Emergency pediatric leukemia crisis with acute severe anemia at {hosp}. Vital signs unstable. Need {units} units {bg} urgently."
    ]
    
    urgent_templates = [
        "Need {units} units of {bg} blood for planned chemotherapy session tomorrow morning at {hosp}. Patient is stable but needs blood ready.",
        "Scheduled orthopedic hip replacement surgery on Friday for patient at {hosp}. Requesting {units} units of {bg} to be reserved.",
        "Thalassemia major patient monthly transfusion cycle at {hosp}. Looking for {units} units of {bg} within the next 24-48 hours.",
        "Patient admitted with severe jaundice and low hemoglobin at {hosp}. Doctor advised transfusion within 24 hours. Need {units} units {bg}.",
        "Gastrointestinal bleed stabilized under observation at {hosp}. Doctor requested backup {units} units {bg} blood before endoscopy."
    ]
    
    routine_templates = [
        "Routine blood replacement for elective gallbladder removal next week at {hosp}. Looking for voluntary donors for {units} units {bg}.",
        "Planning voluntary blood camp at community center. Need donors of all groups, especially {bg}, for general stock replenishment.",
        "Scheduled elective hernia surgery next Tuesday at {hosp}. Requesting {units} units {bg} blood backup as per hospital protocol.",
        "Family member admitted for routine diagnostic laparoscopy next week. Hospital requested {units} units {bg} donor replacement."
    ]
    
    fake_suspicious_templates = [
        "Selling genuine certified {bg} blood packets at cheap wholesale price. Call {phone} for immediate delivery at home, no doctor prescription needed.",
        "Urgently paying cash reward of 10,000 rupees per bottle of {bg} blood! Fast money guaranteed, contact WhatsApp {phone} immediately.",
        "Need 85 units of {bg} blood delivered to private residence address. Will transfer advance money via UPI. Contact {phone}.",
        "Official blood broker service. Guaranteed pure blood bags {bg} available for sale. Pay advance deposit to confirm dispatch.",
        "TEST TEST 12345 blood need quick money lottery winner prize free blood bag donation cash promo discount {phone}",
        "Need 50 units {bg} urgent urgently urgently immediately send money urgent urgent help cash advance."
    ]
    
    hospitals = [
        "AIIMS New Delhi", "Apollo Hospital", "Fortis Memorial", "Manipal Hospital", 
        "Lilavati Hospital", "Christian Medical College", "Narayana Health", "Tata Memorial"
    ]
    
    records = []
    
    # 1. Genuine Critical (~35%)
    for _ in range(int(n_samples * 0.35)):
        bg = np.random.choice(BLOOD_GROUPS)
        units = random.randint(1, 4)
        tmpl = random.choice(critical_templates)
        text = tmpl.format(
            name=f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            units=units,
            bg=bg,
            hosp=random.choice(hospitals),
            phone=f"98{np.random.randint(10000000, 99999999)}"
        )
        freq = random.choices([1, 2, 3], weights=[0.8, 0.15, 0.05])[0]
        records.append({
            "text": text,
            "units_requested": units,
            "frequency_last_24h": freq,
            "urgency": "Critical",
            "urgency_label": 2,
            "is_fake": 0
        })
        
    # 2. Genuine Urgent (~35%)
    for _ in range(int(n_samples * 0.35)):
        bg = np.random.choice(BLOOD_GROUPS)
        units = random.randint(1, 3)
        tmpl = random.choice(urgent_templates)
        text = tmpl.format(
            units=units,
            bg=bg,
            hosp=random.choice(hospitals),
            phone=f"98{np.random.randint(10000000, 99999999)}"
        )
        freq = random.choices([1, 2], weights=[0.9, 0.1])[0]
        records.append({
            "text": text,
            "units_requested": units,
            "frequency_last_24h": freq,
            "urgency": "Urgent",
            "urgency_label": 1,
            "is_fake": 0
        })
        
    # 3. Genuine Routine (~20%)
    for _ in range(int(n_samples * 0.20)):
        bg = np.random.choice(BLOOD_GROUPS)
        units = random.randint(1, 2)
        tmpl = random.choice(routine_templates)
        text = tmpl.format(
            units=units,
            bg=bg,
            hosp=random.choice(hospitals)
        )
        records.append({
            "text": text,
            "units_requested": units,
            "frequency_last_24h": 1,
            "urgency": "Routine",
            "urgency_label": 0,
            "is_fake": 0
        })
        
    # 4. Fake / Scam / Spam Requests (~10%)
    for _ in range(int(n_samples * 0.10)):
        bg = np.random.choice(BLOOD_GROUPS)
        units = random.choice([25, 40, 60, 100, 1])  # Anomalous volume or broker text
        tmpl = random.choice(fake_suspicious_templates)
        text = tmpl.format(
            units=units,
            bg=bg,
            phone=f"99{np.random.randint(10000000, 99999999)}"
        )
        # Frequent burst from spammer
        freq = random.randint(8, 35)
        records.append({
            "text": text,
            "units_requested": units,
            "frequency_last_24h": freq,
            "urgency": random.choice(["Critical", "Urgent", "Routine"]),
            "urgency_label": random.choice([0, 1, 2]),
            "is_fake": 1
        })
        
    random.shuffle(records)
    return pd.DataFrame(records)

def generate_retention_training_data(n_samples=2200):
    """
    Generate donor lifecycle data for Feature 5 (Retention & Churn Prediction).
    Features:
      - donation_frequency: donations per year
      - recency_days: days since last donation
      - total_donations: lifetime count
      - response_rate: past alert response rate
      - tenure_months: months since first registration
      - app_logins_per_month: engagement index
      - distance_to_center_km: donor convenience factor
    Target:
      - churned: 1 (stopped donating / dropped out) or 0 (active recurring donor)
    """
    records = []
    for _ in range(n_samples):
        tenure_months = int(np.random.exponential(18) + 6)
        frequency = float(np.random.gamma(2, 0.8))  # 0 to 4 times/yr
        recency_days = int(np.random.exponential(140) + 30)
        total_donations = max(1, int(frequency * (tenure_months / 12.0) + np.random.normal(0, 1)))
        response_rate = float(np.clip(np.random.beta(4, 3), 0.05, 1.0))
        app_logins = int(np.random.poisson(3.5))
        distance_km = float(np.random.exponential(8.0) + 1.0)
        
        # Churn probability calculation
        # High recency, low frequency, low response, low logins, far distance -> High Churn
        logit = -2.0 \
            + 0.008 * (recency_days - 120) \
            - 1.1 * frequency \
            - 2.4 * (response_rate - 0.4) \
            - 0.3 * app_logins \
            + 0.04 * distance_km \
            + np.random.normal(0, 0.4)
            
        prob = 1.0 / (1.0 + np.exp(-logit))
        churned = 1 if np.random.rand() < prob else 0
        
        records.append({
            "donation_frequency": round(frequency, 2),
            "recency_days": recency_days,
            "total_donations": total_donations,
            "response_rate": round(response_rate, 3),
            "tenure_months": tenure_months,
            "app_logins_per_month": app_logins,
            "distance_to_center_km": round(distance_km, 2),
            "churned": churned,
            "churn_prob": round(prob, 4)
        })
    return pd.DataFrame(records)

def generate_segmentation_data(n_samples=1500):
    """
    Generate donor feature vectors for Feature 6 (Donor Segmentation using K-Means).
    Features:
      - age (18-65)
      - distance_km (0.5 to 35 km)
      - annual_donation_frequency (0.5 to 5.0)
      - emergency_responsiveness (0.0 to 1.0)
      - tenure_years (0.5 to 10.0)
    """
    records = []
    
    # 4 distinct natural behavioral personas:
    # 1. Super Champions (~25%): Mid age, high frequency, very high responsiveness, close
    # 2. Young First-Timers (~30%): 18-26 age, low frequency, moderate responsiveness
    # 3. Emergency Volunteers (~25%): Moderate frequency, exceptionally high emergency response
    # 4. Inactive / Distant (~20%): Older or far away, low frequency, low responsiveness
    
    for _ in range(int(n_samples * 0.25)):
        records.append({
            "age": int(np.random.normal(36, 6)),
            "distance_km": float(np.random.normal(4.5, 2.0)),
            "annual_donation_frequency": float(np.random.normal(3.8, 0.6)),
            "emergency_responsiveness": float(np.random.normal(0.88, 0.08)),
            "tenure_years": float(np.random.normal(4.5, 1.2))
        })
        
    for _ in range(int(n_samples * 0.30)):
        records.append({
            "age": int(np.random.normal(22, 2.5)),
            "distance_km": float(np.random.normal(8.0, 3.5)),
            "annual_donation_frequency": float(np.random.normal(1.2, 0.4)),
            "emergency_responsiveness": float(np.random.normal(0.55, 0.15)),
            "tenure_years": float(np.random.normal(1.0, 0.4))
        })
        
    for _ in range(int(n_samples * 0.25)):
        records.append({
            "age": int(np.random.normal(42, 8)),
            "distance_km": float(np.random.normal(6.5, 2.5)),
            "annual_donation_frequency": float(np.random.normal(1.8, 0.5)),
            "emergency_responsiveness": float(np.random.normal(0.92, 0.06)),
            "tenure_years": float(np.random.normal(3.5, 1.5))
        })
        
    for _ in range(int(n_samples * 0.20)):
        records.append({
            "age": int(np.random.normal(50, 7)),
            "distance_km": float(np.random.normal(18.0, 6.0)),
            "annual_donation_frequency": float(np.random.normal(0.6, 0.3)),
            "emergency_responsiveness": float(np.random.normal(0.25, 0.12)),
            "tenure_years": float(np.random.normal(5.0, 2.0))
        })
        
    df = pd.DataFrame(records)
    # Clip to realistic bounds
    df["age"] = df["age"].clip(18, 65)
    df["distance_km"] = df["distance_km"].clip(0.5, 50.0)
    df["annual_donation_frequency"] = df["annual_donation_frequency"].clip(0.1, 6.0)
    df["emergency_responsiveness"] = df["emergency_responsiveness"].clip(0.05, 1.0)
    df["tenure_years"] = df["tenure_years"].clip(0.1, 15.0)
    return df

if __name__ == "__main__":
    print("Generating sample datasets...")
    donors = generate_donors(50)
    print(f"Generated {len(donors)} donors.")
    responses = generate_response_training_data(100)
    print(f"Generated {len(responses)} response training samples.")
    ts = generate_demand_time_series(30)
    print(f"Generated {len(ts)} demand time series rows.")
    print("Dataset generation verified successfully!")
