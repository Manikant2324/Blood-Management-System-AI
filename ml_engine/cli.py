"""
cli.py
Command-Line Interface for the Blood Donation ML Engine.
Allows direct execution of predictions, evaluations, and benchmarks.
"""

import argparse
import json
import sys
import os

# Ensure workspace root is always in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_engine.api import (
    models,
    cached_metrics,
    seed_donors,
    load_all_models
)

def main():
    parser = argparse.ArgumentParser(description="Blood Donation AI/ML CLI Engine")
    subparsers = parser.add_subparsers(dest="command", help="ML Feature Command")

    # Metrics
    subparsers.add_parser("metrics", help="Print all model evaluation metrics")

    # Donor Response Rank
    rank_p = subparsers.add_parser("rank", help="Rank donors for an emergency request")
    rank_p.add_argument("--blood-group", required=True, help="Required blood group (e.g. O+, A-)")
    rank_p.add_argument("--lat", type=float, required=True, help="Request latitude")
    rank_p.add_argument("--lon", type=float, required=True, help="Request longitude")
    rank_p.add_argument("--hour", type=int, default=14, help="Hour of request (0-23)")
    rank_p.add_argument("--top-k", type=int, default=5, help="Number of top donors to return")

    # Demand Forecast
    fc_p = subparsers.add_parser("forecast", help="Forecast future blood demand")
    fc_p.add_argument("--city", default="Delhi", help="City name")
    fc_p.add_argument("--blood-group", default="O+", help="Blood group")
    fc_p.add_argument("--days", type=int, default=14, help="Forecast horizon days")

    # No-Show Predict
    ns_p = subparsers.add_parser("noshow", help="Predict appointment no-show probability")
    ns_p.add_argument("--donations", type=int, default=2, help="Total past donations")
    ns_p.add_argument("--noshow-rate", type=float, default=0.15, help="Past no-show rate")
    ns_p.add_argument("--urgency", default="Urgent", help="Appointment urgency")
    ns_p.add_argument("--distance", type=float, default=8.0, help="Distance to clinic in km")
    ns_p.add_argument("--lead-time", type=int, default=2, help="Days until appointment")

    # Evaluate Request (Urgency & Fraud)
    ev_p = subparsers.add_parser("evaluate-request", help="Evaluate request text for urgency and fraud")
    ev_p.add_argument("--text", required=True, help="Patient condition / request message")
    ev_p.add_argument("--units", type=int, default=2, help="Blood units requested")
    ev_p.add_argument("--freq", type=int, default=1, help="Requests in past 24 hours")
    ev_p.add_argument("--hospital", default="AIIMS", help="Hospital name")

    # Retention Reminder
    ret_p = subparsers.add_parser("reminder", help="Predict churn and generate smart reminder")
    ret_p.add_argument("--donor-id", default="donor_001", help="Donor ID")
    ret_p.add_argument("--name", default="Rahul", help="Donor Name")
    ret_p.add_argument("--blood-group", default="O+", help="Blood Group")
    ret_p.add_argument("--days-since", type=int, default=100, help="Days since last donation")
    ret_p.add_argument("--frequency", type=float, default=2.0, help="Annual donation frequency")

    # Segmentation
    seg_p = subparsers.add_parser("segment", help="Predict donor persona segment")
    seg_p.add_argument("--age", type=int, default=30, help="Donor age")
    seg_p.add_argument("--distance", type=float, default=6.0, help="Distance to clinic (km)")
    seg_p.add_argument("--frequency", type=float, default=3.0, help="Annual donations")
    seg_p.add_argument("--responsiveness", type=float, default=0.85, help="Response rate")
    seg_p.add_argument("--tenure", type=float, default=3.0, help="Tenure in years")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    load_all_models()

    if args.command == "metrics":
        print(json.dumps(cached_metrics, indent=2))

    elif args.command == "rank":
        model = models["donor_response"]
        res = model.rank_donors(
            donors=seed_donors,
            req_lat=args.lat,
            req_lon=args.lon,
            req_blood_group=args.blood_group,
            req_hour=args.hour,
            top_k=args.top_k
        )
        print(json.dumps(res, indent=2))

    elif args.command == "forecast":
        model = models["demand_forecasting"]
        res = model.forecast_future(city=args.city, blood_group=args.blood_group, days_ahead=args.days)
        print(json.dumps(res, indent=2))

    elif args.command == "noshow":
        model = models["noshow_prediction"]
        res = model.predict_appointment_risk(
            total_past_donations=args.donations,
            past_noshow_rate=args.noshow_rate,
            urgency=args.urgency,
            distance_km=args.distance,
            lead_time_days=args.lead_time
        )
        print(json.dumps(res, indent=2))

    elif args.command == "evaluate-request":
        model = models["fake_request_detector"]
        res = model.analyze_request(
            text=args.text,
            units_requested=args.units,
            frequency_last_24h=args.freq,
            hospital_name=args.hospital
        )
        print(json.dumps(res, indent=2))

    elif args.command == "reminder":
        model = models["retention_reminders"]
        res = model.generate_smart_reminder(
            donor_id=args.donor_id,
            name=args.name,
            blood_group=args.blood_group,
            days_since_last_donation=args.days_since,
            donation_frequency=args.frequency
        )
        print(json.dumps(res, indent=2))

    elif args.command == "segment":
        model = models["donor_segmentation"]
        res = model.predict_segment({
            "age": args.age,
            "distance_km": args.distance,
            "annual_donation_frequency": args.frequency,
            "emergency_responsiveness": args.responsiveness,
            "tenure_years": args.tenure
        })
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
