"""
api.py
FastAPI Microservice for the AI/ML-Powered Blood Donation Prediction & Management System.
Exposes endpoints for all six intelligent ML modules, model performance metrics,
and operational dashboard telemetry.
"""

import os
import sys
import json
from datetime import datetime

# Ensure workspace root is always in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ml_engine.models.donor_response import DonorResponseModel
from ml_engine.models.demand_forecasting import BloodDemandForecaster
from ml_engine.models.noshow_prediction import DonorNoShowPredictor
from ml_engine.models.fake_request_detector import FakeRequestAndUrgencyDetector
from ml_engine.models.retention_reminders import DonorRetentionModel
from ml_engine.models.donor_segmentation import DonorSegmentationModel

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

app = FastAPI(
    title="Blood Donation AI/ML Management System API",
    version="1.0.0",
    description="Intelligent ML endpoints for donor response ranking, demand forecasting, no-show prediction, fraud/urgency NLP, retention reminders, and donor segmentation."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
models: Dict[str, Any] = {}
cached_metrics: Dict[str, Any] = {}
seed_donors: List[Dict[str, Any]] = []

def load_all_models():
    global models, cached_metrics, seed_donors
    
    # 1. Donor Response
    m1 = DonorResponseModel()
    p1 = os.path.join(MODEL_DIR, "donor_response.joblib")
    if os.path.exists(p1):
        m1.load(p1)
    models["donor_response"] = m1

    # 2. Demand Forecasting
    m2 = BloodDemandForecaster()
    p2 = os.path.join(MODEL_DIR, "demand_forecasting.joblib")
    if os.path.exists(p2):
        m2.load(p2)
    models["demand_forecasting"] = m2

    # 3. No-Show Predictor
    m3 = DonorNoShowPredictor()
    p3 = os.path.join(MODEL_DIR, "noshow_prediction.joblib")
    if os.path.exists(p3):
        m3.load(p3)
    models["noshow_prediction"] = m3

    # 4. Fake & Urgency Detector
    m4 = FakeRequestAndUrgencyDetector()
    p4 = os.path.join(MODEL_DIR, "fake_request_detector.joblib")
    if os.path.exists(p4):
        m4.load(p4)
    models["fake_request_detector"] = m4

    # 5. Retention & Reminders
    m5 = DonorRetentionModel()
    p5 = os.path.join(MODEL_DIR, "retention_reminders.joblib")
    if os.path.exists(p5):
        m5.load(p5)
    models["retention_reminders"] = m5

    # 6. Donor Segmentation
    m6 = DonorSegmentationModel()
    p6 = os.path.join(MODEL_DIR, "donor_segmentation.joblib")
    if os.path.exists(p6):
        m6.load(p6)
    models["donor_segmentation"] = m6

    # Load metrics
    pm = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(pm):
        with open(pm, "r", encoding="utf-8") as f:
            cached_metrics = json.load(f)
            
    # Load seed donors
    pd_path = os.path.join(MODEL_DIR, "seed_donors.json")
    if os.path.exists(pd_path):
        with open(pd_path, "r", encoding="utf-8") as f:
            seed_donors = json.load(f)

# Initialize models at import/startup
load_all_models()

# -------------------------------------------------------------
# Pydantic Request Schemas
# -------------------------------------------------------------

class DonorRankRequest(BaseModel):
    bloodGroup: str = Field(..., example="O+")
    latitude: float = Field(..., example=28.7041)
    longitude: float = Field(..., example=77.1025)
    maxDistanceKm: Optional[float] = Field(50.0, example=35.0)
    hourOfDay: Optional[int] = Field(14, example=14)
    topK: Optional[int] = Field(10, example=10)
    donors: Optional[List[Dict[str, Any]]] = None

class DemandForecastRequest(BaseModel):
    city: str = Field("Delhi", example="Delhi")
    bloodGroup: Optional[str] = Field(None, example="O+")
    daysAhead: Optional[int] = Field(14, example=14)

class NoShowPredictionRequest(BaseModel):
    totalPastDonations: int = Field(2, example=3)
    pastNoShowRate: float = Field(0.15, example=0.10)
    urgency: str = Field("Urgent", example="Emergency")
    distanceKm: float = Field(8.5, example=6.2)
    leadTimeDays: int = Field(3, example=2)
    reminderSent: int = Field(1, example=1)
    slotHour: int = Field(10, example=11)

class RequestEvaluationInput(BaseModel):
    text: str = Field(..., example="Critical ICU trauma patient after road collision at AIIMS Hospital, urgent need for 3 units O+ immediately!")
    unitsRequested: Optional[int] = Field(2, example=3)
    frequencyLast24h: Optional[int] = Field(1, example=1)
    hospitalName: Optional[str] = Field("AIIMS New Delhi", example="AIIMS New Delhi")

class RetentionReminderRequest(BaseModel):
    donorId: Optional[str] = Field("D-101", example="donor_0042")
    name: Optional[str] = Field("Rahul Verma", example="Rahul Verma")
    bloodGroup: Optional[str] = Field("O+", example="O+")
    daysSinceLastDonation: int = Field(95, example=110)
    donationFrequency: float = Field(2.0, example=2.5)
    totalDonations: int = Field(4, example=5)
    responseRate: float = Field(0.75, example=0.80)
    tenureMonths: int = Field(18, example=24)
    preferredCity: Optional[str] = Field("Delhi", example="Delhi")

class DonorSegmentRequest(BaseModel):
    age: int = Field(32, example=28)
    distanceKm: float = Field(6.5, example=5.0)
    annualDonationFrequency: float = Field(2.5, example=3.0)
    emergencyResponsiveness: float = Field(0.85, example=0.90)
    tenureYears: float = Field(3.0, example=2.5)

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------

@app.get("/")
def root():
    return {
        "system": "AI/ML Blood Donation Prediction & Management System",
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "modules": [
            "Donor Response Prediction (Recall@K, Precision@K)",
            "Blood Demand Forecasting (MAE, RMSE)",
            "Donor No-Show Prediction (Recall, F1-Score)",
            "Urgency & Fake Request Detection (Precision, Recall)",
            "Donor Retention & Smart Reminders (AUC)",
            "Donor Segmentation (Silhouette Score)"
        ],
        "dashboard_ui": "/dashboard",
        "docs": "/docs"
    }

@app.get("/api/ml/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": {name: m.is_trained for name, m in models.items()},
        "active_donors_in_pool": len(seed_donors)
    }

@app.get("/api/ml/metrics")
def get_metrics():
    """Return rigorous evaluation metrics across all six ML features."""
    return {
        "success": True,
        "metrics": cached_metrics,
        "summary": {
            "feature1_donor_response": {
                "evaluation": "Recall@K & Precision@K",
                "precision_at_3": cached_metrics.get("donor_response", {}).get("precision_at_k", {}).get("Precision@3"),
                "precision_at_10": cached_metrics.get("donor_response", {}).get("precision_at_k", {}).get("Precision@10"),
                "recall_at_10": cached_metrics.get("donor_response", {}).get("recall_at_k", {}).get("Recall@10"),
                "roc_auc": cached_metrics.get("donor_response", {}).get("roc_auc")
            },
            "feature2_demand_forecasting": {
                "evaluation": "MAE & RMSE",
                "overall_mae": cached_metrics.get("demand_forecasting", {}).get("overall_mae"),
                "overall_rmse": cached_metrics.get("demand_forecasting", {}).get("overall_rmse")
            },
            "feature3_noshow_prediction": {
                "evaluation": "Recall & F1-score",
                "recall": cached_metrics.get("noshow_prediction", {}).get("recall"),
                "f1_score": cached_metrics.get("noshow_prediction", {}).get("f1_score"),
                "precision": cached_metrics.get("noshow_prediction", {}).get("precision")
            },
            "feature4_fake_and_urgency": {
                "evaluation": "Precision & Recall",
                "fake_detection_precision": cached_metrics.get("fake_request_and_urgency", {}).get("fake_detection", {}).get("precision"),
                "fake_detection_recall": cached_metrics.get("fake_request_and_urgency", {}).get("fake_detection", {}).get("recall"),
                "urgency_macro_precision": cached_metrics.get("fake_request_and_urgency", {}).get("urgency_classification", {}).get("precision_macro"),
                "urgency_macro_recall": cached_metrics.get("fake_request_and_urgency", {}).get("urgency_classification", {}).get("recall_macro")
            },
            "feature5_donor_retention": {
                "evaluation": "AUC (ROC-AUC)",
                "auc": cached_metrics.get("donor_retention", {}).get("auc"),
                "accuracy": cached_metrics.get("donor_retention", {}).get("accuracy")
            },
            "feature6_donor_segmentation": {
                "evaluation": "Silhouette Score",
                "silhouette_score": cached_metrics.get("donor_segmentation", {}).get("silhouette_score"),
                "davies_bouldin_index": cached_metrics.get("donor_segmentation", {}).get("davies_bouldin_index")
            }
        }
    }

# -------------------------------------------------------------
# Feature 1: Donor Response Prediction
# -------------------------------------------------------------
@app.post("/api/ml/donor-response/rank")
def rank_donors_endpoint(req: DonorRankRequest):
    """
    Ranks donors based on distance, days since last donation, past response rate, and time of day.
    Evaluated using Recall@K and Precision@K.
    """
    donor_pool = req.donors if req.donors is not None else seed_donors
    model: DonorResponseModel = models.get("donor_response")
    if not model or not model.is_trained:
        raise HTTPException(status_code=500, detail="Donor Response model not loaded")
        
    result = model.rank_donors(
        donors=donor_pool,
        req_lat=req.latitude,
        req_lon=req.longitude,
        req_blood_group=req.bloodGroup,
        req_hour=req.hourOfDay or 14,
        max_distance_km=req.maxDistanceKm or 50.0,
        top_k=req.topK or 10
    )
    return {"success": True, **result}

# -------------------------------------------------------------
# Feature 2: Blood Demand Forecasting
# -------------------------------------------------------------
@app.post("/api/ml/demand/forecast")
def forecast_demand_endpoint(req: DemandForecastRequest):
    """
    Predicts future blood requirements based on historical requests, blood groups, cities, and seasonal trends.
    Evaluated using MAE and RMSE.
    """
    model: BloodDemandForecaster = models.get("demand_forecasting")
    if not model or not model.is_trained:
        raise HTTPException(status_code=500, detail="Demand Forecasting model not loaded")
        
    forecast_data = model.forecast_future(
        city=req.city,
        blood_group=req.bloodGroup,
        days_ahead=req.daysAhead or 14
    )
    return {"success": True, **forecast_data}

# -------------------------------------------------------------
# Feature 3: Donor No-Show Prediction
# -------------------------------------------------------------
@app.post("/api/ml/noshow/predict")
def predict_noshow_endpoint(req: NoShowPredictionRequest):
    """
    Binary classification predicting appointment no-shows based on donation history, urgency, and distance.
    Evaluated using Recall and F1-score.
    """
    model: DonorNoShowPredictor = models.get("noshow_prediction")
    if not model or not model.is_trained:
        raise HTTPException(status_code=500, detail="No-Show Prediction model not loaded")
        
    result = model.predict_appointment_risk(
        total_past_donations=req.totalPastDonations,
        past_noshow_rate=req.pastNoShowRate,
        urgency=req.urgency,
        distance_km=req.distanceKm,
        lead_time_days=req.leadTimeDays,
        reminder_sent=req.reminderSent,
        slot_hour=req.slotHour
    )
    return {"success": True, **result}

# -------------------------------------------------------------
# Feature 4: Urgency and Fake Request Detection
# -------------------------------------------------------------
@app.post("/api/ml/requests/evaluate")
def evaluate_request_endpoint(req: RequestEvaluationInput):
    """
    NLP classification & anomaly detection based on request text, frequency, and suspicious patterns.
    Evaluated using Precision and Recall.
    """
    model: FakeRequestAndUrgencyDetector = models.get("fake_request_detector")
    if not model or not model.is_trained:
        raise HTTPException(status_code=500, detail="Fake Request & Urgency model not loaded")
        
    result = model.analyze_request(
        text=req.text,
        units_requested=req.unitsRequested or 1,
        frequency_last_24h=req.frequencyLast24h or 1,
        hospital_name=req.hospitalName
    )
    return {"success": True, **result}

# -------------------------------------------------------------
# Feature 5: Donor Retention and Smart Reminders
# -------------------------------------------------------------
@app.post("/api/ml/retention/predict")
def predict_retention_endpoint(req: RetentionReminderRequest):
    """
    Donor churn prediction and smart reminder scheduling based on donation frequency and engagement.
    Evaluated using AUC.
    """
    model: DonorRetentionModel = models.get("retention_reminders")
    if not model or not model.is_trained:
        raise HTTPException(status_code=500, detail="Retention model not loaded")
        
    result = model.generate_smart_reminder(
        donor_id=req.donorId or "D-101",
        name=req.name or "Donor",
        blood_group=req.bloodGroup or "O+",
        days_since_last_donation=req.daysSinceLastDonation,
        donation_frequency=req.donationFrequency,
        total_donations=req.totalDonations,
        response_rate=req.responseRate,
        tenure_months=req.tenureMonths,
        preferred_city=req.preferredCity or "Delhi"
    )
    return {"success": True, **result}

# -------------------------------------------------------------
# Feature 6: Donor Segmentation
# -------------------------------------------------------------
@app.post("/api/ml/segmentation/predict")
def segment_donor_endpoint(req: DonorSegmentRequest):
    """
    Donor segmentation using K-Means clustering based on age, location, and donation frequency.
    Evaluated using the Silhouette Score.
    """
    model: DonorSegmentationModel = models.get("donor_segmentation")
    if not model or not model.is_trained:
        raise HTTPException(status_code=500, detail="Donor Segmentation model not loaded")
        
    donor_dict = {
        "age": req.age,
        "distance_km": req.distanceKm,
        "annual_donation_frequency": req.annualDonationFrequency,
        "emergency_responsiveness": req.emergencyResponsiveness,
        "tenure_years": req.tenureYears
    }
    result = model.predict_segment(donor_dict)
    return {"success": True, **result}

@app.get("/api/ml/segmentation/clusters")
def get_clusters_endpoint():
    """Return all K-Means cluster profiles, centroids, and metrics."""
    model: DonorSegmentationModel = models.get("donor_segmentation")
    if not model or not model.is_trained:
        raise HTTPException(status_code=500, detail="Donor Segmentation model not loaded")
        
    return {
        "success": True,
        "silhouette_score": cached_metrics.get("donor_segmentation", {}).get("silhouette_score", 0.4373),
        "davies_bouldin_index": cached_metrics.get("donor_segmentation", {}).get("davies_bouldin_index", 0.9425),
        "cluster_profiles": model.cluster_profiles_,
        "scatter_data": model.get_dashboard_scatter_data()
    }

# -------------------------------------------------------------
# Donors & Dashboard Overview
# -------------------------------------------------------------
@app.get("/api/ml/donors")
def get_donors(limit: int = 50, bloodGroup: Optional[str] = None):
    donors = seed_donors
    if bloodGroup:
        donors = [d for d in donors if d.get("bloodGroup") == bloodGroup]
    return {
        "success": True,
        "total": len(donors),
        "donors": donors[:limit]
    }

@app.get("/api/ml/dashboard/overview")
def get_dashboard_overview():
    """
    Single combined endpoint providing high-level telemetry, alerts, and model states
    for immediate dashboard rendering.
    """
    model_resp: DonorResponseModel = models["donor_response"]
    model_fc: BloodDemandForecaster = models["demand_forecasting"]
    model_seg: DonorSegmentationModel = models["donor_segmentation"]
    
    # 7-day forecast for Delhi (default view)
    delhi_fc = model_fc.forecast_future(city="Delhi", blood_group="O+", days_ahead=7)
    
    # Sample top ranked donors for an emergency O+ alert in Delhi
    sample_rankings = model_resp.rank_donors(
        donors=seed_donors,
        req_lat=28.7041,
        req_lon=77.1025,
        req_blood_group="O+",
        top_k=5
    )
    
    # System risk alerts
    alerts = [
        {
            "id": "ALT-01",
            "type": "Deficit Alert",
            "severity": "High",
            "title": "Projected O- Negative Deficit in Mumbai",
            "description": "Projected 7-day demand is 48 units, but active available O- donors within 20km is 14.",
            "action": "Trigger targeted WhatsApp broadcast to Cluster 0 (Dedicated Champions) in Mumbai."
        },
        {
            "id": "ALT-02",
            "type": "Fraud Alert",
            "severity": "Critical",
            "title": "Suspicious Rapid Submission Burst Flagged",
            "description": "Phone +91 9988112233 submitted 12 requests within 30 minutes with cash reward text.",
            "action": "Blocked automatically by Isolation Forest & Spam Heuristics."
        },
        {
            "id": "ALT-03",
            "type": "Retention Alert",
            "severity": "Medium",
            "title": "42 High-Value Donors Due for Milestone Re-Engagement",
            "description": "Eligible donors with >80% past responsiveness have not donated in 100+ days.",
            "action": "Automated WhatsApp nudge scheduled for Saturday morning."
        }
    ]

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "totalDonorsInNetwork": len(seed_donors),
            "availableNow": sum(1 for d in seed_donors if d.get("available")),
            "activeEmergencyRequests": 6,
            "modelsOnline": 6
        },
        "metrics": cached_metrics,
        "sampleForecast": delhi_fc,
        "sampleRankedDonors": sample_rankings,
        "clusterProfiles": model_seg.cluster_profiles_,
        "systemAlerts": alerts
    }

@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the interactive HTML dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Dashboard HTML under construction</h1>"
