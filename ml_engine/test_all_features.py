"""
test_all_features.py
Comprehensive test suite verifying all 6 AI/ML modules, their required metrics,
edge case handling, and API functionality.
"""

import unittest
import os
import sys
import json
import numpy as np
import pandas as pd

# Ensure workspace root is always in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml_engine.models.donor_response import DonorResponseModel
from ml_engine.models.demand_forecasting import BloodDemandForecaster
from ml_engine.models.noshow_prediction import DonorNoShowPredictor
from ml_engine.models.fake_request_detector import FakeRequestAndUrgencyDetector
from ml_engine.models.retention_reminders import DonorRetentionModel
from ml_engine.models.donor_segmentation import DonorSegmentationModel
from ml_engine.api import (
    load_all_models,
    health,
    get_metrics,
    rank_donors_endpoint,
    forecast_demand_endpoint,
    predict_noshow_endpoint,
    evaluate_request_endpoint,
    predict_retention_endpoint,
    get_dashboard_overview,
    DonorRankRequest,
    DemandForecastRequest,
    NoShowPredictionRequest,
    RequestEvaluationInput,
    RetentionReminderRequest
)
from ml_engine.preprocessing import is_blood_compatible, haversine_distance_km

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

class TestAllMLFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_all_models()

    # -------------------------------------------------------------
    # Feature 1: Donor Response Prediction (Recall@K and Precision@K)
    # -------------------------------------------------------------
    def test_feature1_donor_response(self):
        print("\n--> Testing Feature 1: Donor Response Prediction (Recall@K & Precision@K)...")
        model = DonorResponseModel()
        model_path = os.path.join(MODEL_DIR, "donor_response.joblib")
        self.assertTrue(os.path.exists(model_path), "donor_response.joblib must exist")
        model.load(model_path)
        self.assertTrue(model.is_trained)

        # Check metrics
        metrics = model.metrics_
        self.assertIn("precision_at_k", metrics)
        self.assertIn("recall_at_k", metrics)
        self.assertIn("roc_auc", metrics)
        
        # Verify Precision@K and Recall@K values are valid floats
        p_at_5 = metrics["precision_at_k"].get("Precision@5")
        r_at_10 = metrics["recall_at_k"].get("Recall@10")
        self.assertGreater(p_at_5, 0.40, "Precision@5 should be > 0.40")
        self.assertGreater(r_at_10, 0.30, "Recall@10 should be > 0.30")
        self.assertGreater(metrics["roc_auc"], 0.60, "ROC-AUC should be > 0.60")

        # Test donor ranking with edge cases (missing last donation, extreme distance)
        test_donors = [
            {"id": "D1", "name": "Aarav", "bloodGroup": "O+", "latitude": 28.70, "longitude": 77.10, "daysSinceLastDonation": 120, "pastResponseRate": 0.8},
            {"id": "D2", "name": "Bhavna", "bloodGroup": "O+", "latitude": 28.72, "longitude": 77.12, "daysSinceLastDonation": None, "pastResponseRate": None}, # missing values
            {"id": "D3", "name": "Chetan", "bloodGroup": "B+", "latitude": 28.71, "longitude": 77.11, "daysSinceLastDonation": 30, "pastResponseRate": 0.2}, # incompatible blood group
            {"id": "D4", "name": "Deepak", "bloodGroup": "O-", "latitude": 35.00, "longitude": 77.00, "daysSinceLastDonation": 200, "pastResponseRate": 0.9}, # too far
        ]
        
        res = model.rank_donors(test_donors, req_lat=28.70, req_lon=77.10, req_blood_group="O+", max_distance_km=50, top_k=5)
        self.assertGreater(res["count"], 0)
        # Verify D3 (B+) was excluded from O+ request
        ranked_ids = [d["donorId"] for d in res["ranked_donors"]]
        self.assertNotIn("D3", ranked_ids, "B+ donor must not be ranked for O+ recipient")
        print("  ✓ Feature 1 Passed! Precision@5:", p_at_5, "Recall@10:", r_at_10)

    # -------------------------------------------------------------
    # Feature 2: Blood Demand Forecasting (MAE and RMSE)
    # -------------------------------------------------------------
    def test_feature2_demand_forecasting(self):
        print("\n--> Testing Feature 2: Blood Demand Forecasting (MAE & RMSE)...")
        model = BloodDemandForecaster()
        model_path = os.path.join(MODEL_DIR, "demand_forecasting.joblib")
        self.assertTrue(os.path.exists(model_path))
        model.load(model_path)
        self.assertTrue(model.is_trained)

        # Check required evaluation metrics: MAE and RMSE
        mae = model.metrics_.get("overall_mae")
        rmse = model.metrics_.get("overall_rmse")
        self.assertIsNotNone(mae, "MAE must be computed")
        self.assertIsNotNone(rmse, "RMSE must be computed")
        self.assertLess(mae, 8.0, "MAE must be reasonably low (< 8 units)")
        self.assertLess(rmse, 12.0, "RMSE must be reasonably low (< 12 units)")

        # Test multi-day forecast generation and confidence bounds
        fc = model.forecast_future(city="Delhi", blood_group="O+", days_ahead=7)
        self.assertEqual(fc["city"], "Delhi")
        self.assertIn("O+", fc["forecasts"])
        daily = fc["forecasts"]["O+"]["daily_forecast"]
        self.assertEqual(len(daily), 7)
        for d in daily:
            self.assertGreater(d["predictedUnits"], 0)
            self.assertGreaterEqual(d["upperCi"], d["predictedUnits"])
            self.assertLessEqual(d["lowerCi"], d["predictedUnits"])
        print("  ✓ Feature 2 Passed! MAE:", mae, "RMSE:", rmse)

    # -------------------------------------------------------------
    # Feature 3: Donor No-Show Prediction (Recall and F1-score)
    # -------------------------------------------------------------
    def test_feature3_noshow_prediction(self):
        print("\n--> Testing Feature 3: Donor No-Show Prediction (Recall & F1-score)...")
        model = DonorNoShowPredictor()
        model_path = os.path.join(MODEL_DIR, "noshow_prediction.joblib")
        self.assertTrue(os.path.exists(model_path))
        model.load(model_path)
        self.assertTrue(model.is_trained)

        # Check required evaluation metrics: Recall and F1-score
        rec = model.metrics_.get("recall")
        f1 = model.metrics_.get("f1_score")
        self.assertIsNotNone(rec, "Recall must be present")
        self.assertIsNotNone(f1, "F1-score must be present")
        self.assertGreater(rec, 0.50, "Recall should be >= 0.50 for no-show intervention")
        self.assertGreater(f1, 0.45, "F1-score should be >= 0.45")

        # Test high-risk vs low-risk appointment assessment
        high_risk = model.predict_appointment_risk(
            total_past_donations=0, past_noshow_rate=0.75, urgency="Routine", distance_km=35.0, lead_time_days=14
        )
        low_risk = model.predict_appointment_risk(
            total_past_donations=8, past_noshow_rate=0.05, urgency="Emergency", distance_km=2.0, lead_time_days=1
        )
        self.assertGreater(high_risk["noShowProbability"], low_risk["noShowProbability"])
        self.assertEqual(high_risk["riskTier"], "High Risk")
        self.assertEqual(low_risk["riskTier"], "Low Risk")
        print("  ✓ Feature 3 Passed! Recall:", rec, "F1-Score:", f1)

    # -------------------------------------------------------------
    # Feature 4: Urgency and Fake Request Detection (Precision & Recall)
    # -------------------------------------------------------------
    def test_feature4_fake_request_and_urgency(self):
        print("\n--> Testing Feature 4: NLP Urgency & Fake Request Detection (Precision & Recall)...")
        model = FakeRequestAndUrgencyDetector()
        model_path = os.path.join(MODEL_DIR, "fake_request_detector.joblib")
        self.assertTrue(os.path.exists(model_path))
        model.load(model_path)
        self.assertTrue(model.is_trained)

        # Check required evaluation metrics
        fake_metrics = model.metrics_.get("fake_detection", {})
        urgency_metrics = model.metrics_.get("urgency_classification", {})
        
        self.assertGreaterEqual(fake_metrics.get("precision", 0), 0.85)
        self.assertGreaterEqual(fake_metrics.get("recall", 0), 0.85)
        self.assertGreaterEqual(urgency_metrics.get("precision_macro", 0), 0.80)
        self.assertGreaterEqual(urgency_metrics.get("recall_macro", 0), 0.80)

        # Test legitimate critical request
        crit_res = model.analyze_request(
            text="Emergency ICU patient critical road trauma massive hemorrhage at AIIMS Hospital need 3 units O+",
            units_requested=3,
            frequency_last_24h=1,
            hospital_name="AIIMS New Delhi"
        )
        self.assertEqual(crit_res["urgency"]["predictedUrgency"], "Critical")
        self.assertEqual(crit_res["fakeDetection"]["status"], "approved")

        # Test fake commercial / burst request
        fake_res = model.analyze_request(
            text="Selling pure blood bags cheap cash reward wholesale lottery 100 bottles no prescription",
            units_requested=50,
            frequency_last_24h=20,
            hospital_name=""
        )
        self.assertTrue(fake_res["fakeDetection"]["isFakeOrSuspicious"])
        self.assertEqual(fake_res["fakeDetection"]["status"], "flagged")
        self.assertGreater(len(fake_res["fakeDetection"]["suspiciousFlags"]), 0)
        print("  ✓ Feature 4 Passed! Fake Precision:", fake_metrics["precision"], "Urgency Macro Precision:", urgency_metrics["precision_macro"])

    # -------------------------------------------------------------
    # Feature 5: Donor Retention and Smart Reminders (AUC)
    # -------------------------------------------------------------
    def test_feature5_donor_retention(self):
        print("\n--> Testing Feature 5: Donor Retention & Smart Reminders (AUC)...")
        model = DonorRetentionModel()
        model_path = os.path.join(MODEL_DIR, "retention_reminders.joblib")
        self.assertTrue(os.path.exists(model_path))
        model.load(model_path)
        self.assertTrue(model.is_trained)

        # Check required evaluation metric: AUC
        auc = model.metrics_.get("auc")
        self.assertIsNotNone(auc, "AUC must be evaluated")
        self.assertGreater(auc, 0.75, "AUC should be >= 0.75")

        # Test cooldown vs eligible reminder scheduling
        cooldown_res = model.generate_smart_reminder(
            donor_id="D-1", name="Kavita", blood_group="A+", days_since_last_donation=45
        )
        self.assertFalse(cooldown_res["isEligibleNow"])
        self.assertEqual(cooldown_res["daysUntilEligible"], 45)

        eligible_res = model.generate_smart_reminder(
            donor_id="D-2", name="Rahul", blood_group="O+", days_since_last_donation=110, donation_frequency=2.5
        )
        self.assertTrue(eligible_res["isEligibleNow"])
        self.assertEqual(eligible_res["daysUntilEligible"], 0)
        self.assertIn("Rahul", eligible_res["reminderMessage"])
        print("  ✓ Feature 5 Passed! AUC:", auc)

    # -------------------------------------------------------------
    # Feature 6: Donor Segmentation (Silhouette Score)
    # -------------------------------------------------------------
    def test_feature6_donor_segmentation(self):
        print("\n--> Testing Feature 6: Donor Segmentation (Silhouette Score)...")
        model = DonorSegmentationModel()
        model_path = os.path.join(MODEL_DIR, "donor_segmentation.joblib")
        self.assertTrue(os.path.exists(model_path))
        model.load(model_path)
        self.assertTrue(model.is_trained)

        # Check required evaluation metric: Silhouette Score
        sil = model.metrics_.get("silhouette_score")
        db = model.metrics_.get("davies_bouldin_index")
        self.assertIsNotNone(sil, "Silhouette Score must be evaluated")
        self.assertGreater(sil, 0.35, "Silhouette score should indicate well-separated clusters (> 0.35)")

        # Verify cluster profiles
        profiles = model.cluster_profiles_
        self.assertEqual(len(profiles), 4)

        # Test single donor classification
        seg_res = model.predict_segment({
            "age": 22, "distance_km": 8.0, "annual_donation_frequency": 1.0, "emergency_responsiveness": 0.5, "tenure_years": 0.8
        })
        self.assertIn("segmentName", seg_res)
        self.assertIn("actionStrategy", seg_res)
        print("  ✓ Feature 6 Passed! Silhouette Score:", sil, "Davies-Bouldin:", db)

    # -------------------------------------------------------------
    # API Endpoints Verification
    # -------------------------------------------------------------
    def test_fastapi_endpoints(self):
        print("\n--> Testing FastAPI Endpoints...")
        # 1. Health
        h_res = health()
        self.assertEqual(h_res["status"], "healthy")
        self.assertTrue(all(h_res["models_loaded"].values()))

        # 2. Metrics endpoint
        m_res = get_metrics()
        self.assertTrue(m_res["success"])
        self.assertIn("feature1_donor_response", m_res["summary"])
        self.assertIn("feature2_demand_forecasting", m_res["summary"])
        self.assertIn("feature3_noshow_prediction", m_res["summary"])
        self.assertIn("feature4_fake_and_urgency", m_res["summary"])
        self.assertIn("feature5_donor_retention", m_res["summary"])
        self.assertIn("feature6_donor_segmentation", m_res["summary"])

        # 3. Overview endpoint
        ov_res = get_dashboard_overview()
        self.assertTrue(ov_res["success"])
        self.assertEqual(ov_res["stats"]["modelsOnline"], 6)

        # 4. Donor rank endpoint
        r_res = rank_donors_endpoint(DonorRankRequest(
            bloodGroup="O+", latitude=28.7041, longitude=77.1025, topK=3
        ))
        self.assertTrue(r_res["success"])
        self.assertGreater(r_res["count"], 0)

        # 5. Demand forecast endpoint
        f_res = forecast_demand_endpoint(DemandForecastRequest(
            city="Mumbai", bloodGroup="O-", daysAhead=5
        ))
        self.assertTrue(f_res["success"])
        self.assertIn("O-", f_res["forecasts"])

        # 6. Request evaluation endpoint
        e_res = evaluate_request_endpoint(RequestEvaluationInput(
            text="Critical trauma ICU surgery emergency AIIMS Hospital",
            unitsRequested=2, frequencyLast24h=1, hospitalName="AIIMS"
        ))
        self.assertTrue(e_res["success"])
        self.assertEqual(e_res["urgency"]["predictedUrgency"], "Critical")

        print("  ✓ All API endpoints verified and operational!")

if __name__ == "__main__":
    unittest.main()
