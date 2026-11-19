"""
train_all.py
Master training and evaluation pipeline for all six AI/ML modules.
Trains, validates, computes all evaluation metrics, and persists checkpoints to disk.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Ensure workspace root is always in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.model_selection import train_test_split

from ml_engine.dataset import (
    generate_donors,
    generate_response_training_data,
    generate_demand_time_series,
    generate_noshow_training_data,
    generate_request_nlp_data,
    generate_retention_training_data,
    generate_segmentation_data
)
from ml_engine.models.donor_response import DonorResponseModel
from ml_engine.models.demand_forecasting import BloodDemandForecaster
from ml_engine.models.noshow_prediction import DonorNoShowPredictor
from ml_engine.models.fake_request_detector import FakeRequestAndUrgencyDetector
from ml_engine.models.retention_reminders import DonorRetentionModel
from ml_engine.models.donor_segmentation import DonorSegmentationModel

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

def run_training_pipeline():
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_metrics = {}
    print("=" * 70)
    print("BLOOD DONATION AI/ML SYSTEM - TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # Feature 1: Donor Response Prediction
    # ---------------------------------------------------------
    print("\n[1/6] Training Feature 1: Donor Response Prediction (Logistic Regression & Gradient Boosting)...")
    df_resp = generate_response_training_data(n_samples=3000)
    train_resp, val_resp = train_test_split(df_resp, test_size=0.25, random_state=42, stratify=df_resp["responded"])
    
    resp_model = DonorResponseModel(model_type="logistic")
    resp_model.fit(train_resp)
    resp_metrics = resp_model.evaluate(val_resp, k_list=[3, 5, 10])
    resp_model.save(os.path.join(SAVE_DIR, "donor_response.joblib"))
    all_metrics["donor_response"] = resp_metrics
    
    print(f"  ✓ ROC-AUC: {resp_metrics['roc_auc']:.4f}")
    print(f"  ✓ Precision@K: {resp_metrics['precision_at_k']}")
    print(f"  ✓ Recall@K: {resp_metrics['recall_at_k']}")

    # ---------------------------------------------------------
    # Feature 2: Blood Demand Forecasting
    # ---------------------------------------------------------
    print("\n[2/6] Training Feature 2: Blood Demand Forecasting (Time Series Autoregressive Ridge Forecaster)...")
    df_ts = generate_demand_time_series(days=400)
    # Split chronologically: first 320 days train, last 80 days validation
    unique_dates = sorted(df_ts["date"].unique())
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    train_ts = df_ts[df_ts["date"] < split_date]
    val_ts = df_ts[df_ts["date"] >= split_date]
    
    demand_model = BloodDemandForecaster()
    demand_model.fit(train_ts)
    demand_metrics = demand_model.evaluate(val_ts)
    demand_model.save(os.path.join(SAVE_DIR, "demand_forecasting.joblib"))
    all_metrics["demand_forecasting"] = demand_metrics
    
    print(f"  ✓ Overall MAE:  {demand_metrics['overall_mae']:.3f} units")
    print(f"  ✓ Overall RMSE: {demand_metrics['overall_rmse']:.3f} units")

    # ---------------------------------------------------------
    # Feature 3: Donor No-Show Prediction
    # ---------------------------------------------------------
    print("\n[3/6] Training Feature 3: Donor No-Show Prediction (Balanced Random Forest Classifier)...")
    df_noshow = generate_noshow_training_data(n_samples=2500)
    train_ns, val_ns = train_test_split(df_noshow, test_size=0.25, random_state=42, stratify=df_noshow["no_show"])
    
    noshow_model = DonorNoShowPredictor()
    noshow_model.fit(train_ns)
    noshow_metrics = noshow_model.evaluate(val_ns)
    noshow_model.save(os.path.join(SAVE_DIR, "noshow_prediction.joblib"))
    all_metrics["noshow_prediction"] = noshow_metrics
    
    print(f"  ✓ Recall:    {noshow_metrics['recall']:.4f}")
    print(f"  ✓ F1-Score:  {noshow_metrics['f1_score']:.4f}")
    print(f"  ✓ Precision: {noshow_metrics['precision']:.4f}")

    # ---------------------------------------------------------
    # Feature 4: Urgency and Fake Request Detection
    # ---------------------------------------------------------
    print("\n[4/6] Training Feature 4: Urgency & Fake Request Detection (TF-IDF NLP + Isolation Forest)...")
    df_req = generate_request_nlp_data(n_samples=2200)
    train_req, val_req = train_test_split(df_req, test_size=0.25, random_state=42, stratify=df_req["is_fake"])
    
    fake_detector = FakeRequestAndUrgencyDetector()
    fake_detector.fit(train_req)
    fake_metrics = fake_detector.evaluate(val_req)
    fake_detector.save(os.path.join(SAVE_DIR, "fake_request_detector.joblib"))
    all_metrics["fake_request_and_urgency"] = fake_metrics
    
    print(f"  ✓ Fake Detection Precision: {fake_metrics['fake_detection']['precision']:.4f}")
    print(f"  ✓ Fake Detection Recall:    {fake_metrics['fake_detection']['recall']:.4f}")
    print(f"  ✓ Urgency Macro Precision:  {fake_metrics['urgency_classification']['precision_macro']:.4f}")
    print(f"  ✓ Urgency Macro Recall:     {fake_metrics['urgency_classification']['recall_macro']:.4f}")

    # ---------------------------------------------------------
    # Feature 5: Donor Retention and Smart Reminders
    # ---------------------------------------------------------
    print("\n[5/6] Training Feature 5: Donor Retention & Smart Reminders (Gradient Boosting Churn Classifier)...")
    df_ret = generate_retention_training_data(n_samples=2500)
    train_ret, val_ret = train_test_split(df_ret, test_size=0.25, random_state=42, stratify=df_ret["churned"])
    
    retention_model = DonorRetentionModel()
    retention_model.fit(train_ret)
    retention_metrics = retention_model.evaluate(val_ret)
    retention_model.save(os.path.join(SAVE_DIR, "retention_reminders.joblib"))
    all_metrics["donor_retention"] = retention_metrics
    
    print(f"  ✓ AUC (ROC-AUC): {retention_metrics['auc']:.4f}")
    print(f"  ✓ Accuracy:      {retention_metrics['accuracy']:.4f}")

    # ---------------------------------------------------------
    # Feature 6: Donor Segmentation
    # ---------------------------------------------------------
    print("\n[6/6] Training Feature 6: Donor Segmentation (K-Means Clustering)...")
    df_seg = generate_segmentation_data(n_samples=1800)
    train_seg, val_seg = train_test_split(df_seg, test_size=0.20, random_state=42)
    
    seg_model = DonorSegmentationModel(n_clusters=4)
    seg_model.fit(train_seg)
    seg_metrics = seg_model.evaluate(val_seg)
    seg_model.save(os.path.join(SAVE_DIR, "donor_segmentation.joblib"))
    all_metrics["donor_segmentation"] = seg_metrics
    
    print(f"  ✓ Silhouette Score:       {seg_metrics['silhouette_score']:.4f}")
    print(f"  ✓ Davies-Bouldin Index:   {seg_metrics['davies_bouldin_index']:.4f}")

    # ---------------------------------------------------------
    # Generate seed donors pool for live inference & testing
    # ---------------------------------------------------------
    print("\n[+] Generating realistic active donor pool for testing and live application use...")
    seed_donors = generate_donors(n=350)
    donors_list = seed_donors.to_dict(orient="records")
    with open(os.path.join(SAVE_DIR, "seed_donors.json"), "w", encoding="utf-8") as f:
        json.dump(donors_list, f, indent=2)
    print(f"  ✓ Saved {len(donors_list)} donors to seed_donors.json")

    # Save comprehensive metrics JSON
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  ✓ Saved all model evaluation metrics to metrics.json")

    print("\n" + "=" * 70)
    print("ALL SIX AI/ML MODULES TRAINED & EVALUATED SUCCESSFULLY!")
    print("=" * 70)
    return all_metrics

if __name__ == "__main__":
    run_training_pipeline()
