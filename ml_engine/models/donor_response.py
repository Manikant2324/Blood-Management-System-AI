"""
donor_response.py
Feature 1: Donor Response Prediction
Model: Logistic Regression and Gradient Boosting Classifier
Ranks donors based on distance, days since last donation, past response rate, and time of day.
Evaluated using Recall@K and Precision@K (K=3, 5, 10).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import os
from typing import List, Dict, Any, Optional

from ml_engine.preprocessing import (
    haversine_distance_km,
    is_blood_compatible,
    precision_at_k,
    recall_at_k
)

class DonorResponseModel:
    def __init__(self, model_type: str = "logistic"):
        """
        model_type: 'logistic' for Logistic Regression or 'gradient_boost' for Gradient Boosting.
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        if model_type == "gradient_boost":
            self.clf = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42
            )
        else:
            self.clf = LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=42
            )
        self.feature_names = [
            "distance_km",
            "days_since_last_donation",
            "past_response_rate",
            "time_of_day_hour"
        ]
        self.is_trained = False
        self.metrics_ = {}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and impute features cleanly."""
        X = pd.DataFrame()
        # Distance: default 15 km if missing
        X["distance_km"] = df["distance_km"].fillna(15.0).clip(0.1, 150.0)
        # Days since last donation: default 999 (long time ago / first timer) if missing
        X["days_since_last_donation"] = df["days_since_last_donation"].fillna(999.0).clip(0.0, 2000.0)
        # Past response rate: default 0.5 if missing
        X["past_response_rate"] = df["past_response_rate"].fillna(0.5).clip(0.0, 1.0)
        # Time of day: 0 to 23
        X["time_of_day_hour"] = df["time_of_day_hour"].fillna(12).clip(0, 23)
        return X[self.feature_names].values

    def fit(self, train_df: pd.DataFrame, target_col: str = "responded"):
        """Train the model and fit the scaler."""
        X = self._prepare_features(train_df)
        y = train_df[target_col].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_trained = True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability P(responded = 1)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        probs = self.clf.predict_proba(X_scaled)[:, 1]
        return probs

    def evaluate(self, val_df: pd.DataFrame, k_list: List[int] = [3, 5, 10]) -> Dict[str, Any]:
        """
        Evaluate model using Precision@K, Recall@K, ROC-AUC, and Accuracy.
        Computes both operational event-level Precision@K / Recall@K
        (simulating 100 realistic emergency broadcast batches of 25 donors)
        and dataset-wide ranking metrics.
        """
        X = self._prepare_features(val_df)
        y_true = val_df["responded"].values
        X_scaled = self.scaler.transform(X)
        
        probs = self.clf.predict_proba(X_scaled)[:, 1]
        auc = float(roc_auc_score(y_true, probs))
        preds = (probs >= 0.5).astype(int)
        acc = float(accuracy_score(y_true, preds))
        
        # 1. Operational Event-Level Precision@K and Recall@K
        # Simulate 100 alert events with batches of candidate donors
        rng = np.random.RandomState(42)
        n_sim = 150
        batch_size = 25
        event_p_at_k = {f"Precision@{k}": [] for k in k_list}
        event_r_at_k = {f"Recall@{k}": [] for k in k_list}
        
        for _ in range(n_sim):
            idx = rng.choice(len(val_df), size=batch_size, replace=False)
            batch_y = y_true[idx]
            batch_probs = probs[idx]
            
            if sum(batch_y) == 0:
                continue
                
            for k in k_list:
                event_p_at_k[f"Precision@{k}"].append(precision_at_k(batch_y, batch_probs, k=k))
                event_r_at_k[f"Recall@{k}"].append(recall_at_k(batch_y, batch_probs, k=k))
                
        avg_p_at_k = {k: round(float(np.mean(v)), 4) for k, v in event_p_at_k.items()}
        avg_r_at_k = {k: round(float(np.mean(v)), 4) for k, v in event_r_at_k.items()}
            
        self.metrics_ = {
            "model_type": self.model_type,
            "roc_auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "precision_at_k": avg_p_at_k,
            "recall_at_k": avg_r_at_k,
            "feature_importances": self.get_feature_importances()
        }
        return self.metrics_

    def get_feature_importances(self) -> Dict[str, float]:
        """Extract coefficients or feature importances."""
        if not self.is_trained:
            return {}
        if hasattr(self.clf, "coef_"):
            coeffs = self.clf.coef_[0]
            return {name: round(float(c), 4) for name, c in zip(self.feature_names, coeffs)}
        elif hasattr(self.clf, "feature_importances_"):
            fi = self.clf.feature_importances_
            return {name: round(float(f), 4) for name, f in zip(self.feature_names, fi)}
        return {}

    def rank_donors(
        self,
        donors: List[Dict[str, Any]],
        req_lat: float,
        req_lon: float,
        req_blood_group: str,
        req_hour: int = 14,
        max_distance_km: float = 50.0,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Intelligent Donor Ranking for an incoming request.
        1. Filters by ABO/Rh blood compatibility.
        2. Calculates exact Haversine distance.
        3. Extracts features and computes ML response probability.
        4. Returns top-K ranked donors with explanation breakdown.
        """
        if not donors:
            return {"count": 0, "ranked_donors": []}

        compatible_candidates = []
        feature_rows = []

        for d in donors:
            # Check availability
            if not d.get("available", True):
                continue
                
            # Compatibility check
            donor_bg = d.get("bloodGroup", "")
            if req_blood_group and not is_blood_compatible(donor_bg, req_blood_group):
                continue
                
            d_lat = d.get("latitude")
            d_lon = d.get("longitude")
            dist = haversine_distance_km(req_lat, req_lon, d_lat, d_lon)
            if dist > max_distance_km:
                continue
                
            days_since = d.get("daysSinceLastDonation")
            if days_since is None:
                days_since = 999.0
            else:
                try:
                    days_since = float(days_since)
                except (ValueError, TypeError):
                    days_since = 999.0

            resp_rate = d.get("pastResponseRate")
            if resp_rate is None:
                resp_rate = 0.5
            else:
                try:
                    resp_rate = float(resp_rate)
                except (ValueError, TypeError):
                    resp_rate = 0.5
            
            compatible_candidates.append({
                "donor": d,
                "distance_km": dist,
                "days_since_last_donation": days_since,
                "past_response_rate": resp_rate
            })
            
            feature_rows.append({
                "distance_km": dist,
                "days_since_last_donation": days_since,
                "past_response_rate": resp_rate,
                "time_of_day_hour": req_hour
            })

        if not compatible_candidates:
            return {"count": 0, "ranked_donors": [], "message": "No compatible donors within search radius"}

        feat_df = pd.DataFrame(feature_rows)
        probs = self.predict_proba(feat_df)

        ranked = []
        for cand, prob in zip(compatible_candidates, probs):
            donor_info = cand["donor"]
            # Eligibility status flag
            is_physically_eligible = cand["days_since_last_donation"] >= 90
            
            ranked.append({
                "donorId": donor_info.get("id"),
                "name": donor_info.get("name", f"{donor_info.get('firstName', '')} {donor_info.get('lastName', '')}".strip()),
                "bloodGroup": donor_info.get("bloodGroup"),
                "phone": donor_info.get("phone"),
                "email": donor_info.get("email"),
                "city": donor_info.get("city"),
                "distanceKm": cand["distance_km"],
                "daysSinceLastDonation": cand["days_since_last_donation"],
                "pastResponseRate": cand["past_response_rate"],
                "responseProbability": round(float(prob), 4),
                "eligibilityStatus": "Eligible" if is_physically_eligible else f"Wait {int(90 - cand['days_since_last_donation'])} days",
                "recommendationTier": "Highly Recommended" if prob >= 0.70 else ("Recommended" if prob >= 0.45 else "Backup Donor")
            })

        # Rank descending by response probability
        ranked.sort(key=lambda x: x["responseProbability"], reverse=True)
        top_matches = ranked[:top_k]

        return {
            "count": len(top_matches),
            "total_compatible": len(ranked),
            "ranked_donors": top_matches
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"clf": self.clf, "scaler": self.scaler, "metrics": self.metrics_}, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.clf = data["clf"]
        self.scaler = data["scaler"]
        self.metrics_ = data.get("metrics", {})
        self.is_trained = True
