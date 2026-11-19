"""
noshow_prediction.py
Feature 3: Donor No-Show Prediction
Binary classification based on donation history, urgency, and distance.
Evaluated using Recall and F1-score.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, confusion_matrix
import joblib
import os
from typing import Dict, Any, List, Optional

class DonorNoShowPredictor:
    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            class_weight="balanced",  # handle no-show class imbalance
            random_state=42
        )
        self.scaler = StandardScaler()
        self.urgency_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.numeric_features = [
            "total_past_donations",
            "past_noshow_rate",
            "distance_km",
            "lead_time_days",
            "reminder_sent",
            "slot_hour"
        ]
        self.is_trained = False
        self.metrics_ = {}

    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        df = df.copy()
        
        # Impute missing values
        df["total_past_donations"] = df.get("total_past_donations", 0).fillna(0).clip(0, 100)
        df["past_noshow_rate"] = df.get("past_noshow_rate", 0.25).fillna(0.25).clip(0.0, 1.0)
        df["distance_km"] = df.get("distance_km", 10.0).fillna(10.0).clip(0.1, 150.0)
        df["lead_time_days"] = df.get("lead_time_days", 2).fillna(2).clip(0, 60)
        df["reminder_sent"] = df.get("reminder_sent", 1).fillna(1).clip(0, 1)
        df["slot_hour"] = df.get("slot_hour", 11).fillna(11).clip(7, 21)
        
        # Urgency category
        urgency_series = df.get("urgency", pd.Series(["Routine"] * len(df))).fillna("Routine")
        urgency_df = pd.DataFrame({"urgency": urgency_series})
        
        if is_training:
            urgency_ohe = self.urgency_encoder.fit_transform(urgency_df)
            num_scaled = self.scaler.fit_transform(df[self.numeric_features].values)
        else:
            urgency_ohe = self.urgency_encoder.transform(urgency_df)
            num_scaled = self.scaler.transform(df[self.numeric_features].values)
            
        return np.hstack([num_scaled, urgency_ohe])

    def fit(self, train_df: pd.DataFrame, target_col: str = "no_show"):
        """Train Random Forest classifier on appointment records."""
        X = self._prepare_features(train_df, is_training=True)
        y = train_df[target_col].values
        self.clf.fit(X, y)
        self.is_trained = True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns probability of no-show P(no_show = 1)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")
        X = self._prepare_features(df, is_training=False)
        return self.clf.predict_proba(X)[:, 1]

    def evaluate(self, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model using Recall and F1-score (required),
        plus Precision, Accuracy, and Confusion Matrix.
        """
        X = self._prepare_features(val_df, is_training=False)
        y_true = val_df["no_show"].values
        
        probs = self.clf.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        rec = float(recall_score(y_true, preds))
        f1 = float(f1_score(y_true, preds))
        prec = float(precision_score(y_true, preds))
        acc = float(accuracy_score(y_true, preds))
        cm = confusion_matrix(y_true, preds).tolist()
        
        self.metrics_ = {
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "accuracy": round(acc, 4),
            "confusion_matrix": cm,
            "feature_importance": self.get_feature_importances()
        }
        return self.metrics_

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        feature_names = self.numeric_features + list(self.urgency_encoder.get_feature_names_out(["urgency"]))
        fi = self.clf.feature_importances_
        return {name: round(float(f), 4) for name, f in zip(feature_names, fi)}

    def predict_appointment_risk(
        self,
        total_past_donations: int = 2,
        past_noshow_rate: float = 0.1,
        urgency: str = "Urgent",
        distance_km: float = 8.5,
        lead_time_days: int = 3,
        reminder_sent: int = 1,
        slot_hour: int = 10
    ) -> Dict[str, Any]:
        """
        Assess no-show probability for an individual appointment and return actionable intervention.
        """
        input_df = pd.DataFrame([{
            "total_past_donations": total_past_donations,
            "past_noshow_rate": past_noshow_rate,
            "urgency": urgency,
            "distance_km": distance_km,
            "lead_time_days": lead_time_days,
            "reminder_sent": reminder_sent,
            "slot_hour": slot_hour
        }])
        
        prob = float(self.predict_proba(input_df)[0])
        
        if prob >= 0.55:
            risk_tier = "High Risk"
            recommendations = [
                "Schedule a standby backup donor for this slot.",
                "Trigger automated WhatsApp confirmation request with 1-click confirmation button.",
                "Make direct phone confirmation call 12 hours prior."
            ]
        elif prob >= 0.30:
            risk_tier = "Moderate Risk"
            recommendations = [
                "Send 24-hour SMS reminder with clinic directions and parking guidance.",
                "Prompt donor to confirm attendance 6 hours before appointment."
            ]
        else:
            risk_tier = "Low Risk"
            recommendations = [
                "Standard automated appointment reminder sufficient.",
                "High probability donor will attend as scheduled."
            ]
            
        return {
            "noShowProbability": round(prob, 4),
            "attendanceProbability": round(1.0 - prob, 4),
            "riskTier": risk_tier,
            "recommendations": recommendations,
            "factors": {
                "distanceKm": distance_km,
                "leadTimeDays": lead_time_days,
                "pastNoShowRate": past_noshow_rate,
                "urgency": urgency
            }
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "clf": self.clf,
            "scaler": self.scaler,
            "urgency_encoder": self.urgency_encoder,
            "metrics": self.metrics_
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.clf = data["clf"]
        self.scaler = data["scaler"]
        self.urgency_encoder = data["urgency_encoder"]
        self.metrics_ = data.get("metrics", {})
        self.is_trained = True
