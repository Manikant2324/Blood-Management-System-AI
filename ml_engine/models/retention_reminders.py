"""
retention_reminders.py
Feature 5: Donor Retention and Smart Reminders
Churn prediction and smart reminder scheduling based on donation frequency and engagement.
Evaluated using AUC (ROC-AUC).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import joblib
import os
from typing import Dict, Any, List, Optional

class DonorRetentionModel:
    def __init__(self):
        self.clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            "donation_frequency",
            "recency_days",
            "total_donations",
            "response_rate",
            "tenure_months",
            "app_logins_per_month",
            "distance_to_center_km"
        ]
        self.is_trained = False
        self.metrics_ = {}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_names].fillna(0.0).values
        return X

    def fit(self, train_df: pd.DataFrame, target_col: str = "churned"):
        """Train churn prediction model."""
        X = self._prepare_features(train_df)
        y = train_df[target_col].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_trained = True

    def predict_churn_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns churn probability P(churned = 1)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def evaluate(self, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model using AUC (required), plus Accuracy, Precision, and Recall.
        """
        X = self._prepare_features(val_df)
        y_true = val_df["churned"].values
        X_scaled = self.scaler.transform(X)
        
        probs = self.clf.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        auc = float(roc_auc_score(y_true, probs))
        acc = float(accuracy_score(y_true, preds))
        prec = float(precision_score(y_true, preds))
        rec = float(recall_score(y_true, preds))
        
        self.metrics_ = {
            "auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "feature_importances": self.get_feature_importances()
        }
        return self.metrics_

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        fi = self.clf.feature_importances_
        return {name: round(float(f), 4) for name, f in zip(self.feature_names, fi)}

    def generate_smart_reminder(
        self,
        donor_id: str,
        name: str,
        blood_group: str,
        days_since_last_donation: int = 95,
        donation_frequency: float = 2.0,
        total_donations: int = 4,
        response_rate: float = 0.75,
        tenure_months: int = 18,
        preferred_city: str = "Delhi"
    ) -> Dict[str, Any]:
        """
        Generates individual retention analysis, churn risk, and smart reminder schedule.
        """
        donor_df = pd.DataFrame([{
            "donation_frequency": donation_frequency,
            "recency_days": days_since_last_donation,
            "total_donations": total_donations,
            "response_rate": response_rate,
            "tenure_months": tenure_months,
            "app_logins_per_month": 3,
            "distance_to_center_km": 6.5
        }])
        
        churn_prob = float(self.predict_churn_proba(donor_df)[0])
        
        # Medical eligibility standard (90 days interval for whole blood)
        COOLDOWN_DAYS = 90
        is_eligible = days_since_last_donation >= COOLDOWN_DAYS
        days_until_eligible = max(0, COOLDOWN_DAYS - days_since_last_donation)
        
        now = datetime.now()
        
        if not is_eligible:
            schedule_status = "In Cooldown"
            scheduled_date = (now + timedelta(days=days_until_eligible)).strftime("%Y-%m-%d")
            reminder_channel = "Calendar Notification"
            urgency_level = "Informational"
            message = (
                f"Hello {name}, thank you for your commitment to saving lives! "
                f"You are currently in your recovery period and will be eligible to donate blood again in {days_until_eligible} days on {scheduled_date}."
            )
        elif churn_prob >= 0.65:
            schedule_status = "High Churn Risk - Urgent Retention"
            scheduled_date = now.strftime("%Y-%m-%d")
            reminder_channel = "Personalized WhatsApp & Phone Outreach"
            urgency_level = "High"
            message = (
                f"Dear {name}, patients in {preferred_city} are urgently seeking {blood_group} blood. "
                f"Your past {total_donations} donations saved up to {total_donations * 3} lives. "
                f"We would love to welcome you back this week! Book your priority slot today."
            )
        elif churn_prob >= 0.35:
            schedule_status = "Moderate Churn Risk - Nudge"
            scheduled_date = (now + timedelta(days=2)).strftime("%Y-%m-%d")
            reminder_channel = "SMS & App Notification"
            urgency_level = "Medium"
            message = (
                f"Hi {name}, you are now eligible to donate {blood_group} blood! "
                f"Local blood banks in {preferred_city} have scheduled convenient donation drives this weekend. Tap to view centers nearby."
            )
        else:
            schedule_status = "Loyal Super-Donor - Routine Schedule"
            scheduled_date = (now + timedelta(days=5)).strftime("%Y-%m-%d")
            reminder_channel = "Email & App Notification"
            urgency_level = "Normal"
            message = (
                f"Greetings {name}! As one of our most valued {blood_group} donors, "
                f"your next donation window has opened. Thank you for your continued dedication."
            )
            
        return {
            "donorId": donor_id,
            "name": name,
            "bloodGroup": blood_group,
            "churnProbability": round(churn_prob, 4),
            "retentionRisk": "High" if churn_prob >= 0.65 else ("Moderate" if churn_prob >= 0.35 else "Low (Loyal)"),
            "isEligibleNow": is_eligible,
            "daysSinceLastDonation": days_since_last_donation,
            "daysUntilEligible": days_until_eligible,
            "scheduledReminderDate": scheduled_date,
            "recommendedChannel": reminder_channel,
            "urgencyLevel": urgency_level,
            "reminderMessage": message
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "clf": self.clf,
            "scaler": self.scaler,
            "metrics": self.metrics_
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.clf = data["clf"]
        self.scaler = data["scaler"]
        self.metrics_ = data.get("metrics", {})
        self.is_trained = True
