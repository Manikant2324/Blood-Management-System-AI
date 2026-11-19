"""
fake_request_detector.py
Feature 4: Urgency and Fake Request Detection
NLP classification and anomaly detection based on request text, frequency, and suspicious patterns.
Evaluated using Precision and Recall.
"""

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import joblib
import os
from typing import Dict, Any, List, Optional

from ml_engine.preprocessing import clean_text

SUSPICIOUS_PHRASES = [
    "selling", "wholesale", "cash reward", "pure blood bags", "pay advance",
    "money reward", "no prescription", "blood broker", "lottery", "urgent cash",
    "delivery at home", "upi advance", "guaranteed pure"
]

CRITICAL_MEDICAL_KEYWORDS = [
    "icu", "accident", "trauma", "hemorrhage", "hemoglobin", "bypass", "platelet",
    "bleeding", "code red", "critical", "leukemia", "emergency", "delivery room", "unstable"
]

class FakeRequestAndUrgencyDetector:
    def __init__(self):
        # NLP Pipeline for Urgency
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=800,
            preprocessor=clean_text
        )
        self.urgency_clf = LogisticRegression(
            C=2.0, max_iter=1000, class_weight="balanced", random_state=42
        )
        self.urgency_labels = ["Routine", "Urgent", "Critical"]
        
        # Anomaly Detector for Fake / Burst / Scam Patterns
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.10,  # ~10% expected anomalous
            random_state=42
        )
        
        # Fake Classifier (supervised on metadata + text features)
        self.fake_clf = LogisticRegression(
            C=1.0, class_weight="balanced", random_state=42
        )
        
        self.is_trained = False
        self.metrics_ = {}

    def _extract_anomaly_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract multi-signal metadata features for anomaly & fraud detection."""
        features = []
        for _, row in df.iterrows():
            text = str(row.get("text", "")).lower()
            units = float(row.get("units_requested", 1))
            freq = float(row.get("frequency_last_24h", 1))
            
            # Count suspicious commercial/spam phrase hits
            suspicious_count = sum(1 for phrase in SUSPICIOUS_PHRASES if phrase in text)
            # Count medical emergency keyword hits
            med_count = sum(1 for kw in CRITICAL_MEDICAL_KEYWORDS if kw in text)
            
            text_length = len(text)
            has_phone = 1.0 if re.search(r"\d{10}", text) else 0.0
            
            features.append([
                units,
                freq,
                suspicious_count,
                med_count,
                text_length,
                has_phone
            ])
        return np.array(features)

    def fit(self, train_df: pd.DataFrame):
        """Train NLP urgency classifier, anomaly detector, and fake classifier."""
        # 1. Train TF-IDF + Urgency Classifier
        texts = train_df["text"].values
        y_urgency = train_df["urgency_label"].values
        X_tfidf = self.tfidf.fit_transform(texts)
        self.urgency_clf.fit(X_tfidf, y_urgency)
        
        # 2. Extract anomaly features
        X_anom = self._extract_anomaly_features(train_df)
        self.anomaly_detector.fit(X_anom)
        
        # 3. Train Fake Classifier
        y_fake = train_df["is_fake"].values
        self.fake_clf.fit(X_anom, y_fake)
        
        self.is_trained = True

    def evaluate(self, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model performance using Precision and Recall (required),
        for both Fake Detection and Urgency Classification.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")
            
        texts = val_df["text"].values
        y_urgency_true = val_df["urgency_label"].values
        y_fake_true = val_df["is_fake"].values
        
        # 1. Urgency Evaluation
        X_tfidf = self.tfidf.transform(texts)
        y_urgency_pred = self.urgency_clf.predict(X_tfidf)
        
        urgency_precision_macro = float(precision_score(y_urgency_true, y_urgency_pred, average="macro"))
        urgency_recall_macro = float(recall_score(y_urgency_true, y_urgency_pred, average="macro"))
        urgency_f1_macro = float(f1_score(y_urgency_true, y_urgency_pred, average="macro"))
        
        # 2. Fake Detection Evaluation
        X_anom = self._extract_anomaly_features(val_df)
        fake_probs = self.fake_clf.predict_proba(X_anom)[:, 1]
        fake_preds = (fake_probs >= 0.5).astype(int)
        
        fake_prec = float(precision_score(y_fake_true, fake_preds, zero_division=0))
        fake_rec = float(recall_score(y_fake_true, fake_preds, zero_division=0))
        fake_f1 = float(f1_score(y_fake_true, fake_preds, zero_division=0))
        fake_acc = float(accuracy_score(y_fake_true, fake_preds))
        
        self.metrics_ = {
            "fake_detection": {
                "precision": round(fake_prec, 4),
                "recall": round(fake_rec, 4),
                "f1_score": round(fake_f1, 4),
                "accuracy": round(fake_acc, 4)
            },
            "urgency_classification": {
                "precision_macro": round(urgency_precision_macro, 4),
                "recall_macro": round(urgency_recall_macro, 4),
                "f1_macro": round(urgency_f1_macro, 4)
            }
        }
        return self.metrics_

    def analyze_request(
        self,
        text: str,
        units_requested: int = 1,
        frequency_last_24h: int = 1,
        hospital_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Real-time comprehensive request evaluation.
        Produces Urgency Classification + Anomaly / Fake Detection breakdown.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")
            
        clean_req_text = text if text and str(text).strip() else "Urgent blood request needed"
        
        # 1. NLP Urgency Classification
        X_tfidf = self.tfidf.transform([clean_req_text])
        urgency_probs = self.urgency_clf.predict_proba(X_tfidf)[0]
        urgency_idx = int(np.argmax(urgency_probs))
        predicted_urgency = self.urgency_labels[urgency_idx]
        urgency_confidence = float(urgency_probs[urgency_idx])
        
        # 2. Fraud & Anomaly Analysis
        df_single = pd.DataFrame([{
            "text": clean_req_text,
            "units_requested": units_requested,
            "frequency_last_24h": frequency_last_24h
        }])
        X_anom = self._extract_anomaly_features(df_single)
        
        # Isolation Forest score (< 0 indicates anomalous outlier)
        isolation_score = float(self.anomaly_detector.decision_function(X_anom)[0])
        is_isolation_outlier = isolation_score < 0.0
        
        # Supervised fake probability
        fake_prob = float(self.fake_clf.predict_proba(X_anom)[0, 1])
        
        # 3. Rule-based suspicious pattern detection
        suspicious_flags = []
        lower_t = clean_req_text.lower()
        
        for phrase in SUSPICIOUS_PHRASES:
            if phrase in lower_t:
                suspicious_flags.append(f"Commercial/Fraud keyword detected: '{phrase}'")
                fake_prob = max(fake_prob, 0.85)
                
        if units_requested > 10:
            suspicious_flags.append(f"Extremely high volume requested ({units_requested} units) without hospital verification")
            fake_prob = max(fake_prob, 0.75)
            
        if frequency_last_24h > 5:
            suspicious_flags.append(f"High submission burst frequency ({frequency_last_24h} requests in past 24 hours)")
            fake_prob = max(fake_prob, 0.70)
            
        if not hospital_name or len(hospital_name.strip()) < 3:
            suspicious_flags.append("Missing or unverified hospital/clinic affiliation")
            
        if is_isolation_outlier and not suspicious_flags:
            suspicious_flags.append("Statistical behavioral outlier detected by Isolation Forest")
            
        # Determine overall safety verdict
        if fake_prob >= 0.70 or len(suspicious_flags) >= 2:
            verdict = "Blocked (High Risk / Suspected Fake)"
            status = "flagged"
        elif fake_prob >= 0.40 or len(suspicious_flags) == 1:
            verdict = "Manual Review Required by Medical Team"
            status = "review"
        else:
            verdict = "Verified Legitimate Request"
            status = "approved"
            
        return {
            "urgency": {
                "predictedUrgency": predicted_urgency,
                "confidence": round(urgency_confidence, 4),
                "probabilities": {
                    "Routine": round(float(urgency_probs[0]), 4),
                    "Urgent": round(float(urgency_probs[1]), 4),
                    "Critical": round(float(urgency_probs[2]), 4)
                }
            },
            "fakeDetection": {
                "isFakeOrSuspicious": status != "approved",
                "fraudProbability": round(fake_prob, 4),
                "anomalyScore": round(isolation_score, 4),
                "verdict": verdict,
                "status": status,
                "suspiciousFlags": suspicious_flags
            }
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "tfidf": self.tfidf,
            "urgency_clf": self.urgency_clf,
            "anomaly_detector": self.anomaly_detector,
            "fake_clf": self.fake_clf,
            "metrics": self.metrics_
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.tfidf = data["tfidf"]
        self.urgency_clf = data["urgency_clf"]
        self.anomaly_detector = data["anomaly_detector"]
        self.fake_clf = data["fake_clf"]
        self.metrics_ = data.get("metrics", {})
        self.is_trained = True
