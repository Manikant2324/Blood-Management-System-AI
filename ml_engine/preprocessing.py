"""
preprocessing.py
Shared data preprocessing, distance calculation, blood compatibility rules,
and specialized evaluation metric calculations (Precision@K, Recall@K, MAE, RMSE, AUC, Silhouette).
"""

import math
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple

COMPATIBILITY_MAP = {
    "O-": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],
    "O+": ["O+", "A+", "B+", "AB+"],
    "A-": ["A-", "A+", "AB-", "AB+"],
    "A+": ["A+", "AB+"],
    "B-": ["B-", "B+", "AB-", "AB+"],
    "B+": ["B+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"],
}

CITY_CENTROIDS = {
    "mumbai": (19.0760, 72.8777),
    "delhi": (28.7041, 77.1025),
    "bengaluru": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
}

def is_blood_compatible(donor_group: str, recipient_group: str) -> bool:
    """Check whole blood/RBC compatibility."""
    if not donor_group or not recipient_group:
        return False
    d = donor_group.strip().upper()
    r = recipient_group.strip().upper()
    allowed = COMPATIBILITY_MAP.get(d, [])
    return r in allowed

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two GPS coordinates in kilometers."""
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except (TypeError, ValueError):
        return 25.0  # safe default if missing
        
    R = 6371.0  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(R * c, 2)

def clean_text(text: Optional[str]) -> str:
    """Sanitize and normalize text for NLP processing."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s\d]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==========================================
# Specialized Evaluation Metrics
# ==========================================

def precision_at_k(actual_positives: List[int], predicted_scores: List[float], k: int = 5) -> float:
    """
    Precision@K: Proportion of the top-K recommended items that are actual positive responses.
    """
    if len(actual_positives) == 0 or len(predicted_scores) == 0:
        return 0.0
    k = min(k, len(actual_positives))
    if k <= 0:
        return 0.0
    # Sort indices by descending predicted score
    ranked_indices = np.argsort(predicted_scores)[::-1][:k]
    num_hits = sum(actual_positives[i] for i in ranked_indices)
    return float(num_hits / k)

def recall_at_k(actual_positives: List[int], predicted_scores: List[float], k: int = 5) -> float:
    """
    Recall@K: Proportion of all true positive responses captured in the top-K recommendations.
    """
    total_positives = sum(actual_positives)
    if total_positives == 0:
        return 0.0
    k = min(k, len(actual_positives))
    if k <= 0:
        return 0.0
    ranked_indices = np.argsort(predicted_scores)[::-1][:k]
    num_hits = sum(actual_positives[i] for i in ranked_indices)
    return float(num_hits / total_positives)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
