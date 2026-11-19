"""
donor_segmentation.py
Feature 6: Donor Segmentation
Donor segmentation using K-Means clustering based on age, location, and donation frequency.
Evaluated using the Silhouette Score.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import joblib
import os
from typing import Dict, Any, List, Optional

CLUSTER_METADATA = {
    0: {
        "name": "Dedicated Champions",
        "badge": "Champion",
        "description": "Consistent, frequent donors with high emergency responsiveness and long tenure. The backbone of blood bank reserves.",
        "strategy": "VIP recognition, blood bank ambassador invites, early notification for critical shortages."
    },
    1: {
        "name": "Young First-Timers & Potentials",
        "badge": "First-Timer",
        "description": "Younger cohort (ages 18-25) in early donation stages. Highly active on mobile, strong potential for long-term loyalty.",
        "strategy": "Gamification, digital social badges, university campus drives, educational myth-busting content."
    },
    2: {
        "name": "Emergency Rapid Responders",
        "badge": "Rapid Responder",
        "description": "Donors who donate occasionally for routine drives but demonstrate exceptional 90%+ response rates during urgent SOS alerts.",
        "strategy": "Reserve priority alerts for genuine Code-Red emergencies to prevent donor alert fatigue."
    },
    3: {
        "name": "Lapsed / Distant Donors",
        "badge": "At-Risk",
        "description": "Donors residing farther from donation centers or showing declining frequency. At risk of permanent churn.",
        "strategy": "Mobile van donation camps near their neighborhoods, flexible weekend appointment slots, win-back surveys."
    }
}

class DonorSegmentationModel:
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=15,
            max_iter=300,
            random_state=42
        )
        self.pca = PCA(n_components=2, random_state=42)
        self.features = [
            "age",
            "distance_km",
            "annual_donation_frequency",
            "emergency_responsiveness",
            "tenure_years"
        ]
        self.is_trained = False
        self.metrics_ = {}
        self.cluster_profiles_ = {}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.features].fillna(df[self.features].mean()).values
        return X

    def fit(self, train_df: pd.DataFrame):
        """Fit scaler and K-Means clustering model."""
        X = self._prepare_features(train_df)
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        self.pca.fit(X_scaled)
        self.is_trained = True
        
        # Build cluster profile centroids and assign canonical names
        self._build_cluster_profiles(train_df, X_scaled)

    def _build_cluster_profiles(self, df: pd.DataFrame, X_scaled: np.ndarray):
        labels = self.kmeans.labels_
        df_labeled = df.copy()
        df_labeled["cluster"] = labels
        
        profiles = {}
        for c in range(self.n_clusters):
            c_data = df_labeled[df_labeled["cluster"] == c]
            meta = CLUSTER_METADATA.get(c, {
                "name": f"Segment {c+1}",
                "badge": "Standard",
                "description": "General donor group.",
                "strategy": "Standard communications."
            })
            
            profiles[c] = {
                "clusterId": c,
                "name": meta["name"],
                "badge": meta["badge"],
                "description": meta["description"],
                "strategy": meta["strategy"],
                "donorCount": len(c_data),
                "percentage": round(100.0 * len(c_data) / max(1, len(df_labeled)), 1),
                "averages": {
                    "age": round(float(c_data["age"].mean()), 1),
                    "distanceKm": round(float(c_data["distance_km"].mean()), 1),
                    "annualDonationFrequency": round(float(c_data["annual_donation_frequency"].mean()), 2),
                    "emergencyResponsiveness": round(float(c_data["emergency_responsiveness"].mean()), 2),
                    "tenureYears": round(float(c_data["tenure_years"].mean()), 1)
                }
            }
        self.cluster_profiles_ = profiles

    def evaluate(self, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate clustering quality using Silhouette Score (required),
        plus Davies-Bouldin index and Inertia.
        """
        X = self._prepare_features(val_df)
        X_scaled = self.scaler.transform(X)
        labels = self.kmeans.predict(X_scaled)
        
        sil_score = float(silhouette_score(X_scaled, labels))
        db_score = float(davies_bouldin_score(X_scaled, labels))
        inertia = float(self.kmeans.inertia_)
        
        self.metrics_ = {
            "n_clusters": self.n_clusters,
            "silhouette_score": round(sil_score, 4),
            "davies_bouldin_index": round(db_score, 4),
            "inertia": round(inertia, 2),
            "cluster_profiles": self.cluster_profiles_
        }
        return self.metrics_

    def predict_segment(self, donor: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a single donor into their segment persona."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")
            
        df = pd.DataFrame([{
            "age": donor.get("age", 30),
            "distance_km": donor.get("distance_km", donor.get("distanceKm", 10.0)),
            "annual_donation_frequency": donor.get("annual_donation_frequency", donor.get("totalDonations", 1.5) / 2.0),
            "emergency_responsiveness": donor.get("emergency_responsiveness", donor.get("pastResponseRate", 0.6)),
            "tenure_years": donor.get("tenure_years", 2.0)
        }])
        
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        cluster_id = int(self.kmeans.predict(X_scaled)[0])
        pca_coords = self.pca.transform(X_scaled)[0].tolist()
        
        profile = self.cluster_profiles_.get(cluster_id, CLUSTER_METADATA.get(cluster_id, {}))
        
        return {
            "clusterId": cluster_id,
            "segmentName": profile.get("name", f"Segment {cluster_id}"),
            "badge": profile.get("badge", "Active"),
            "description": profile.get("description", ""),
            "actionStrategy": profile.get("strategy", ""),
            "pcaCoordinates": [round(pca_coords[0], 3), round(pca_coords[1], 3)]
        }

    def get_dashboard_scatter_data(self, sample_size: int = 150) -> List[Dict[str, Any]]:
        """Return 2D PCA mapped points for interactive clustering scatter plot."""
        if not self.is_trained:
            return []
        centers_scaled = self.kmeans.cluster_centers_
        centers_pca = self.pca.transform(centers_scaled)
        
        centers_data = []
        for i, pt in enumerate(centers_pca):
            centers_data.append({
                "clusterId": i,
                "isCenter": True,
                "x": round(float(pt[0]), 3),
                "y": round(float(pt[1]), 3),
                "name": self.cluster_profiles_.get(i, {}).get("name", f"Cluster {i} Centroid")
            })
        return centers_data

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "kmeans": self.kmeans,
            "scaler": self.scaler,
            "pca": self.pca,
            "metrics": self.metrics_,
            "profiles": self.cluster_profiles_
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.kmeans = data["kmeans"]
        self.scaler = data["scaler"]
        self.pca = data["pca"]
        self.metrics_ = data.get("metrics", {})
        self.cluster_profiles_ = data.get("profiles", {})
        self.is_trained = True
