# AI/ML-Powered Blood Donation Prediction & Management System

## Overview
This system implements six intelligent machine learning features for an advanced Blood Donation Prediction and Management platform, transitioning from static heuristics to production-grade predictive intelligence.

---

## 1. Architectural Summary of the 6 Intelligent Modules

| # | Feature | Target Task | Primary ML Algorithm | Required Evaluation Metrics | Validation Performance |
|---|---|---|---|---|---|
| **1** | **Donor Response Prediction** | Rank donors based on distance, recency, response rate, and time of day | Logistic Regression & Gradient Boosting with Standard Scaling | **Recall@K & Precision@K** (K=3, 5, 10), ROC-AUC | **P@3: 66.7%**, **P@5: 66.1%**, **P@10: 59.9%**, **Recall@10: 55.8%**, **ROC-AUC: 0.699** |
| **2** | **Blood Demand Forecasting** | Predict daily future requirements across 8 blood groups and 8 Indian metro cities | Autoregressive Time-Series Ridge Forecaster with Cyclical & Monsoon Trends | **MAE & RMSE** | **Overall MAE: 3.52 units**, **Overall RMSE: 5.42 units** |
| **3** | **Donor No-Show Prediction** | Predict appointment attendance vs no-show failure | Balanced Random Forest Classifier (120 Trees) | **Recall & F1-score**, Precision | **Recall: 66.3%**, **F1-Score: 0.555**, Precision: 47.8% |
| **4** | **Urgency & Fake Request Detection** | Dual clinical NLP urgency classification + fraud/burst anomaly detection | TF-IDF (1-2 n-grams) + Calibrated Classifier + Isolation Forest | **Precision & Recall** (Fake & Urgency) | **Fraud Precision: 100%**, **Fraud Recall: 100%**, **Urgency Macro Precision: 91.9%**, **Macro Recall: 93.4%** |
| **5** | **Donor Retention & Smart Reminders** | Predict donor churn risk and calculate optimal scientific reminder timing | Gradient Boosting Classifier | **AUC (ROC-AUC)**, Accuracy | **AUC: 0.8236**, **Accuracy: 95.7%** |
| **6** | **Donor Segmentation** | Cluster donors into actionable behavioral personas | K-Means++ Clustering (K=4) | **Silhouette Score**, Davies-Bouldin Index | **Silhouette Score: 0.4373**, **Davies-Bouldin: 0.9425** |

---

## 2. Detailed Technical Breakdown

### Feature 1: Donor Response Prediction
- **Inputs**:
  - `distance_km`: Calculated via Haversine distance from request coordinates.
  - `days_since_last_donation`: Cooldown recency (90 days minimum eligibility threshold in India; 999 for first-timers).
  - `past_response_rate`: Historical proportion of broadcast alerts responded to.
  - `time_of_day_hour`: Hour of alert (0–23), modeling circadian response dips (sleeping hours 22:00–06:00).
  - Compatibility filter: Universal ABO/Rh rules (`O-` universal RBC donor, `AB+` universal recipient).
- **Evaluation**:
  - Simulated across realistic alert batches of candidate donors.
  - Evaluated on $K \in \{3, 5, 10\}$:
    - $\text{Precision@5} = 66.13\%$
    - $\text{Recall@10} = 55.81\%$
    - $\text{ROC-AUC} = 0.6987$
- **Output**: Ranked donor pool with response probability $P(\text{response}=1)$, eligibility status ("Eligible" vs "Wait X days"), and recommendation badges ("Highly Recommended", "Recommended", "Backup Donor").

### Feature 2: Blood Demand Forecasting
- **Inputs**:
  - Daily aggregated request time-series across 8 blood groups and 8 Indian cities (Delhi, Mumbai, Bengaluru, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad).
  - Cyclical calendar features: $\sin(2\pi \cdot \text{day}/7)$, $\cos(2\pi \cdot \text{day}/7)$, $\sin(2\pi \cdot \text{month}/12)$, $\cos(2\pi \cdot \text{month}/12)$.
  - Monsoon & dengue surge indicator (July–October annual spike).
  - City population demand scaling factors.
- **Evaluation**:
  - Chronological 80/20 train/test split.
  - $\text{Overall MAE} = 3.52\text{ units}$
  - $\text{Overall RMSE} = 5.42\text{ units}$
- **Output**: Multi-day forecast horizon (7, 14, 30 days) with projected demand and 95% confidence intervals ($\pm 1.96 \cdot \text{RMSE}$).

### Feature 3: Donor No-Show Prediction
- **Inputs**:
  - `total_past_donations`: Historical donor loyalty count.
  - `past_noshow_rate`: Past appointment absenteeism rate.
  - `urgency`: Request urgency tier (Emergency, Urgent, Routine).
  - `distance_km`: Travel distance to donation facility.
  - `lead_time_days`: Days between booking and scheduled appointment.
  - `reminder_sent`: Confirmation SMS / WhatsApp status flag.
- **Evaluation**:
  - $\text{Recall} = 66.29\%$
  - $\text{F1-score} = 0.5553$
  - $\text{Precision} = 47.77\%$
- **Output**: Probability of no-show, Risk Tier ("Low Risk", "Moderate Risk", "High Risk"), and tailored operational actions (e.g., standby backup donor dispatch, 1-click WhatsApp confirmation, 12h phone check).

### Feature 4: Urgency and Fake Request Detection
- **Inputs**:
  - Clinical request message text / patient note.
  - Units requested (detecting abnormal spikes >10–20 units).
  - 24-hour submission burst frequency from same phone/IP.
  - Hospital / Clinic affiliation verification.
- **Architecture**:
  - **NLP Urgency**: TF-IDF (1–2 n-grams, medical stopword handling) + Calibrated Logistic Classifier for 3 classes: Critical, Urgent, Routine.
  - **Fraud & Anomaly**: Isolation Forest on submission volume, burst rates, and lexical fraud patterns ("selling blood", "cash reward", "lottery", "no prescription").
- **Evaluation**:
  - Fake Detection Precision: $100\%$, Recall: $100\%$
  - Urgency Macro Precision: $91.92\%$, Macro Recall: $93.38\%$
- **Output**: Urgency classification with confidence scores, anomaly score, fraud probability, and specific suspicion warning flags.

### Feature 5: Donor Retention and Smart Reminders
- **Inputs**:
  - Annual donation frequency, recency (days since last donation), lifetime donations, alert responsiveness, tenure in months, and app logins.
- **Evaluation**:
  - $\text{ROC-AUC} = 0.8236$
  - $\text{Accuracy} = 95.68\%$
- **Smart Reminder Engine**:
  - Medical 90-day cooldown enforcement (blocks reminders during recovery).
  - Churn risk classification (Low, Moderate, High).
  - Generates optimal send date, recommended channel (WhatsApp, SMS, Email), and personalized message copy.

### Feature 6: Donor Segmentation
- **Inputs**:
  - Age, distance to center (km), annual donation frequency, emergency responsiveness, and tenure (years).
- **Evaluation**:
  - K-Means++ with $K=4$.
  - $\text{Silhouette Score} = 0.4373$
  - $\text{Davies-Bouldin Index} = 0.9425$
- **Cluster Personas**:
  1. **Dedicated Champions (25%)**: Consistent, frequent donors with high emergency responsiveness. *Strategy: VIP recognition, early shortage alerts.*
  2. **Young First-Timers & Potentials (30%)**: Age 18–25, mobile-active, early donation journey. *Strategy: Gamification, campus blood drives, digital milestone badges.*
  3. **Emergency Rapid Responders (25%)**: Moderate regular cadence, but 90%+ response during critical SOS alerts. *Strategy: Reserve alerts for Code-Red emergencies to prevent fatigue.*
  4. **Lapsed / Distant Donors (20%)**: Located farther or showing declining frequency. *Strategy: Mobile van donation camps, flexible weekend slots.*

---

## 3. Directory Layout

```
d:\Blood_management_system-main\
├── ml_engine\
│   ├── dataset.py                # Realistic dataset generators for all 6 tasks
│   ├── preprocessing.py          # Distance, ABO compatibility, text sanitizing, metrics
│   ├── train_all.py              # Master training and evaluation pipeline
│   ├── api.py                    # FastAPI server exposing REST endpoints
│   ├── cli.py                    # CLI tool for predictions and evaluations
│   ├── test_all_features.py      # Automated verification test suite
│   ├── dashboard.html            # Standalone executive interactive Web Dashboard
│   ├── saved_models\
│   │   ├── donor_response.joblib
│   │   ├── demand_forecasting.joblib
│   │   ├── noshow_prediction.joblib
│   │   ├── fake_request_detector.joblib
│   │   ├── retention_reminders.joblib
│   │   ├── donor_segmentation.joblib
│   │   ├── metrics.json          # Serialized evaluation metrics for all 6 models
│   │   └── seed_donors.json       # Realistic active donor pool (350 records)
│   └── models\
│       ├── donor_response.py
│       ├── demand_forecasting.py
│       ├── noshow_prediction.py
│       ├── fake_request_detector.py
│       ├── retention_reminders.py
│       └── donor_segmentation.py
└── Blood_management_system-main\
    └── src\
        ├── app\
        │   ├── api\
        │   │   ├── match\route.ts          # Upgraded with ML Donor Response ranking
        │   │   └── ml\[...slug]\route.ts   # Next.js API proxy to ML microservice
        │   └── ml-dashboard\page.tsx       # Next.js ML Dashboard UI
        └── components\
            └── Navbar.tsx                  # Added AI / ML Hub navigation link
```

---

## 4. How to Run and Test

### A. Run Automated Verification Test Suite
```powershell
$env:PYTHONPATH="d:\Blood_management_system-main"
& "C:\Users\MANIKANT\AppData\Local\Python\pythoncore-3.14-64\python.exe" "d:\Blood_management_system-main\ml_engine\test_all_features.py"
```

### B. Use the CLI Tool
```powershell
# 1. Inspect all evaluation metrics
& "C:\Users\MANIKANT\AppData\Local\Python\pythoncore-3.14-64\python.exe" "d:\Blood_management_system-main\ml_engine\cli.py" metrics

# 2. Rank donors for an incoming O+ emergency request in Delhi
& "C:\Users\MANIKANT\AppData\Local\Python\pythoncore-3.14-64\python.exe" "d:\Blood_management_system-main\ml_engine\cli.py" rank --blood-group O+ --lat 28.7041 --lon 77.1025 --top-k 5

# 3. Forecast 14-day blood demand for O- in Mumbai
& "C:\Users\MANIKANT\AppData\Local\Python\pythoncore-3.14-64\python.exe" "d:\Blood_management_system-main\ml_engine\cli.py" forecast --city Mumbai --blood-group O- --days 14

# 4. Predict appointment no-show risk
& "C:\Users\MANIKANT\AppData\Local\Python\pythoncore-3.14-64\python.exe" "d:\Blood_management_system-main\ml_engine\cli.py" noshow --donations 1 --noshow-rate 0.4 --distance 22.0 --lead-time 7

# 5. Evaluate an incoming request for clinical urgency and fraud signals
& "C:\Users\MANIKANT\AppData\Local\Python\pythoncore-3.14-64\python.exe" "d:\Blood_management_system-main\ml_engine\cli.py" evaluate-request --text "Patient in emergency ICU severe bleeding trauma at AIIMS Hospital" --units 3

# 6. Generate smart retention reminder
& "C:\Users\MANIKANT\AppData\Local\Python\pythoncore-3.14-64\python.exe" "d:\Blood_management_system-main\ml_engine\cli.py" reminder --name "Sunil Mehta" --days-since 120 --frequency 2.5

# 7. Classify donor into behavioral cluster
& "C:\Users\MANIKANT\AppData\Local\Python\pythoncore-3.14-64\python.exe" "d:\Blood_management_system-main\ml_engine\cli.py" segment --age 23 --distance 9.0 --frequency 1.1 --responsiveness 0.55 --tenure 1.0
```

### C. Launch the Web Dashboard & API Server
To run the FastAPI server (accessible via browser at `http://127.0.0.1:8000/dashboard` and `http://127.0.0.1:8000/docs`):
```powershell
$env:PYTHONPATH="d:\Blood_management_system-main"
& "C:\Users\MANIKANT\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m uvicorn ml_engine.api:app --host 127.0.0.1 --port 8000
```
Alternatively, simply double-click or open `d:\Blood_management_system-main\ml_engine\dashboard.html` in Chrome or Edge to interact with the dashboard immediately!
