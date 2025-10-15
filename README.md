# Customer Churn Prediction & AI Service

An end-to-end **Customer Churn Prediction** project — from data modeling and explainability to deployment via FastAPI for real-time churn risk scoring.  
Designed as part of a case study to evaluate modeling, experimentation, and deployment skills.

---

## 🎯 Problem Statement
Telecom companies face high customer turnover, directly affecting revenue and customer lifetime value.  
The objective of this case study is to build a predictive model that classifies customers likely to churn and deploy it as an AI microservice for real-time scoring.  

📄 Reference: *Case Study Document – Customer Churn Prediction & AI Solution Deployment*  
📊 Dataset: *customer_churn_dataset.csv*

---

## 🧩 Approach & Methodology

### 1. Data Preparation
- Used the provided preprocessed dataset containing customer demographics, account details, and service usage.  
- Applied **feature selection, imputation**, and **encoding** using `ColumnTransformer` in `src/pipeline.py`.

### 2. Modeling
- Trained multiple models: **Logistic Regression**, **Random Forest**, and **MLPClassifier** (neural network proxy).  
- Evaluated using **GridSearchCV** with **ROC-AUC** as the main metric.  
- Selected best model and serialized it as `best_model.pkl`.

### 3. Evaluation
- Metrics tracked: **ROC-AUC**, **F1**, **Precision**, **Recall**, and **Confusion Matrix**.  
- Feature importance computed using **Permutation Importance** for explainability.  
- Results stored in `output/metrics.json` and `output/feature_importance.csv`.

### 4. Deployment
- Built a **FastAPI** service (`api/main.py`) that:
  - Loads the trained model (`models/best_model.pkl`)
  - Serves predictions through REST API endpoints
  - Allows **live model reload** without restarting the server
  - Exposes metadata via `/version`

### 5. Business Insights
- Key churn drivers identified:
  - Contract type (month-to-month most risky)
  - Short tenure customers
  - High monthly charges
  - Fiber optic internet users  
- Recommendations:
  - Offer annual contracts or loyalty discounts.
  - Target high-charge, low-tenure customers for retention.
  - Bundle additional services to increase stickiness.

---

## ⚙️ How to Run

### 1.Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### 2.Train the Model
python src/train.py --data data/customer_churn_dataset.csv --target Churn --outdir output

Artifacts generated:
	•	models/best_model.pkl – saved model
	•	output/metrics.json – model metrics
	•	output/predictions.csv – predictions
	•	output/feature_importance.csv – feature importance
###3.Run the FastAPI Service
uvicorn api.main:app --reload

###4.Test Endpoints
-Check API status:
curl http://127.0.0.1:8000/health
-Make a prediction:
curl -X POST "http://127.0.0.1:8000/predict_churn" \
  -H "Content-Type: application/json" \
  -d '{
    "customerID":"0001-XYZ",
    "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":2,
    "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
    "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
    "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
    "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
    "MonthlyCharges":75.9,"TotalCharges":151.65
  }'


Method. Endpoint  Description
GET /health Returns API status and model load state
GET /version Displays current model version, timestamp, and performance metrics
POST /predict_churn?threshold=0.45 Returns churn probability; adjustable threshold
POST /reload_model Reloads latest model after retraining without restarting the API

-Example with custom threshold:
curl -X POST "http://127.0.0.1:8000/predict_churn?threshold=0.40" \
  -H "Content-Type: application/json" \
  -d @api/sample_event.json

-Repository Structure
keshav-customer-churn-case-study/
│
├── api/
│   ├── main.py                # FastAPI application
│   ├── sample_event.json      # Example API input
│
├── data/
│   └── customer_churn_dataset.csv
│
├── models/
│   └── best_model.pkl         # Trained model
│
├── output/
│   ├── metrics.json
│   ├── feature_importance.csv
│   └── predictions.csv
│
├── src/
│   ├── pipeline.py            # Preprocessing & feature handling
│   ├── train.py               # Model training & evaluation
│   └── utils.py
│
├── requirements.txt
└── README.md