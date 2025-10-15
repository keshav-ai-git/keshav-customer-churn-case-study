Customer Churn Prediction & AI Solution Deployment

This repository presents an end-to-end AI solution for predicting telecom customer churn and deploying the model as a real-time inference API using FastAPI.
It demonstrates the full data science workflow — from model experimentation and explainability to API deployment — aligning with business impact metrics.

⸻

🚀 Problem Statement

Customer churn is one of the most critical KPIs in the telecom industry.
The objective of this case study is to:
	•	Predict the likelihood of a customer churning based on their demographic, billing, and service usage data.
	•	Deploy the best-performing model as a production-ready API for real-time scoring.
	•	Enable interpretability by identifying top churn drivers and linking them to actionable insights.

⸻

🧩 Approach

1️⃣ Data Preparation
	•	Used the provided preprocessed dataset (customer_churn_dataset.csv).
	•	Verified missing values, encoded categorical variables, and scaled numerical features.
	•	Split dataset into train/test (80:20) for model validation.

2️⃣ Modeling
	•	Baseline models: Logistic Regression and Random Forest
	•	Advanced model: MLPClassifier (Artificial Neural Network)
	•	Hyperparameter optimization with GridSearchCV
	•	Model evaluation using ROC-AUC, F1-score, and Confusion Matrix

3️⃣ Explainability
	•	Global feature importance via Permutation Importance
	•	Identified top churn drivers such as:
	•	Contract Type
	•	Tenure
	•	Monthly Charges
	•	Internet Service Type

4️⃣ Deployment
	•	Built FastAPI microservice (api/main.py) with endpoints:
	•	POST /predict_churn → Predict churn probability
	•	GET /health → Check model readiness
	•	GET /version → View model metadata & metrics
	•	POST /reload_model → Hot-reload updated model
	•	Deployed locally with Uvicorn for real-time inference.

###How to Run

🔧 Setup Environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

###Train & Generate Artifacts
python src/train.py --data data/customer_churn_dataset.csv --target Churn --outdir output


Artifacts created:
	•	models/best_model.pkl – Serialized best model
	•	output/metrics.json – ROC-AUC, F1, and parameters
	•	output/feature_importance.csv – Top churn drivers
	•	output/predictions.csv – Test set predictions
###Start API Server
uvicorn api.main:app --reload

Open in browser:
👉 http://127.0.0.1:8000/docs (Swagger UI)

###Example API Usage

Sample Request
{
  "customerID": "0001-XYZ",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.9,
  "TotalCharges": 151.65
}

###Sample Response

{
  "churn_probability": 0.4495,
  "recommendation": "no_action"
}
### Results & Insights

Metric. Logistic Regression. Random Forest. MLP
ROC-AUC 0.51 0.56 0.60
F1-Score 0.63 0.66 0.70


Key churn drivers (Permutation Importance):
	•	Contract type (month-to-month customers churn most)
	•	Tenure (shorter tenure = higher churn risk)
	•	InternetService = Fiber optic
	•	High monthly charges

Business Recommendations:
	•	Encourage annual/long-term contracts
	•	Offer loyalty rewards to new customers
	•	Promote bundled plans for Fiber customers
	•	Run personalized retention campaigns for high-charge customers
### Repository Structure

keshav-customer-churn-case-study/
├── api/
│   ├── main.py
│   ├── sample_event.json
│
├── data/
│   └── customer_churn_dataset.csv
│
├── models/
│   └── best_model.pkl
│
├── output/
│   ├── metrics.json
│   ├── feature_importance.csv
│   └── predictions.csv
│
├── src/
│   ├── pipeline.py
│   ├── train.py
│   └── utils.py
│
├── docs/
│   └── Churn Prediction API - Swagger UI.pdf
│
└── README.md


### Assumptions
	•	Dataset is pre-cleaned and balanced.
	•	Churn is binary: Yes (1) / No (0).
	•	Cost asymmetry: false negatives (missed churners) weigh higher than false positives.
	•	The model is optimized for retention-driven business outcomes, not just accuracy.

⸻

🏁 Conclusion

This project demonstrates a complete AI-driven churn prediction system with strong focus on:
	•	End-to-end reproducibility
	•	Explainable insights
	•	Business actionability
	•	Real-time deployment readiness

