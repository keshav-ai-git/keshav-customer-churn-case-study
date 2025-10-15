from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from pydantic import BaseModel, RootModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Churn Prediction API", version="1.0.0")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"
model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

class CustomerEvent(RootModel[Dict[str, Any]]):
    pass

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict_churn")
def predict(evt: CustomerEvent):
    if model is None:
        return {"error": "Model not loaded. Train first."}
    df = pd.DataFrame([evt.root])

    # --- Fix: ensure all expected columns exist ---
    expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is not None:
        # Add missing columns with dummy values
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

    # Try to cast numerics where possible
    df = df.apply(pd.to_numeric, errors="ignore")
    prob = float(model.predict_proba(df[expected_cols])[0, 1])

    return {
        "churn_probability": prob,
        "recommendation": "flag_for_retention" if prob >= 0.45 else "no_action"
    }

# NEW: Reload endpoint (add this at the end)
@app.post("/reload_model")
def reload_model():
    """Reloads the trained churn model from disk without restarting the API."""
    global model
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")
    try:
        model = joblib.load(MODEL_PATH)
        return {"reloaded": True, "model_path": str(MODEL_PATH)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {e}")

from datetime import datetime
import json

METRICS_PATH = MODEL_PATH.parents[1] / "output" / "metrics.json"

@app.get("/version")
def version():
    info = {"model_loaded": model is not None}
    try:
        # model file timestamp
        if MODEL_PATH.exists():
            ts = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat()
            info["model_path"] = str(MODEL_PATH)
            info["model_last_modified"] = ts
        # training metrics
        if METRICS_PATH.exists():
            info["metrics"] = json.loads(METRICS_PATH.read_text())
    except Exception as e:
        info["error"] = f"{e}"
    return info