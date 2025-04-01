# backend/api/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import xgboost as xgb
import pandas as pd
import os
from src.preprocessing import preprocess_input

router = APIRouter()

# Load model and scaler
MODEL_PATH = "models/xgb_obesity_model.json"
SCALER_PATH = "models/scaler.pkl"

try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")

# Request Schema
class InputData(BaseModel):
    Age: int
    BMI: float
    Blood_Pressure_Systolic: int
    Blood_Pressure_Diastolic: int
    Cholesterol: int
    Blood_Sugar: int
    Diet_Quality: int
    Region: str
    Gender: str
    Physical_Activity: str

@router.post("/single")
def predict_single(input: InputData):
    try:
        input_df = pd.DataFrame([input.dict()])
        processed = preprocess_input(input_df, scaler)
        prediction = model.predict(processed)[0]
        label_map = {0: "Low", 1: "Medium", 2: "High"}
        return {"risk_prediction": label_map.get(prediction, "Unknown")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
def predict_batch(file: bytes):
    try:
        df = pd.read_csv(pd.compat.StringIO(file.decode("utf-8")))
        processed = preprocess_input(df, scaler)
        predictions = model.predict(processed)
        label_map = {0: "Low", 1: "Medium", 2: "High"}
        labels = [label_map.get(p, "Unknown") for p in predictions]
        return {"predictions": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")