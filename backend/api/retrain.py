# backend/api/retrain.py
from fastapi import APIRouter, UploadFile, File
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

router = APIRouter()

@router.post("/")
async def retrain_model(file: UploadFile = File(...)):
    # Load and clean data
    df = pd.read_csv(file.file)
    df.columns = df.columns.str.strip()

    if 'Obesity_Risk' not in df.columns:
        return {"error": "'Obesity_Risk' column missing in uploaded file."}

    # Label encoding
    label_maps = {
        'Region': {'Urban': 1, 'Rural': 0},
        'Gender': {'Male': 1, 'Female': 0},
        'Physical_Activity': {'Low': 0, 'Moderate': 1, 'High': 2}
    }
    for col, mapping in label_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Target column
    df['Obesity_Risk_Category'] = df['Obesity_Risk'].apply(lambda x: 0 if x < 0.4 else 1 if x <= 0.7 else 2)

    # Features & Target
    X = df.drop(columns=['Obesity_Risk', 'Obesity_Risk_Category'], errors='ignore')
    y = df['Obesity_Risk_Category']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.pkl")

    # Retrain model
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    model.fit(X_scaled, y)
    model.save_model("models/xgb_obesity_model.json")

    return {"message": "Model retrained and saved successfully."}