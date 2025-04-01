import pandas as pd
import joblib

# Load the saved StandardScaler
scaler = joblib.load("models/xgb_obesity_model.pkl")

# Columns used during training
numerical_columns = [
    'Age',
    'BMI',
    'Blood_Pressure_Systolic',
    'Blood_Pressure_Diastolic',
    'Cholesterol',
    'Blood_Sugar',
    'Diet_Quality'
]

# Categorical feature encodings used during training
label_maps = {
    'Region': {'Urban': 1, 'Rural': 0},
    'Gender': {'Male': 1, 'Female': 0},
    'Physical_Activity': {'Low': 0, 'Moderate': 1, 'High': 2}
}

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    label_maps = {
        'Region': {'Urban': 1, 'Rural': 0},
        'Gender': {'Male': 1, 'Female': 0},
        'Physical_Activity': {'Low': 0, 'Moderate': 1, 'High': 2}
    }

    for col, mapping in label_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Preserve column order for scaler
    df = df[numerical_columns + list(label_maps.keys())]

    # Scale numeric features
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    return df

