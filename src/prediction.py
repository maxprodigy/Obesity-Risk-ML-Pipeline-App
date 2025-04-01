import pandas as pd
from src.preprocessing import preprocess_input
from src.model import load_model

# Load model
model = load_model()

# Labels
target_names = ['Low', 'Medium', 'High']

def predict_single(data_dict):
    df = pd.DataFrame([data_dict])
    processed = preprocess_input(df)
    prediction = model.predict(processed)[0]
    return target_names[prediction]

def predict_batch(df):
    processed = preprocess_input(df)
    predictions = model.predict(processed)
    return [target_names[pred] for pred in predictions]
