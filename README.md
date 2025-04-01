# Obesity Risk Prediction System

This project combines a Machine Learning pipeline for obesity risk prediction with a web application interface.

## Project Structure

```
ML-Pipeline-Obesity-Risk/
├── README.md
├── notebook/
│   └── obesity_risk_analysis.ipynb    # Jupyter notebook with model development
├── src/
│   ├── preprocessing.py               # Data preprocessing functions
│   ├── model.py                      # Model definition and prediction
│   └── prediction.py                 # Prediction interface
├── data/
│   ├── train/                        # Training data
│   │   └── Synthetic_Health_Dataset.csv
│   └── test/                         # Test data
├── models/                           # Trained model files
│   ├── xgb_obesity_model.pkl         # StandardScaler
│   └── xgb_obesity_model.json        # XGBoost model
├── frontend/                         # React frontend application
└── backend/                          # FastAPI backend server
    └── api/                          # API endpoints
```

## Setup Instructions

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
```

3. Start the application:
```bash
# Using the provided batch file
.\run_app.bat
```

## Features

- Machine Learning Pipeline:
  - Data preprocessing and feature engineering
  - XGBoost model for obesity risk prediction
  - Model evaluation and validation

- Web Application:
  - React frontend with modern UI
  - FastAPI backend
  - Real-time predictions
  - Interactive data visualization

## Model Information

The obesity risk prediction model:
- Uses XGBoost classifier
- Takes 10 input features including:
  - Region
  - Age
  - Gender
  - BMI
  - Blood Pressure (Systolic/Diastolic)
  - Cholesterol
  - Blood Sugar
  - Physical Activity
  - Diet Quality

## Usage

1. Start the application using `run_app.bat`
2. Access the web interface at `http://localhost:3000`
3. Enter patient information
4. Get instant obesity risk predictions

## Development

- ML Pipeline: Use the Jupyter notebook in `notebook/` for model development
- Frontend: React components in `frontend/`
- Backend: FastAPI server in `backend/`

## API Documentation

FastAPI automatically generates API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 