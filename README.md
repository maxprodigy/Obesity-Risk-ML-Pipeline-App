---

# Obesity Risk Prediction System

A full-stack machine learning web application that predicts obesity risk based on health and lifestyle indicators. The system integrates a trained XGBoost model with a modern React frontend and a FastAPI backend, enabling real-time predictions, model retraining, and interactive data visualizations.

---

## Video Presentation

**Watch the walkthrough here**: *[https://youtu.be/XrULIUyNBNE?si=kOiT4chrQXs63JD-]*  

---

## Project Overview

**Goal**: Build an intelligent system that predicts obesity risk and supports model retraining and visualization for both research and practical public health use.

This tool is suitable for:
- Health professionals and researchers
- Developers seeking ML web integration examples
- Public health policy analysts

---

## Project Structure

```
obesity-risk-predictor/
├── frontend/               # React app (user interface)
│   └── src/
├── backend/                # FastAPI app (API logic)
│   └── api/
│       ├── predict.py
│       ├── retrain.py
│       └── main.py
├── models/                 # Trained ML model and scaler
├── data/                   # Datasets
│   └── train/
├── notebook/               # Model development notebook
├── src/                    # ML training and preprocessing logic
│   ├── train_model.py
│   ├── preprocessing.py
│   └── model.py
├── Dockerfile.*            # Deployment configuration
├── .env.production         # Environment variables
└── README.md
```

---

## Features

### Machine Learning
- Predicts obesity risk using XGBoost
- Trained on synthetic and real datasets
- Supports retraining with uploaded CSVs
- Generates and stores feature importance

### Web Application
- React + Material UI [frontend](https://obesity-risk-frontend.vercel.app/)
- FastAPI [backend](https://obesity-risk-api.onrender.com/) with REST endpoints
- Data visualizations by risk category and health metrics
- Health check and status indicators

### Stack Used
- React (Frontend)
- FastAPI (Backend)
- XGBoost, Pandas, Scikit-Learn (ML)
- Docker (Deployment)

---

## Setup Instructions (Local)

### Prerequisites
- Python 3.8+
- Node.js + npm
- Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/maxprodigy/obesity-risk-backend.git
cd obesity-risk-backend
```

---

### 2. Start the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload
```

API Docs:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run start
```

App runs at: [http://localhost:3000](http://localhost:3000)

---

## API Endpoints

| Endpoint           | Method | Description                             |
|--------------------|--------|-----------------------------------------|
| `/predict`         | POST   | Predicts obesity risk from input data   |
| `/retrain`         | POST   | Retrains the model using new CSV data   |
| `/validate`        | POST   | Validates CSV structure before retraining |
| `/visualize/data`  | GET    | Returns data for charts                 |
| `/health`          | GET    | Returns API health status               |

---

## Model Behavior

### Input Features
- Age
- Gender
- Height and Weight (used to calculate BMI)
- Physical Activity

### Output
- Obesity Risk Class: Low, Medium, High
- Confidence levels for each class

### Model Files
Trained models are stored in:

```
/models/
├── model.joblib       # Trained XGBoost model
├── scaler.joblib      # StandardScaler instance
```

These are automatically updated after retraining.

---

## Sample Visualizations

> 📷 **Screenshot Example**  

![Obesity Risk Visualization](https://github.com/maxprodigy/obesity-risk-backend/blob/main/notebook/Screenshot%202025-04-02%20185347.png)

Includes:
- Risk distribution (low/medium/high)
- BMI vs risk correlation
- Age group analysis
- Gender-based risk trends

---

## Testing

Run local tests with:
```bash
python src/test_model.py
python src/test_api.py
```

---

## Deployment

- **Frontend**: Vercel (auto-deployed via GitHub)
- **Backend**: Render (Dockerfile backend setup)
- Ensure `.env.production` in frontend contains:
  ```env
  REACT_APP_API_URL=https://obesity-risk-api.onrender.com/
  NODE_ENV=production
  ```

---

## Author

Built by **Peter Johnson**  
For an academic assignment involving ML deployment, full-stack integration, and model lifecycle management.

---
