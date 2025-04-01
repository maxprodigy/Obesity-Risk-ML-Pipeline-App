# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.api import predict, retrain

app = FastAPI(
    title="Obesity Risk Prediction API",
    description="ML backend for classification, visualization, and retraining",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(retrain.router, prefix="/retrain", tags=["Retraining"])

@app.get("/")
def root():
    return {"message": "Obesity Risk API is running."}