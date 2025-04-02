from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
# Fix the import path to find model.py
try:
    # Try relative import first (if in the same directory)
    from .model import ObesityRiskModel, load_model, predict_single
except ImportError:
    try:
        # Try direct import (if in the Python path)
        from model import ObesityRiskModel, load_model, predict_single
    except ImportError:
        # Fallback to src.model as a last resort
        try:
            from src.model import ObesityRiskModel, load_model, predict_single
        except ImportError:
            import logging
            logging.error("Failed to import model module. API will run with limited functionality.")
            
            # Define dummy classes/functions for basic operation
            class ObesityRiskModel:
                def __init__(self):
                    self.risk_levels = ['Low', 'Medium', 'High']
                
                def load(self):
                    pass
                
                def predict(self, features):
                    return "Medium", {"Low": 0.2, "Medium": 0.6, "High": 0.2}
            
            def load_model():
                return ObesityRiskModel()
            
            def predict_single(model, features):
                return model.predict(features)

import pandas as pd
import logging
import traceback
import os
from typing import Optional, Dict, List
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
import pickle
from io import BytesIO
import json
import sys
from enum import Enum

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create necessary directories
models_dir = os.path.join(PROJECT_ROOT, "models")
os.makedirs(models_dir, exist_ok=True)

logs_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'api.log'))
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://obesity-risk-frontend.vercel.app",  # Vercel
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a region encoding dictionary
REGION_ENCODING = {
    "North-Central": 0,
    "Northeast": 1,
    "Northwest": 2,
    "South-South": 3,
    "Southeast": 4,
    "Southwest": 5,
    "National": 6
}

# Define column names in the correct order
FEATURE_COLUMNS = [
    'region',
    'bmi',
    'age',
    'gender',
    'blood_pressure_systolic',
    'blood_pressure_diastolic',
    'blood_sugar',
    'cholesterol',
    'physical_activity',
    'diet_quality'
]

# Define column mappings for flexibility
COLUMN_MAPPINGS = {
    # Standard names
    'Region': 'region',
    'Age': 'age',
    'Gender': 'gender',
    'BMI': 'bmi',
    'Blood_Pressure_Systolic': 'blood_pressure_systolic',
    'Blood_Pressure_Diastolic': 'blood_pressure_diastolic',
    'Cholesterol': 'cholesterol',
    'Blood_Sugar': 'blood_sugar',
    'Physical_Activity': 'physical_activity',
    'Diet_Quality': 'diet_quality',
    'Obesity_Risk': 'obesity_risk',
    
    # Alternative names
    'region': 'region',
    'age': 'age',
    'gender': 'gender',
    'bmi': 'bmi',
    'blood_pressure_systolic': 'blood_pressure_systolic',
    'blood_pressure_diastolic': 'blood_pressure_diastolic',
    'cholesterol': 'cholesterol',
    'blood_sugar': 'blood_sugar',
    'physical_activity': 'physical_activity',
    'diet_quality': 'diet_quality',
    'obesity_risk': 'obesity_risk',
    
    # Common variations
    'systolic': 'blood_pressure_systolic',
    'diastolic': 'blood_pressure_diastolic',
    'bp_systolic': 'blood_pressure_systolic',
    'bp_diastolic': 'blood_pressure_diastolic',
    'physical': 'physical_activity',
    'diet': 'diet_quality',
    'risk': 'obesity_risk'
}

def map_columns(df):
    """Map DataFrame columns to standard names based on COLUMN_MAPPINGS."""
    mapped_columns = {}
    for col in df.columns:
        # Try exact match first
        if col in COLUMN_MAPPINGS:
            mapped_columns[col] = COLUMN_MAPPINGS[col]
        else:
            # Try case-insensitive match
            col_lower = col.lower()
            matches = [
                standard_name
                for input_name, standard_name in COLUMN_MAPPINGS.items()
                if input_name.lower() == col_lower
            ]
            if matches:
                mapped_columns[col] = matches[0]
    
    return mapped_columns

# Load the model
model_instance = None
try:
    model_instance = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())

class FeatureData(BaseModel):
    age: int
    gender: str
    height: float
    weight: float
    family_history: str
    favc: str
    fcvc: str
    ncp: int
    caec: str
    smoke: str
    ch2o: float
    scc: str
    faf: str
    tue: float
    calc: str
    mtrans: str

class PhysicalActivity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"

class PredictionInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age in years")
    gender: Gender = Field(..., description="Gender (male/female)")
    height: float = Field(..., ge=50, le=250, description="Height in cm")
    weight: float = Field(..., ge=20, le=300, description="Weight in kg")
    physical_activity: PhysicalActivity = Field(..., description="Physical activity level (low/medium/high)")

class PredictionResponse(BaseModel):
    risk_level: str
    probabilities: Dict[str, float]

class RetrainingResponse(BaseModel):
    status: str
    message: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None

@app.get("/")
async def root():
    return {"message": "Obesity Risk Prediction API is running"}

@app.get("/health")
async def health_check():
    if model_instance is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "Model loaded and ready"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionInput):
    try:
        if model_instance is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Calculate BMI
        height_m = data.height / 100  # Convert cm to m
        bmi = data.weight / (height_m ** 2)
        logger.info(f"Calculated BMI: {bmi}")
        
        # Map gender to numeric (0 for female, 1 for male)
        gender_numeric = 1 if data.gender == Gender.MALE else 0
        
        # Map physical activity to numeric (1=low, 2=medium, 3=high)
        activity_map = {
            PhysicalActivity.LOW: 1,
            PhysicalActivity.MEDIUM: 2,
            PhysicalActivity.HIGH: 3
        }
        physical_activity_numeric = activity_map[data.physical_activity]
        
        # Create feature array with only the features we trained on
        features = np.array([
            float(bmi),
            float(data.age),
            float(gender_numeric),
            float(physical_activity_numeric)
        ])
        
        logger.info(f"Input features: {features}")
        
        # Get prediction
        risk_level, probabilities = model_instance.predict(features)
        logger.info(f"Prediction successful. Risk level: {risk_level}, Probabilities: {probabilities}")
        
        return PredictionResponse(
            risk_level=risk_level,
            probabilities=probabilities
        )
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/upload")
async def validate_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Read the file content
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        
        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower()
        
        # Check for required columns (flexible approach)
        # Define column patterns to look for
        required_patterns = [
            ['age'],
            ['gender', 'sex'],
            ['height', 'stature'],
            ['weight', 'mass'],
            ['physical', 'activity', 'exercise', 'faf']
        ]
        
        # For each pattern group, check if at least one column matches
        missing_columns = []
        for pattern_group in required_patterns:
            if not any(any(pattern in col for pattern in pattern_group) for col in df.columns):
                missing_columns.append('/'.join(pattern_group))
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Save the validated file for retraining
        save_path = os.path.join(PROJECT_ROOT, "data", "train", "training_data.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            # Reset file pointer to beginning
            f.write(content)
        
        logger.info(f"File validated and saved to {save_path}")
        
        return {
            "status": "success",
            "message": f"File validated successfully",
            "rows": len(df),
            "columns": len(df.columns),
            "column_list": list(df.columns)
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Error parsing CSV file")
    except Exception as e:
        logger.error(f"Error validating file: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_data(df):
    """Preprocess the data by encoding categorical variables and handling missing values."""
    # Create label encoders for categorical columns
    categorical_columns = ['Region', 'Gender', 'Physical_Activity', 'Diet_Quality']
    label_encoders = {}
    
    # Encode categorical variables
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    # Convert all remaining columns to numeric, replacing any errors with median
    numeric_columns = [col for col in df.columns if col not in categorical_columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if df[column].isnull().any():
            df[column] = df[column].fillna(df[column].median())
    
    return df, label_encoders

@app.post("/validate")
async def validate_csv_format(file: UploadFile = File(...)):
    """Validate the uploaded CSV file"""
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        
        # Convert column names to lowercase for case-insensitive comparison
        df.columns = df.columns.str.lower()
        
        # Define column patterns to look for
        column_patterns = {
            'age': ['age', 'years'],
            'gender': ['gender', 'sex'],
            'height': ['height', 'heightcm', 'height_cm'],
            'weight': ['weight', 'weightkg', 'weight_kg'],
            'physical_activity': ['physical', 'activity', 'exercise', 'faf'],
            'diet': ['diet', 'nutrition', 'eating', 'favc', 'fcvc']
        }
        
        # Find matching columns
        found_columns = {}
        for key, patterns in column_patterns.items():
            try:
                for col in df.columns:
                    if any(pattern in col.lower() for pattern in patterns):
                        found_columns[key] = col
                        break
            except Exception as e:
                logger.warning(f"Error processing column pattern {key}: {str(e)}")
                continue
        
        # Log found columns
        logger.info(f"Found matching columns: {found_columns}")
        
        # Save the file for training
        os.makedirs("data/train", exist_ok=True)
        file_path = os.path.join("data/train", "training_data.csv")
        df.to_csv(file_path, index=False)
        
        return {
            "status": "success",
            "message": "File validated successfully",
            "rows": len(df),
            "matched_columns": found_columns,
            "all_columns": list(df.columns)
        }
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/visualize/data")
async def get_visualization_data():
    """Generate visualization data from the training dataset"""
    try:
        # Use the Nigeria Obesity Risk dataset
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "train",
            "Nigeria Obesity Risk Synthetic Data - Synthetic_Health_Dataset.csv"
        )
        
        if not os.path.exists(dataset_path):
            raise HTTPException(
                status_code=404,
                detail="Dataset not found. Please ensure the Nigeria Obesity Risk dataset is in the correct location."
            )
        
        logger.info(f"Reading dataset from: {dataset_path}")
        
        # Read the dataset
        df = pd.read_csv(dataset_path)
        df.columns = df.columns.str.lower()  # Convert to lowercase for consistency
        
        # Initialize response data
        response_data = {}
        
        # 1. Risk Distribution
        if 'obesity_risk' in df.columns:
            risk_counts = df['obesity_risk'].value_counts().sort_index()
            risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
            risk_values = [int(risk_counts.get(i, 0)) for i in range(3)]
            
            response_data["risk_distribution"] = {
                'labels': risk_labels,
                'data': risk_values
            }
        else:
            # Generate simulated risk distribution data if real data isn't available
            logger.info("Real risk distribution data not found, generating simulated data")
            response_data["risk_distribution"] = {
                'labels': ['Low Risk', 'Medium Risk', 'High Risk'],
                'data': [35, 45, 20]  # Simulated distribution values
            }
        
        # 2. BMI Analysis
        if 'bmi' in df.columns and 'obesity_risk' in df.columns:
            # Create BMI ranges and calculate risk correlation
            bmi_ranges = pd.cut(df['bmi'], bins=10)
            bmi_risk = df.groupby(bmi_ranges)['obesity_risk'].mean()
            
            response_data["bmi_correlation"] = {
                'labels': [f"{round(interval.left, 1)}-{round(interval.right, 1)}" for interval in bmi_risk.index],
                'data': [float(risk) for risk in bmi_risk.values]
            }
        
        # 3. Age Distribution
        if 'age' in df.columns and 'obesity_risk' in df.columns:
            age_ranges = pd.cut(df['age'], bins=6)
            age_risk = df.groupby(age_ranges)['obesity_risk'].mean()
            
            response_data["age_distribution"] = {
                'labels': [f"{round(interval.left)}-{round(interval.right)}" for interval in age_risk.index],
                'data': [float(val) for val in age_risk.values]
            }
        
        # 4. Gender Distribution
        if 'gender' in df.columns and 'obesity_risk' in df.columns:
            gender_risk = df.groupby('gender')['obesity_risk'].mean()
            gender_counts = df['gender'].value_counts()
            
            response_data["gender_distribution"] = {
                'labels': gender_risk.index.tolist(),
                'data': [float(risk) for risk in gender_risk.values]
            }
        
        if not response_data:
            raise HTTPException(
                status_code=400,
                detail="No suitable data found for visualization in the Nigeria Obesity Risk dataset"
            )
        
        logger.info("Successfully generated visualization data")
        logger.info(f"Generated visualizations: {list(response_data.keys())}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error generating visualization data: {str(e)}"
        )

@app.post("/retrain", response_model=RetrainingResponse)
async def retrain_model(
    file: UploadFile = File(...),
):
    """
    Retrain the model using a new CSV file
    """
    logger.info("Starting model retraining process")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a unique file path for the uploaded file
    file_path = os.path.join(PROJECT_ROOT, "data", "train", file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        # Save the uploaded file
        logger.info(f"Using uploaded file: {file.filename}")
        contents = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        # Load the training data
        try:
            # Reset file position to beginning to read it again
            df = pd.read_csv(file_path)
            logger.info(f"Loaded training data with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Convert column names to lowercase for easier detection
            lower_cols = [col.lower() if isinstance(col, str) else col for col in df.columns]
            logger.info(f"Columns found in dataset: {lower_cols}")
            
            # Attempt to identify dataset type
            dataset_type = None
            if 'nobeyesdad' in lower_cols:
                dataset_type = 'obesity_dataset'
                logger.info("Detected obesity dataset format with 'nobeyesdad' column")
            
            # Import prepare_data and train_model from train_model.py
            from .train_model import prepare_data, train_model
            
            # Prepare data for training
            try:
                # Try with detected dataset type first
                X, y = prepare_data(df, dataset_type=dataset_type)
                
                # Train the model
                model, scaler, accuracy, f1 = train_model(X, y)
                
                return RetrainingResponse(
                    status="success",
                    message="Model retrained successfully",
                    accuracy=float(accuracy),
                    f1_score=float(f1)
                )
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # If preparation fails, try synthetic data generation
                try:
                    logger.info("Attempting to train with synthetic target generation")
                    
                    # Force synthetic data generation by passing dataset_type=None
                    X, y = prepare_data(df, dataset_type=None)
                    
                    # Train the model with synthetic data
                    model, scaler, accuracy, f1 = train_model(X, y)
                    
                    return RetrainingResponse(
                        status="success",
                        message="Model retrained successfully with synthetic targets",
                        accuracy=float(accuracy),
                        f1_score=float(f1)
                    )
                except Exception as inner_e:
                    logger.error(f"Error during synthetic data training: {str(inner_e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Failed to train model: {str(e)}. Synthetic attempt also failed: {str(inner_e)}"
                    )
                
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="The uploaded file is empty")
        except pd.errors.ParserError:
            raise HTTPException(status_code=400, detail="Error parsing CSV file. Please ensure it's a valid CSV format.")
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
    except Exception as e:
        logger.error(f"Error in retraining endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Retrain failed: {str(e)}")

def prepare_prediction_features(input_data: dict) -> List[float]:
    """Prepare features for prediction in the expected order"""
    logging.info(f"Preparing features from input data: {input_data}")
    
    # Calculate BMI
    try:
        height_m = float(input_data['height']) / 100  # convert cm to m
        weight_kg = float(input_data['weight'])
        bmi = weight_kg / (height_m * height_m)
        logging.info(f"Calculated BMI: {bmi}")
    except (ValueError, KeyError, ZeroDivisionError) as e:
        logging.error(f"Error calculating BMI: {str(e)}")
        # Default to average BMI if calculation fails
        bmi = 25.0
        logging.info(f"Using default BMI: {bmi}")
    
    # Process age
    try:
        age = float(input_data['age'])
        # Clip age to reasonable range
        age = max(1, min(age, 120))
    except (ValueError, KeyError):
        age = 45.0  # Default to middle age
        logging.info(f"Using default age: {age}")
    
    # Process gender
    gender_numeric = 1.0  # Default to male
    if 'gender' in input_data:
        gender = str(input_data['gender']).lower()
        if any(m in gender for m in ['f', 'female', 'woman', '0']):
            gender_numeric = 0.0
        elif any(m in gender for m in ['m', 'male', 'man', '1']):
            gender_numeric = 1.0
        logging.info(f"Mapped gender '{input_data['gender']}' to {gender_numeric}")
    
    # Process physical activity
    try:
        if 'faf' in input_data:
            faf = str(input_data['faf']).lower()
            # Map to numeric scale (1=low, 2=medium, 3=high)
            if faf in ['1', 'low', 'rarely', 'never']:
                physical_activity = 1.0
            elif faf in ['3', 'high', 'always', 'daily']:
                physical_activity = 3.0
            else:
                physical_activity = 2.0  # Medium or default
        else:
            physical_activity = 2.0  # Default to medium
    except Exception:
        physical_activity = 2.0  # Default to medium
    logging.info(f"Mapped physical activity to: {physical_activity}")
    
    # Calculate diet quality score (0-10)
    diet_factors = [
        input_data.get('favc', '').lower() == 'no',  # Healthy if not frequent high-caloric food
        input_data.get('fcvc', '') == '3',           # Healthy if high vegetable consumption
        input_data.get('calc', '').lower() == 'yes', # Healthy if monitoring calories
        input_data.get('caec', '').lower() == 'no'   # Healthy if not eating between meals
    ]
    diet_quality = sum(1 for factor in diet_factors if factor) * 2.5  # Scale to 0-10
    logging.info(f"Calculated diet quality score: {diet_quality}")
    
    # Get or set default values for other health metrics
    try:
        blood_pressure = float(input_data.get('blood_pressure', 120))
    except (ValueError, TypeError):
        blood_pressure = 120.0  # Default systolic BP
    
    try:
        diastolic_bp = float(input_data.get('diastolic_bp', 80))
    except (ValueError, TypeError):
        diastolic_bp = 80.0  # Default diastolic BP
        
    try:
        blood_sugar = float(input_data.get('blood_sugar', 90))
    except (ValueError, TypeError):
        blood_sugar = 90.0  # Default blood sugar level
        
    try:
        cholesterol = float(input_data.get('cholesterol', 180))
    except (ValueError, TypeError):
        cholesterol = 180.0  # Default cholesterol level
    
    # Create feature list in the correct order
    # Note: this order must match the order used during training
    features = [
        float(0),         # Default region code
        float(bmi),       # BMI calculated from height & weight
        float(age),       # Age in years
        float(gender_numeric),  # Gender (1=male, 0=female)
        float(blood_pressure),  # Systolic blood pressure
        float(diastolic_bp),    # Diastolic blood pressure
        float(blood_sugar),     # Blood sugar level
        float(cholesterol),     # Cholesterol level
        float(physical_activity),  # Physical activity (1-3)
        float(diet_quality)     # Diet quality score (0-10)
    ]
    
    logging.info(f"Final prepared features: {features}")
    return features

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
