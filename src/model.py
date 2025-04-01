import xgboost as xgb
import os
import numpy as np
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObesityRiskModel:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.risk_levels = ['Low', 'Medium', 'High']
        
    def load(self):
        """Load both the scaler and the XGBoost model"""
        try:
            base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            
            # Load the scaler
            scaler_path = os.path.join(base_path, "scaler.joblib")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
            
            # Load the XGBoost model
            model_path = os.path.join(base_path, "model.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            self.model = joblib.load(model_path)
            logger.info("Loaded XGBoost model successfully")
            
            # Verify model configuration
            if not isinstance(self.model, xgb.XGBClassifier):
                raise TypeError("Loaded model is not an XGBClassifier")
            
            logger.info(f"Model objective: {self.model.objective}")
            logger.info(f"Number of classes: {getattr(self.model, 'n_classes_', 3)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
    def predict(self, features):
        """Make predictions using the scaler and model"""
        try:
            if self.scaler is None or self.model is None:
                raise ValueError("Model not loaded. Call load() first.")
            
            # Convert to numpy array if not already
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)
            
            # Ensure 2D array
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            logger.info(f"Input features shape: {features.shape}")
            
            # Scale the features
            scaled_features = self.scaler.transform(features)
            logger.info(f"Scaled features shape: {scaled_features.shape}")
            
            # Get prediction and probabilities
            probabilities = self.model.predict_proba(scaled_features)
            prediction = self.model.predict(scaled_features)
            
            logger.info(f"Raw prediction: {prediction}")
            logger.info(f"Probabilities shape: {probabilities.shape}")
            
            # Map prediction to risk level
            prediction_idx = int(prediction[0])
            risk_level = self.risk_levels[prediction_idx]
            
            # Create probabilities dictionary
            prob_dict = {level: float(prob) for level, prob in zip(self.risk_levels, probabilities[0])}
            
            logger.info(f"Predicted risk level: {risk_level}")
            logger.info(f"Probabilities: {prob_dict}")
            
            return risk_level, prob_dict
            
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise

def load_model():
    """Load the model from file"""
    model = ObesityRiskModel()
    model.load()
    return model

def predict_single(model, features):
    """Make a single prediction"""
    return model.predict(features)
