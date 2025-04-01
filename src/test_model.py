"""
Test script to validate that the model can load and make predictions correctly.
This helps isolate if issues are with the model itself or with the API.
"""

import numpy as np
import os
import sys
import traceback

# Add the current directory to path so we can import model.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Region encoding map - must match the one in api.py
REGION_ENCODING = {
    "Northwest": 0,
    "Northeast": 1,
    "Southwest": 2,
    "Southeast": 3,
    "Central": 4,
    "US": 5,
    "North": 6,
    "South": 7,
    "East": 8,
    "West": 9
}

try:
    print("Testing model loading and prediction...")
    from model import load_model
    
    # Try to load the model
    print("Loading model...")
    model = load_model()
    print("Model loaded successfully!")
    
    # Create a sample input for testing
    # Use numeric values for all features including region
    region_name = "Northwest"
    region_code = REGION_ENCODING.get(region_name, 0)
    print(f"Using region: {region_name} (encoded as {region_code})")
    
    # All features as numeric values
    sample_input = np.array([
        [region_code, 25.0, 35, 1, 120, 80, 90, 180, 1, 7]
    ], dtype=float)  # Use float dtype to ensure compatibility
    
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Sample input data: {sample_input}")
    print(f"Sample input dtype: {sample_input.dtype}")
    
    # Try to make a prediction
    print("Making prediction...")
    prediction = model.predict(sample_input)
    probability = model.predict_proba(sample_input)
    
    print(f"Prediction result: {prediction[0]}")
    print(f"Probability: {probability[0]}")
    print(f"Risk level: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")
    
    print("Model test completed successfully!")
    
except Exception as e:
    print(f"Error testing model: {str(e)}")
    print(traceback.format_exc())
    sys.exit(1)

if __name__ == "__main__":
    print("Model testing complete. If you're seeing this, the test ran without crashing.") 