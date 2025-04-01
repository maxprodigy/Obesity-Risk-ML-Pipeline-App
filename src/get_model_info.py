"""
Script to get information about the XGBoost model, including:
- Feature names if available
- Feature importance
- Other model parameters that can help with debugging
"""

import os
import sys
import traceback
import numpy as np

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
    print("Getting model information...")
    from model import load_model
    
    # Try to load the model
    print("Loading model...")
    model = load_model()
    print("Model loaded successfully!")
    
    # Print model properties
    print("\nModel properties:")
    print(f"Model type: {type(model).__name__}")
    print(f"Model parameters: {model.get_params()}")
    
    # Try to get feature names if they exist
    try:
        if hasattr(model, 'feature_names_') and model.feature_names_ is not None:
            print(f"\nFeature names: {model.feature_names_}")
        else:
            print("\nModel doesn't have feature_names_ attribute or it's None")
    except Exception as e:
        print(f"Error getting feature names: {str(e)}")
    
    # Try to get feature importances if they exist
    try:
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            importances = model.feature_importances_
            print(f"\nFeature importances: {importances}")
            
            # If we have feature names, print feature importance pairs
            if hasattr(model, 'feature_names_') and model.feature_names_ is not None:
                feature_importance = sorted(zip(model.feature_names_, importances), 
                                           key=lambda x: x[1], reverse=True)
                print("\nFeature importance ranking:")
                for feature, importance in feature_importance:
                    print(f"{feature}: {importance:.4f}")
    except Exception as e:
        print(f"Error getting feature importances: {str(e)}")
    
    # Try to get classes if they exist
    try:
        if hasattr(model, 'classes_') and model.classes_ is not None:
            print(f"\nClasses: {model.classes_}")
    except Exception as e:
        print(f"Error getting classes: {str(e)}")
        
    # If the model is XGBoost, print more specific information
    try:
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            print(f"\nBooster dump (first few lines):")
            dump = booster.get_dump()
            for i, tree in enumerate(dump[:2]):  # Show only first 2 trees
                print(f"Tree {i} (truncated): {tree[:200]}...")
            print(f"Total number of trees: {len(dump)}")
    except Exception as e:
        print(f"Error getting booster information: {str(e)}")
    
    # Test prediction with properly encoded data
    print("\n\nTesting prediction with properly encoded data:")
    region_code = REGION_ENCODING.get("Northwest", 0)
    test_data = np.array([[region_code, 25.0, 35, 1, 120, 80, 90, 180, 1, 7]], dtype=float)
    print(f"Test input (all numeric, region encoded): {test_data}")
    
    try:
        prediction = model.predict(test_data)
        probability = model.predict_proba(test_data)
        print(f"Prediction successful: {prediction[0]} (probability: {probability[0][1]:.4f})")
    except Exception as e:
        print(f"Error during test prediction: {str(e)}")
        print(traceback.format_exc())
    
    print("\nModel information retrieval completed successfully!")
    
except Exception as e:
    print(f"Error analyzing model: {str(e)}")
    print(traceback.format_exc())
    sys.exit(1)

if __name__ == "__main__":
    print("Model analysis complete.") 