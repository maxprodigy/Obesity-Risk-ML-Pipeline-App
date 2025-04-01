import joblib
import os
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline

def examine_model():
    print("Testing model loading and structure...")
    
    # Try loading both model files
    pkl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "models", "xgb_obesity_model.pkl")
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "models", "xgb_obesity_model.json")
    
    print("\nTrying PKL file...")
    try:
        pkl_model = joblib.load(pkl_path)
        print(f"PKL Model type: {type(pkl_model)}")
        if isinstance(pkl_model, Pipeline):
            print("Pipeline steps:")
            for name, step in pkl_model.named_steps.items():
                print(f"- {name}: {type(step)}")
        elif isinstance(pkl_model, xgb.XGBClassifier):
            print("Direct XGBoost classifier")
            print(f"Feature names: {pkl_model.feature_names_in_ if hasattr(pkl_model, 'feature_names_in_') else 'None'}")
    except Exception as e:
        print(f"Error loading PKL: {str(e)}")

    print("\nTrying JSON file...")
    try:
        # Try loading as raw booster
        json_model = xgb.Booster()
        json_model.load_model(json_path)
        print("Successfully loaded JSON model")
        
        # Test with sample data
        print("\nTesting prediction with sample data...")
        sample_data = np.array([[5.0, 23.0, 44.0, 1.0, 65.0, 57.0, 45.0, 121.0, 1.0, 6.3]], dtype=np.float32)
        
        # Try prediction with JSON model
        dmatrix = xgb.DMatrix(sample_data)
        json_pred = json_model.predict(dmatrix)
        print(f"JSON model prediction shape: {json_pred.shape}")
        print(f"JSON model prediction: {json_pred}")
        
    except Exception as e:
        print(f"Error with JSON model: {str(e)}")

if __name__ == "__main__":
    examine_model() 