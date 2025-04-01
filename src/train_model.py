import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
import joblib
import logging
from sklearn.metrics import accuracy_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def prepare_data(provided_df=None, dataset_type=None):
    """Prepare data for training"""
    logging.info("Starting data preparation")
    
    if provided_df is None:
        raise ValueError("No dataset provided. Please provide a DataFrame for training.")
    
    logging.info(f"Using provided dataset with {provided_df.shape[0]} rows and {provided_df.shape[1]} columns")
    df = provided_df.copy()
    
    # Convert column names to lowercase for consistency
    df.columns = [str(col).lower().strip() for col in df.columns]
    logging.info(f"Available columns: {list(df.columns)}")
    
    # Handle different dataset formats
    if dataset_type == 'obesity_dataset':
        # Convert object columns to string type first
        string_columns = ['gender', 'faf', 'nobeyesdad']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Map the obesity levels to numeric values if present
        if 'nobeyesdad' in df.columns:
            obesity_map = {
                'insufficient_weight': 0,
                'normal_weight': 0,
                'overweight_level_i': 1,
                'overweight_level_ii': 1,
                'obesity_type_i': 2,
                'obesity_type_ii': 2,
                'obesity_type_iii': 2
            }
            df['obesity_risk'] = df['nobeyesdad'].str.lower().map(obesity_map)
            # Fill any NaN values with the most common value
            if df['obesity_risk'].isna().any():
                most_common = df['obesity_risk'].value_counts().idxmax() if not df['obesity_risk'].value_counts().empty else 1
                logging.warning(f"Found {df['obesity_risk'].isna().sum()} missing obesity_risk values. Filling with {most_common}")
                df['obesity_risk'] = df['obesity_risk'].fillna(most_common)
        
        # Map gender if present
        if 'gender' in df.columns:
            df['gender_numeric'] = df['gender'].str.lower().map({'male': 1, 'female': 0, 'm': 1, 'f': 0})
        
        # Map physical activity frequency (FAF) if present
        if 'faf' in df.columns:
            activity_map = {'no': 1, 'sometimes': 2, 'yes': 3}
            df['physical_activity_numeric'] = df['faf'].str.lower().map(activity_map)
        
        # Calculate BMI from weight and height if both present
        if 'weight' in df.columns and 'height' in df.columns:
            height_m = pd.to_numeric(df['height'], errors='coerce')  # Convert to numeric
            weight = pd.to_numeric(df['weight'], errors='coerce')    # Convert to numeric
            df['bmi'] = weight / (height_m ** 2)
            df['bmi'] = df['bmi'].clip(10, 60)
    else:
        # Generic dataset format
        # First check and map columns that might be present
        
        # Gender mapping
        gender_columns = ['gender', 'sex', 'gender_minmax', 'gender_z']
        gender_found = False
        for col in gender_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df['gender_numeric'] = df[col].str.lower().map(
                    lambda x: 1 if x in ['male', 'm', '1', 'man'] else 0 if x in ['female', 'f', '0', 'woman'] else None
                )
                gender_found = True
                logging.info(f"Mapped gender from column: {col}")
                break
        
        if not gender_found:
            logging.warning("No gender column found. Using default values.")
            df['gender_numeric'] = 0  # Default to female
        
        # Physical activity mapping
        activity_columns = ['physical_activity', 'faf', 'activity', 'phys_activity', 'physical_activity_minmax', 'faf_minmax']
        activity_found = False
        for col in activity_columns:
            if col in df.columns:
                if pd.api.types.is_object_dtype(df[col].dtype):
                    df[col] = df[col].astype(str)
                    # Map string values
                    activity_map = {
                        'low': 1, 'no': 1, 'none': 1, 'never': 1, 'rarely': 1,
                        'medium': 2, 'moderate': 2, 'sometimes': 2, 'occasional': 2,
                        'high': 3, 'yes': 3, 'frequent': 3, 'always': 3
                    }
                    df['physical_activity_numeric'] = df[col].str.lower().map(
                        lambda x: activity_map.get(x, 2) if isinstance(x, str) else None
                    )
                else:
                    # If numeric, scale to 1-3 range
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if min_val != max_val:
                        df['physical_activity_numeric'] = 1 + 2 * (df[col] - min_val) / (max_val - min_val)
                    else:
                        df['physical_activity_numeric'] = 2  # Default to medium if all values are the same
                
                activity_found = True
                logging.info(f"Mapped physical activity from column: {col}")
                break
        
        if not activity_found:
            logging.warning("No physical activity column found. Using default values.")
            df['physical_activity_numeric'] = 2  # Default to medium activity
        
        # BMI calculation or mapping
        if 'bmi' in df.columns:
            # Already has BMI
            df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        elif 'weight' in df.columns and 'height' in df.columns:
            # Calculate BMI
            height_m = pd.to_numeric(df['height'], errors='coerce')
            if (height_m > 3).any():  # Heights likely in cm
                height_m = height_m / 100  # Convert cm to m
            
            weight = pd.to_numeric(df['weight'], errors='coerce')
            df['bmi'] = weight / (height_m ** 2)
        else:
            logging.warning("Cannot calculate BMI. Missing weight or height columns.")
            # Create synthetic BMI based on obesity risk if available
            if 'obesity_risk' in df.columns or 'obesity' in df.columns:
                risk_col = 'obesity_risk' if 'obesity_risk' in df.columns else 'obesity'
                risk_values = pd.to_numeric(df[risk_col], errors='coerce')
                # Map risk levels to reasonable BMI values (low: 22, medium: 28, high: 35)
                df['bmi'] = 22 + risk_values * 6.5
            else:
                # Generate reasonable synthetic BMI
                df['bmi'] = np.random.normal(25, 4, size=len(df)).clip(18, 40)
    
    # Handle age
    age_columns = ['age', 'age_bin', 'age_bin_minmax', 'age_minmax']
    age_found = False
    for col in age_columns:
        if col in df.columns:
            df['age'] = pd.to_numeric(df[col], errors='coerce')
            age_found = True
            logging.info(f"Mapped age from column: {col}")
            break
    
    if not age_found:
        logging.warning("No age column found. Using default values.")
        # Generate reasonable synthetic age
        df['age'] = np.random.normal(45, 15, size=len(df)).clip(18, 90)
    
    # Select features for model
    feature_list = [
        'bmi',
        'age',
        'gender_numeric',
        'physical_activity_numeric'
    ]
    
    # Create feature matrix
    X = pd.DataFrame()
    X['bmi'] = df['bmi'] if 'bmi' in df.columns else np.random.normal(25, 4, size=len(df)).clip(18, 40)
    X['age'] = df['age'] if 'age' in df.columns else np.random.normal(45, 15, size=len(df)).clip(18, 90)
    X['gender_numeric'] = df['gender_numeric'] if 'gender_numeric' in df.columns else 0
    X['physical_activity_numeric'] = df['physical_activity_numeric'] if 'physical_activity_numeric' in df.columns else 2
    
    # Fill any missing values with appropriate defaults
    X = X.fillna({
        'bmi': X['bmi'].mean() if not X['bmi'].isnull().all() else 25,
        'age': 45,
        'gender_numeric': 0,
        'physical_activity_numeric': 2
    })
    
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Feature summary:\n{X.describe()}")
    
    # Get or calculate obesity risk
    if 'obesity_risk' in df.columns and not df['obesity_risk'].isna().all():
        # Make sure we have no NaN values before trying to convert to int
        df['obesity_risk'] = df['obesity_risk'].fillna(1)  # Fill any remaining NaNs with medium risk
        y = df['obesity_risk'].astype(int).values
    elif 'nobeyesdad' in df.columns:
        # Try to generate obesity risk from nobeyesdad column
        obesity_map = {
            'insufficient_weight': 0,
            'normal_weight': 0,
            'overweight_level_i': 1,
            'overweight_level_ii': 1,
            'obesity_type_i': 2,
            'obesity_type_ii': 2,
            'obesity_type_iii': 2
        }
        df['obesity_risk'] = df['nobeyesdad'].str.lower().map(obesity_map)
        df['obesity_risk'] = df['obesity_risk'].fillna(1)  # Fill any remaining NaNs with medium risk
        y = df['obesity_risk'].astype(int).values
    else:
        # Calculate synthetic risk based on BMI and other factors
        risk_score = (
            0.4 * (X['bmi'] >= 30).astype(float) +   # High BMI contribution
            0.2 * (X['bmi'] >= 25).astype(float) +   # Overweight contribution
            0.1 * ((X['age'] - 18) / 70) +           # Age contribution
            0.1 * (3 - X['physical_activity_numeric']) / 2  # Physical activity contribution
        )
        
        # Create balanced classes
        y = pd.qcut(risk_score, q=3, labels=[0, 1, 2]).astype(int).values
    
    logging.info(f"Target class distribution: {np.bincount(y)}")
    return X, y

def train_model(X, y):
    """Train the model and return it along with performance metrics"""
    logging.info("Starting model training...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='mlogloss',
        objective='multi:softprob',
        num_class=3
    )
    
    # Fit the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Save the model and scaler
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model using joblib
    model_path = os.path.join(models_dir, "model.joblib")
    joblib.dump(model, model_path)
    logging.info(f"Saved model to: {model_path}")
    
    # Save scaler using joblib
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Saved scaler to: {scaler_path}")
    
    logging.info(f"Training completed with accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")
    
    return model, scaler, accuracy, f1

if __name__ == "__main__":
    try:
        # Load both datasets
        nigeria_dataset_path = os.path.join(PROJECT_ROOT, "data", "train", 
                                          "Nigeria Obesity Risk Synthetic Data - Synthetic_Health_Dataset.csv")
        obesity_dataset_path = os.path.join(PROJECT_ROOT, "data", "train", 
                                          "ObesityDataSet_raw_and_data_synthetic.csv")
        
        datasets = []
        
        # Load Nigeria dataset if available
        if os.path.exists(nigeria_dataset_path):
            logging.info(f"Loading Nigeria dataset from: {nigeria_dataset_path}")
            nigeria_data = pd.read_csv(nigeria_dataset_path)
            X_nigeria, y_nigeria = prepare_data(nigeria_data, dataset_type='nigeria')
            datasets.append((X_nigeria, y_nigeria))
        
        # Load Obesity dataset if available
        if os.path.exists(obesity_dataset_path):
            logging.info(f"Loading Obesity dataset from: {obesity_dataset_path}")
            obesity_data = pd.read_csv(obesity_dataset_path)
            X_obesity, y_obesity = prepare_data(obesity_data, dataset_type='obesity_dataset')
            datasets.append((X_obesity, y_obesity))
        
        if not datasets:
            raise ValueError("No datasets found. Please ensure at least one dataset is available.")
        
        # Combine datasets
        X = pd.concat([X for X, _ in datasets])
        y = np.concatenate([y for _, y in datasets])
        
        logging.info(f"Combined dataset shape: {X.shape}")
        logging.info(f"Class distribution: {np.bincount(y)}")
        
        # Train model on combined data
        model, scaler, accuracy, f1 = train_model(X, y)
        
        # Report final results
        logging.info("===== Final Training Results =====")
        logging.info(f"Test Accuracy: {accuracy:.4f}")
        logging.info(f"Test F1 Score: {f1:.4f}")
        logging.info("Model training completed successfully!")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        exit(1) 