import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / 'models'
MODEL_PATH = MODELS_DIR / 'xgboost_model.pkl'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'
COLS_PATH = MODELS_DIR / 'feature_columns.pkl'

def test_model_files_exist():
    """Test if model artifacts exist."""
    assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"
    assert SCALER_PATH.exists(), f"Scaler file not found at {SCALER_PATH}"
    assert COLS_PATH.exists(), f"Feature columns file not found at {COLS_PATH}"

def test_model_prediction():
    """Test if model can make predictions on dummy data."""
    if not MODEL_PATH.exists():
        pytest.skip("Model file not found")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(COLS_PATH)
    
    # Create dummy input with correct shape
    n_features = len(feature_cols)
    dummy_input = pd.DataFrame(np.random.rand(1, n_features), columns=feature_cols)
    
    # Scale
    dummy_scaled = scaler.transform(dummy_input)
    
    # Predict
    prediction = model.predict(dummy_scaled)
    
    assert len(prediction) == 1, "Should return one prediction"
    assert isinstance(prediction[0], (float, np.floating)), "Prediction should be a float"
