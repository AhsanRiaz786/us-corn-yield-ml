"""
Model loading utilities for the Streamlit dashboard.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import joblib
import streamlit as st
from app.config import (
    MODEL_FILES, SCALER_FILE, FEATURE_COLUMNS_FILE, 
    MODEL_METADATA, MODELS_DIR
)


@st.cache_resource
def load_model(model_name='xgboost'):
    """
    Load a trained model with caching.
    
    Args:
        model_name: Name of model to load ('xgboost', 'random_forest', etc.)
    
    Returns:
        Trained model object
    """
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FILES.keys())}")
    
    model_path = MODEL_FILES[model_name]
    
    if not model_path.exists():
        # Return None instead of crashing, so UI can handle it
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None


@st.cache_resource
def load_state_encoder():
    """Load the state encoder mapping."""
    encoder_path = MODELS_DIR / 'state_encoder.pkl'
    if not encoder_path.exists():
        return None
    try:
        return joblib.load(encoder_path)
    except Exception as e:
        st.error(f"Error loading state encoder: {e}")
        return None


@st.cache_resource
def load_scaler():
    """Load the feature scaler with caching."""
    if not SCALER_FILE.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_FILE}")
    
    scaler = joblib.load(SCALER_FILE)
    return scaler


@st.cache_resource
def load_feature_columns():
    """Load the feature column names with caching."""
    if not FEATURE_COLUMNS_FILE.exists():
        raise FileNotFoundError(f"Feature columns file not found: {FEATURE_COLUMNS_FILE}")
    
    with open(FEATURE_COLUMNS_FILE, 'rb') as f:
        feature_columns = joblib.load(f)
    
    return feature_columns


def get_model_metadata(model_name):
    """Get metadata for a model."""
    return MODEL_METADATA.get(model_name, {})


def get_available_models():
    """Get list of available model names (files exist)."""
    available = []
    for name, path in MODEL_FILES.items():
        if path.exists():
            available.append(name)
    return available

