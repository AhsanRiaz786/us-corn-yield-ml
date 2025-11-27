"""
Prediction utilities for the Streamlit dashboard.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from app.utils.model_loader import load_model, load_scaler, load_feature_columns, get_model_metadata, load_state_encoder
from app.utils.data_loader import get_historical_yields, get_county_soil_data, get_latest_county_data
from app.config import DEFAULT_VALUES, FEATURE_CATEGORIES
from app.utils.data_loader import load_data


def prepare_features(
    state, 
    county, 
    year,
    historical_data=None,
    weather_overrides=None,
    soil_overrides=None,
    df=None
):
    """
    Prepare feature vector for prediction.
    
    Args:
        state: State name
        county: County name
        year: Prediction year
        historical_data: DataFrame with historical yields (optional)
        weather_overrides: Dict of weather feature overrides (optional)
        soil_overrides: Dict of soil feature overrides (optional)
        df: Full dataset (optional, will load if not provided)
    
    Returns:
        pd.DataFrame: Feature vector ready for scaling and prediction
    """
    if df is None:
        df = load_data()
    
    # Get feature column names
    feature_columns = load_feature_columns()
    
    # Initialize feature dictionary
    features = {}
    
    # Load county's latest data for defaults
    latest_data = get_latest_county_data(df, state, county)
    
    # Get historical yields for lag features
    if historical_data is None:
        historical_data = get_historical_yields(df, state, county, current_year=year)
    
    # Historical features
    if len(historical_data) >= 1:
        features['Yield_Lag1'] = historical_data['Yield_BU_ACRE'].iloc[-1]
    else:
        # Use state average if no history
        state_data = df[df['State'] == state]
        features['Yield_Lag1'] = state_data['Yield_BU_ACRE'].mean()
    
    if len(historical_data) >= 2:
        features['Yield_Lag2'] = historical_data['Yield_BU_ACRE'].iloc[-2]
    else:
        state_data = df[df['State'] == state]
        features['Yield_Lag2'] = state_data['Yield_BU_ACRE'].mean()
    
    if len(historical_data) >= 3:
        features['Yield_3yr_Avg'] = historical_data['Yield_BU_ACRE'].tail(3).mean()
    else:
        state_data = df[df['State'] == state]
        features['Yield_3yr_Avg'] = state_data['Yield_BU_ACRE'].mean()
    
    # Get soil data
    soil_data = get_county_soil_data(df, state, county)
    if soil_data is None:
        soil_data = {}
    
    # Soil features (use overrides if provided)
    soil_features = ['Soil_AWC', 'Soil_Clay_Pct', 'Soil_pH', 'Soil_Organic_Matter_Pct']
    for feat in soil_features:
        if soil_overrides and feat in soil_overrides:
            features[feat] = soil_overrides[feat]
        elif soil_data and feat in soil_data and soil_data[feat] is not None:
            features[feat] = soil_data[feat]
        else:
            features[feat] = DEFAULT_VALUES.get(feat, 0.0)
    
    # Get weather defaults from latest data or state average
    if latest_data is not None:
        weather_defaults = latest_data.to_dict()
    else:
        state_data = df[df['State'] == state]
        weather_defaults = state_data.mean().to_dict()
    
    # Weather features (use overrides if provided, else defaults)
    weather_cols = [
        col for category in FEATURE_CATEGORIES.values() 
        for col in category 
        if any(x in col.lower() for x in ['gdd', 'temp', 'precip', 'heat', 'rh', 'stress', 'anomaly', 'weeks', 'humidity'])
    ]
    
    # Calculate state averages for fallback
    state_data = df[df['State'] == state]
    state_averages = state_data[weather_cols].mean().to_dict()
    
    for col in weather_cols:
        if col not in feature_columns:
            continue
        
        if weather_overrides and col in weather_overrides:
            features[col] = weather_overrides[col]
        else:
            # Use state average as default instead of latest year (which might be outlier)
            if col in state_averages and pd.notna(state_averages[col]):
                features[col] = float(state_averages[col])
            else:
                features[col] = 0.0
    
    # Area features
    if latest_data is not None:
        features['Abandonment_Rate'] = float(latest_data.get('Abandonment_Rate', DEFAULT_VALUES['Abandonment_Rate']))
        features['Harvest_Efficiency'] = float(latest_data.get('Harvest_Efficiency', DEFAULT_VALUES['Harvest_Efficiency']))
    else:
        features['Abandonment_Rate'] = DEFAULT_VALUES['Abandonment_Rate']
        features['Harvest_Efficiency'] = DEFAULT_VALUES['Harvest_Efficiency']
    
    # State encoding
    state_encoder = load_state_encoder()
    if state_encoder:
        features['State_Encoded'] = state_encoder.get(state, 0)
    else:
        # Fallback if encoder not found (should not happen in prod)
        # This is risky but better than crashing
        state_encoder_fallback = {state: idx for idx, state in enumerate(sorted(df['State'].unique()))}
        features['State_Encoded'] = state_encoder_fallback.get(state, 0)
    
    # Create feature vector in correct order
    feature_vector = pd.DataFrame([features])
    
    # Ensure all required columns exist
    for col in feature_columns:
        if col not in feature_vector.columns:
            feature_vector[col] = 0.0
    
    # Reorder to match training order
    feature_vector = feature_vector[feature_columns]
    
    return feature_vector


def predict_yield(
    state,
    county,
    year,
    model_name='xgboost',
    weather_overrides=None,
    soil_overrides=None,
    df=None
):
    """
    Make a yield prediction.
    
    Args:
        state: State name
        county: County name
        year: Prediction year
        model_name: Model to use for prediction
        weather_overrides: Dict of weather feature overrides
        soil_overrides: Dict of soil feature overrides
        df: Full dataset (optional)
    
    Returns:
        dict: Prediction results with yield, confidence, and metadata
    """
    # Load model and scaler
    model = load_model(model_name)
    scaler = load_scaler()
    model_metadata = get_model_metadata(model_name)
    
    # Prepare features
    features = prepare_features(
        state, county, year,
        weather_overrides=weather_overrides,
        soil_overrides=soil_overrides,
        df=df
    )
    
    # Scale features ONLY if model requires it (Ridge)
    # XGBoost, Random Forest, and Gradient Boosting were trained on unscaled data
    if model_name == 'ridge':
        features_scaled = scaler.transform(features)
        features_final = pd.DataFrame(features_scaled, columns=features.columns)
    else:
        features_final = features
    
    # Make prediction
    prediction = model.predict(features_final)[0]
    
    # Get confidence interval (using MAE as proxy)
    mae = model_metadata.get('mae', 11.22)
    confidence_interval = (prediction - mae, prediction + mae)
    
    return {
        'predicted_yield': float(prediction),
        'confidence_lower': float(confidence_interval[0]),
        'confidence_upper': float(confidence_interval[1]),
        'mae': mae,
        'model_name': model_name,
        'features_used': features.to_dict('records')[0]
    }

