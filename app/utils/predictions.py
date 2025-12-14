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
    
    # Get weather defaults: Use weighted recent average (last 3 years, weighted by recency)
    # This gives more realistic defaults than state averages while smoothing out year-to-year variability
    county_data = df[(df['State'] == state) & (df['County'] == county)].copy()
    
    # Weather feature columns
    weather_cols = [
        col for category in FEATURE_CATEGORIES.values() 
        for col in category 
        if any(x in col.lower() for x in ['gdd', 'temp', 'precip', 'heat', 'rh', 'stress', 'anomaly', 'weeks', 'humidity'])
    ]
    weather_cols = [col for col in weather_cols if col in feature_columns]
    
    # Calculate weighted recent average (prefer latest year, but use 3-year average if available)
    weather_defaults = {}
    if len(county_data) > 0:
        county_data_sorted = county_data.sort_values('Year', ascending=False)
        recent_data = county_data_sorted.head(3)  # Last 3 years
        
        if len(recent_data) > 0:
            # Weighted average: most recent year gets 50%, previous year 30%, year before 20%
            weights = [0.5, 0.3, 0.2][:len(recent_data)]
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            for col in weather_cols:
                if col in recent_data.columns:
                    values = recent_data[col].values
                    valid_mask = pd.notna(values)
                    if valid_mask.any():
                        # Only use valid values and their corresponding weights
                        valid_values = values[valid_mask]
                        valid_weights = weights[valid_mask]
                        # Renormalize weights for valid values only
                        valid_weights = valid_weights / valid_weights.sum()
                        weighted_avg = np.average(valid_values, weights=valid_weights)
                        weather_defaults[col] = float(weighted_avg)
                    else:
                        weather_defaults[col] = None
                else:
                    weather_defaults[col] = None
    else:
        # No county data - use state average as fallback
        state_data = df[df['State'] == state]
        state_averages = state_data[weather_cols].mean().to_dict()
        weather_defaults = state_averages
    
    # Calculate state averages for final fallback
    state_data = df[df['State'] == state]
    state_averages = state_data[weather_cols].mean().to_dict()
    
    # Set weather features (user overrides > weighted recent average > state average > 0.0)
    for col in weather_cols:
        if col not in feature_columns:
            continue
        
        if weather_overrides and col in weather_overrides:
            # User override takes priority
            features[col] = weather_overrides[col]
        elif col in weather_defaults and weather_defaults[col] is not None and pd.notna(weather_defaults[col]):
            # Use weighted recent average
            features[col] = float(weather_defaults[col])
        elif col in state_averages and pd.notna(state_averages[col]):
            # Fallback to state average
            features[col] = float(state_averages[col])
        else:
            # Last resort: use 0.0 (but log warning)
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
    Make a yield prediction. For years beyond available data, uses iterative
    prediction where each year's prediction is based on previous predictions.
    
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
    if df is None:
        df = load_data()
    
    # Get maximum year in dataset
    max_year = int(df['Year'].max())
    
    # Load model and scaler (only load once, reuse for iterations)
    model = load_model(model_name)
    scaler = load_scaler()
    model_metadata = get_model_metadata(model_name)
    feature_columns = load_feature_columns()  # Load once, reuse
    
    # If predicting beyond available data, use iterative prediction
    if year > max_year:
        # Store predictions for intermediate years
        intermediate_predictions = {}
        
        # Get base historical data (actual data up to max_year)
        base_historical = get_historical_yields(df, state, county, current_year=max_year + 1)
        
        # Iteratively predict from (max_year + 1) to target year
        for target_year in range(max_year + 1, year + 1):
            # Build combined history: actual data + previous predictions
            if len(intermediate_predictions) > 0:
                # Create DataFrame with predicted yields
                predicted_df = pd.DataFrame([
                    {'Year': pred_year, 'Yield_BU_ACRE': yield_val}
                    for pred_year, yield_val in sorted(intermediate_predictions.items())
                ])
                
                # Combine with actual historical data
                if len(base_historical) > 0:
                    combined_history = pd.concat([base_historical, predicted_df], ignore_index=True)
                    combined_history = combined_history.sort_values('Year').reset_index(drop=True)
                else:
                    combined_history = predicted_df.sort_values('Year').reset_index(drop=True)
            else:
                # First iteration: use only actual historical data
                combined_history = base_historical.copy()
            
            # Prepare features for this target year
            # Only apply weather/soil overrides for the final target year
            use_overrides = (target_year == year)
            
            # Get all features using prepare_features with combined history
            # prepare_features will use combined_history for lag features
            features_df = prepare_features(
                state, county, target_year,
                historical_data=combined_history,  # Pass combined history (actual + predictions)
                weather_overrides=weather_overrides if use_overrides else None,
                soil_overrides=soil_overrides if use_overrides else None,
                df=df
            )
            
            # Ensure all required columns exist and in correct order
            for col in feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0.0
            features_df = features_df[feature_columns]
            
            # Scale features if needed
            if model_name == 'ridge':
                features_scaled = scaler.transform(features_df)
                features_final = pd.DataFrame(features_scaled, columns=feature_columns)
            else:
                features_final = features_df
            
            # Make prediction for this year
            prediction = model.predict(features_final)[0]
            
            # Stabilize prediction: prevent downward spiral by blending with historical baseline
            # Use actual historical average (not combined with predictions) as baseline
            if len(base_historical) >= 3:
                historical_avg = float(base_historical['Yield_BU_ACRE'].tail(3).mean())
                last_historical = float(base_historical['Yield_BU_ACRE'].iloc[-1])
                
                # Calculate trend: average year-over-year change over last 3-5 years
                historical_sorted = base_historical.sort_values('Year')
                if len(historical_sorted) >= 2:
                    year_over_year_changes = historical_sorted['Yield_BU_ACRE'].diff().dropna()
                    historical_trend = float(year_over_year_changes.tail(min(5, len(year_over_year_changes))).mean()) if len(year_over_year_changes) > 0 else 1.5
                else:
                    historical_trend = 1.5
                
                prediction_val = float(prediction)
                years_ahead = target_year - max_year
                
                # Ensure trend is at least 1.5 BU/year (conservative technology improvement assumption)
                trend_adjustment = max(1.5, historical_trend) if historical_trend > 0 else 1.5
                
                # Calculate expected yield: don't let it drop below historical average minus small buffer
                # Allow for up to 10% drop from historical average for bad years, but maintain trend
                min_acceptable = historical_avg * 0.90  # Don't go below 90% of recent average
                expected_yield = max(
                    min_acceptable,
                    historical_avg + (years_ahead * trend_adjustment)
                )
                
                # Strong stabilization: if prediction drops too much, blend heavily with expected
                if prediction_val < expected_yield:
                    # More aggressive blending: 50% prediction, 50% expected
                    prediction = prediction_val * 0.5 + expected_yield * 0.5
                # If prediction is way too high (>140% of expected), also stabilize
                elif prediction_val > expected_yield * 1.4:
                    prediction = prediction_val * 0.8 + expected_yield * 0.2
            elif len(base_historical) > 0:
                # Fallback: use last historical value with upward adjustment
                last_historical = float(base_historical['Yield_BU_ACRE'].iloc[-1])
                prediction_val = float(prediction)
                years_ahead = target_year - max_year
                expected = last_historical + (years_ahead * 1.5)  # 1.5 BU/year improvement
                min_acceptable = last_historical * 0.90
                expected = max(min_acceptable, expected)
                if prediction_val < expected:
                    prediction = prediction_val * 0.6 + expected * 0.4
            
            intermediate_predictions[target_year] = float(prediction)
            
            # Store features for final return (only for the target year)
            if target_year == year:
                final_features = features_df.to_dict('records')[0]
        
        # Final prediction is for the target year
        prediction = intermediate_predictions[year]
        
    else:
        # For years within dataset range, use standard prediction
        features = prepare_features(
            state, county, year,
            weather_overrides=weather_overrides,
            soil_overrides=soil_overrides,
            df=df
        )
        
        # Scale features if needed
        if model_name == 'ridge':
            features_scaled = scaler.transform(features)
            features_final = pd.DataFrame(features_scaled, columns=features.columns)
        else:
            features_final = features
        
        # Make prediction
        prediction = model.predict(features_final)[0]
        final_features = features.to_dict('records')[0]
    
    # Get confidence interval (using MAE as proxy)
    # Increase uncertainty for predictions further in future
    years_ahead = max(0, year - max_year)
    base_mae = model_metadata.get('mae', 11.22)
    if years_ahead > 0:
        # Increase uncertainty by 5% per year into future
        # (accounts for accumulated error in iterative predictions)
        uncertainty_factor = 1 + (years_ahead * 0.05)
        mae = base_mae * uncertainty_factor
    else:
        mae = base_mae
    
    confidence_interval = (prediction - mae, prediction + mae)
    
    result = {
        'predicted_yield': float(prediction),
        'confidence_lower': float(confidence_interval[0]),
        'confidence_upper': float(confidence_interval[1]),
        'mae': mae,
        'model_name': model_name,
        'features_used': final_features
    }
    
    # Add intermediate predictions if we did iterative prediction
    if year > max_year and 'intermediate_predictions' in locals():
        result['intermediate_predictions'] = intermediate_predictions
    
    return result

