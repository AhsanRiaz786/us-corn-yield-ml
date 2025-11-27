"""
Yield Prediction Page - Main prediction interface for corn yield forecasting.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from app.utils import (
    load_data, get_unique_states, get_counties_by_state,
    get_year_range, get_historical_yields, get_county_soil_data,
    predict_yield
)
from app.utils.model_loader import get_available_models
from app.utils.visualizations import plot_historical_vs_predicted
from app.config import MODEL_METADATA


st.set_page_config(page_title="Yield Prediction", layout="wide")
st.title("Corn Yield Prediction")

# Load data
try:
    df = load_data()
    states = get_unique_states(df)
    year_min, year_max = get_year_range(df)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar - Input panel
st.sidebar.header("Prediction Inputs")

# Location selection
selected_state = st.sidebar.selectbox("State", states, index=0 if 'Iowa' in states else None)

if selected_state:
    counties = get_counties_by_state(df, selected_state)
    if counties:
        selected_county = st.sidebar.selectbox("County", counties)
    else:
        st.sidebar.warning("No counties available for selected state")
        selected_county = None
else:
    selected_county = None

# Year selection
current_year = year_max + 1  # Default to next year
selected_year = st.sidebar.number_input(
    "Prediction Year",
    min_value=year_min + 4,
    max_value=year_max + 10,
    value=current_year
)

# Model selection
st.sidebar.subheader("Model Selection")
st.sidebar.subheader("Model Selection")
model_names = get_available_models()
if not model_names:
    st.error("No models available. Please check deployment.")
    st.stop()
    
model_display_names = [MODEL_METADATA[m]['name'] for m in model_names]
selected_model_idx = st.sidebar.selectbox(
    "Model",
    range(len(model_names)),
    format_func=lambda x: model_display_names[x],
    index=0  # XGBoost
)
selected_model = model_names[selected_model_idx]

# Historical context
if selected_state and selected_county:
    st.sidebar.subheader("Historical Context")
    
    try:
        historical = get_historical_yields(df, selected_state, selected_county, current_year=selected_year)
        
        if len(historical) > 0:
            latest_yield = historical['Yield_BU_ACRE'].iloc[-1]
            avg_yield = historical['Yield_BU_ACRE'].mean()
            
            st.sidebar.metric("Latest Yield", f"{latest_yield:.1f} BU/ACRE")
            st.sidebar.metric("Historical Average", f"{avg_yield:.1f} BU/ACRE")
            
            # Show last 3 years
            if len(historical) >= 3:
                st.sidebar.write("**Last 3 Years:**")
                for _, row in historical.tail(3).iterrows():
                    st.sidebar.write(f"{int(row['Year'])}: {row['Yield_BU_ACRE']:.1f} BU/ACRE")
        else:
            st.sidebar.info("No historical data available for this county")
    except Exception as e:
        st.sidebar.warning(f"Could not load historical data: {str(e)}")

# Advanced options
with st.sidebar.expander("Advanced Options"):
    use_custom_weather = st.checkbox("Override Weather Data", False)
    
    weather_overrides = {}
    if use_custom_weather:
        st.write("**Weather Parameters**")
        weather_overrides['gdd_total'] = st.number_input("Total GDD", value=2800.0, step=50.0)
        weather_overrides['precip_total'] = st.number_input("Total Precipitation (mm)", value=500.0, step=10.0)
        weather_overrides['weeks_heat_stress'] = st.number_input("Heat Stress Weeks", value=2.0, step=0.5)
    
    use_custom_soil = st.checkbox("Override Soil Data", False)
    
    soil_overrides = {}
    if use_custom_soil:
        st.write("**Soil Properties**")
        try:
            default_soil = get_county_soil_data(df, selected_state, selected_county) if selected_state and selected_county else {}
        except:
            default_soil = {}
        
        from app.config import DEFAULT_VALUES
        soil_overrides['Soil_AWC'] = st.number_input(
            "AWC (Available Water Capacity)",
            value=float(default_soil.get('Soil_AWC', DEFAULT_VALUES['Soil_AWC'])),
            step=0.5
        )
        soil_overrides['Soil_Clay_Pct'] = st.number_input(
            "Clay %",
            value=float(default_soil.get('Soil_Clay_Pct', DEFAULT_VALUES['Soil_Clay_Pct'])),
            step=1.0
        )
        soil_overrides['Soil_pH'] = st.number_input(
            "pH",
            value=float(default_soil.get('Soil_pH', DEFAULT_VALUES['Soil_pH'])),
            step=0.1,
            min_value=4.0,
            max_value=9.0
        )
        soil_overrides['Soil_Organic_Matter_Pct'] = st.number_input(
            "Organic Matter %",
            value=float(default_soil.get('Soil_Organic_Matter_Pct', DEFAULT_VALUES['Soil_Organic_Matter_Pct'])),
            step=0.1
        )

# Main content area
if not selected_state or not selected_county:
    st.info("Please select a state and county from the sidebar to make a prediction.")
else:
    # Make prediction button
    if st.button("Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Generating prediction..."):
            try:
                # Make prediction
                result = predict_yield(
                    selected_state,
                    selected_county,
                    selected_year,
                    model_name=selected_model,
                    weather_overrides=weather_overrides if use_custom_weather else None,
                    soil_overrides=soil_overrides if use_custom_soil else None,
                    df=df
                )
                
                # Store in session state
                st.session_state['last_prediction'] = result
                st.session_state['prediction_state'] = selected_state
                st.session_state['prediction_county'] = selected_county
                st.session_state['prediction_year'] = selected_year
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.stop()
    
    # Display results if available
    if 'last_prediction' in st.session_state:
        result = st.session_state['last_prediction']
        
        st.markdown("---")
        st.header("Prediction Results")
        
        # Main prediction display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.metric(
                f"Predicted Yield for {st.session_state.get('prediction_year', selected_year)}",
                f"{result['predicted_yield']:.1f} BU/ACRE",
                delta=f"±{result['mae']:.1f} BU/ACRE"
            )
        
        with col2:
            st.metric(
                "Lower Bound",
                f"{result['confidence_lower']:.1f} BU/ACRE"
            )
        
        with col3:
            st.metric(
                "Upper Bound",
                f"{result['confidence_upper']:.1f} BU/ACRE"
            )
        
        # Comparison to historical
        try:
            historical = get_historical_yields(df, selected_state, selected_county, current_year=selected_year)
            
            if len(historical) > 0:
                latest_yield = historical['Yield_BU_ACRE'].iloc[-1]
                avg_yield = historical['Yield_BU_ACRE'].mean()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    diff_from_latest = result['predicted_yield'] - latest_yield
                    st.metric(
                        "vs. Previous Year",
                        f"{diff_from_latest:+.1f} BU/ACRE",
                        delta=f"{diff_from_latest/latest_yield*100:+.1f}%"
                    )
                
                with col2:
                    diff_from_avg = result['predicted_yield'] - avg_yield
                    st.metric(
                        "vs. Historical Average",
                        f"{diff_from_avg:+.1f} BU/ACRE",
                        delta=f"{diff_from_avg/avg_yield*100:+.1f}%"
                    )
        except:
            pass
        
        # Visualization
        st.subheader("Historical Trend and Prediction")
        
        try:
            historical = get_historical_yields(df, selected_state, selected_county, current_year=selected_year)
            
            if len(historical) > 0:
                fig = plot_historical_vs_predicted(
                    historical,
                    result['predicted_yield'],
                    st.session_state.get('prediction_year', selected_year)
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate historical plot: {str(e)}")
        
        # Model information
        st.subheader("Model Information")
        model_info = MODEL_METADATA[selected_model]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{model_info['r2']:.3f}")
        with col2:
            st.metric("Mean Absolute Error", f"{model_info['mae']:.2f} BU/ACRE")
        with col3:
            st.metric("RMSE", f"{model_info['rmse']:.2f} BU/ACRE")
        
        st.caption(model_info['description'])

