"""
About Page - Project methodology, documentation, and technical details.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st


st.set_page_config(page_title="About", layout="wide")
st.title("About the Project")

# Project Overview
st.header("Project Overview")

st.write("""
This project develops a machine learning system for predicting corn yields at the county level 
across the United States. The system integrates multiple data sources and uses advanced ensemble 
methods to provide accurate yield forecasts.

**Problem Statement:**
Corn yield prediction is crucial for agricultural planning, supply chain management, and food 
security. Traditional methods rely heavily on expert knowledge and simple statistical models. 
This project demonstrates how machine learning can improve prediction accuracy by incorporating 
complex interactions between weather, soil, and historical patterns.

**Objectives:**
- Develop accurate county-level corn yield prediction models
- Integrate data from multiple authoritative sources (USDA, NASA)
- Compare multiple machine learning approaches
- Provide an interactive platform for predictions and analysis
""")

st.markdown("---")

# Data Sources
st.header("Data Sources")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("USDA NASS")
    st.write("""
    **National Agricultural Statistics Service**
    - Historical corn yield data
    - Area planted and harvested
    - Production statistics
    - County-level granularity
    - 1981-2023 coverage
    """)

with col2:
    st.subheader("NASA POWER")
    st.write("""
    **Prediction of Worldwide Energy Resources**
    - Daily weather data
    - Temperature, precipitation, solar radiation
    - Growing season aggregation (April-September)
    - County-level coordinates
    """)

with col3:
    st.subheader("USDA NRCS")
    st.write("""
    **Natural Resources Conservation Service**
    - Soil property data (gSSURGO)
    - Available water capacity
    - pH, clay content, organic matter
    - County-aggregated values
    """)

st.markdown("---")

# Methodology
st.header("Methodology")

st.subheader("Feature Engineering")

feature_tabs = st.tabs([
    "Historical Features",
    "Weather Features",
    "Soil Features",
    "Area Features"
])

with feature_tabs[0]:
    st.write("""
    **Historical Yield Patterns**
    - Lag features (1-3 years prior yields)
    - Rolling averages (3-year, 5-year)
    - Year-over-year changes
    
    These features capture persistence in agricultural productivity due to:
    - Soil quality
    - Farm management practices
    - Infrastructure and expertise
    - Technological adoption
    """)

with feature_tabs[1]:
    st.write("""
    **Weather Features**
    - Growing Degree Days (GDD) by growth stage
    - Heat stress indicators (days above thresholds)
    - Precipitation totals and anomalies
    - Temperature extremes and means
    - Relative humidity
    - Combined stress metrics
    
    Weather is aggregated over the critical growing season (April-September).
    """)

with feature_tabs[2]:
    st.write("""
    **Soil Properties**
    - Available Water Capacity (AWC)
    - Clay percentage
    - pH level
    - Organic matter percentage
    
    These provide context on soil quality and water retention capacity.
    """)

with feature_tabs[3]:
    st.write("""
    **Area Features**
    - Abandonment rate (harvested/planted)
    - Harvest efficiency metrics
    
    These reflect agricultural practices and field conditions.
    """)

st.markdown("---")

st.subheader("Model Training")

st.write("""
**Data Split:**
- Training: 70% of data (random split)
- Validation: 15% of data
- Test: 15% of data

**Cross-Validation:**
- 5-fold cross-validation for hyperparameter tuning
- RandomizedSearchCV for efficient parameter space exploration

**Feature Scaling:**
- StandardScaler (mean=0, std=1) applied to all features
- Essential for linear models and gradient boosting

**Hyperparameter Tuning:**
- RandomizedSearchCV with 3-fold CV
- 50-100 iterations per model
- Parallel processing for efficiency
""")

st.markdown("---")

# Models
st.header("Models Evaluated")

for model_name, metadata in {
    'xgboost': 'XGBoost Regressor',
    'random_forest': 'Random Forest Regressor',
    'gradient_boosting': 'Gradient Boosting Regressor',
    'ridge': 'Ridge Regression'
}.items():
    from app.config import MODEL_METADATA
    info = MODEL_METADATA[model_name]
    
    with st.expander(f"{info['name']} - R² = {info['r2']:.3f}"):
        st.write(info['description'])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²", f"{info['r2']:.3f}")
        with col2:
            st.metric("MAE", f"{info['mae']:.2f} BU/ACRE")
        with col3:
            st.metric("RMSE", f"{info['rmse']:.2f} BU/ACRE")

st.markdown("---")

# Results Summary
st.header("Results Summary")

st.write("""
**Best Model Performance:**
- **XGBoost Regressor** achieves R² = 0.863, MAE = 11.22 BU/ACRE
- Significant improvement over baseline methods
- Strong performance across diverse geographic and temporal conditions

**Key Findings:**
1. Historical yield patterns are the strongest predictors (53% of feature importance)
2. Weather features, especially heat stress, significantly impact yields
3. Ensemble methods (boosting, random forest) outperform linear models
4. Model performs best on high-yield regions and stable weather years
5. Extreme weather events (e.g., 2012 drought) pose challenges for all models

**Limitations:**
- Requires historical yield data (cannot predict new counties without history)
- Performance degrades during extreme weather events
- County-level aggregation may mask field-level variability
- Static soil data doesn't capture temporal changes
""")

st.markdown("---")





st.markdown("---")

# Contact & Credits
st.header("Credits")

st.write("""
**Author:** Ahsan Riaz  
**Course:** CS 245 - Machine Learning  
**Institution:** National Univeristy of Science and Technology, Islamabad 
**Date:** November 2025

**Acknowledgments:**
- USDA NASS for yield and production data
- NASA POWER for weather data
- USDA NRCS for soil data
""")

