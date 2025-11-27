"""
Configuration constants for the Streamlit dashboard.
"""
import sys
from pathlib import Path

# Add project root to path if not already there
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Project root directory (parent of app/)
PROJECT_ROOT = project_root

# Paths
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'

# Model files
MODEL_FILES = {
    'xgboost': MODELS_DIR / 'xgboost_model.pkl',
    'random_forest': MODELS_DIR / 'randomforest_model.pkl',
    'gradient_boosting': MODELS_DIR / 'gradientboosting_model.pkl',
    'ridge': MODELS_DIR / 'ridge_model.pkl',
}

SCALER_FILE = MODELS_DIR / 'scaler.pkl'
FEATURE_COLUMNS_FILE = MODELS_DIR / 'feature_columns.pkl'

# Data files
DATA_FILE = DATA_DIR / 'modeling_dataset_final.csv'

# Model metadata
MODEL_METADATA = {
    'xgboost': {
        'name': 'XGBoost Regressor',
        'r2': 0.863,
        'mae': 11.22,
        'rmse': 15.59,
        'description': 'Gradient boosting with regularization. Best performing model.'
    },
    'random_forest': {
        'name': 'Random Forest Regressor',
        'r2': 0.844,
        'mae': 12.14,
        'rmse': 16.62,
        'description': 'Ensemble of decision trees. Fast training, good performance.'
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting Regressor',
        'r2': 0.859,
        'mae': 11.38,
        'rmse': 15.82,
        'description': 'Sequential ensemble learning. Strong performance.'
    },
    'ridge': {
        'name': 'Ridge Regression',
        'r2': 0.713,
        'mae': 16.94,
        'rmse': 22.51,
        'description': 'Linear model with L2 regularization. Interpretable baseline.'
    }
}

# Feature categories for grouping
FEATURE_CATEGORIES = {
    'Historical': ['Yield_Lag1', 'Yield_Lag2', 'Yield_3yr_Avg'],
    'Soil': ['Soil_AWC', 'Soil_Clay_Pct', 'Soil_pH', 'Soil_Organic_Matter_Pct'],
    'Weather - Temperature': [
        'gdd_total', 'gdd_vegetative', 'gdd_reproductive', 'gdd_grainfill',
        'temp_mean_season', 'temp_max_season', 'temp_min_season',
        'temp_mean_reproductive', 'temp_max_reproductive'
    ],
    'Weather - Heat Stress': [
        'weeks_heat_stress', 'weeks_extreme_heat', 'heat_moisture_stress',
        'temp_std_season', 'temp_range_avg'
    ],
    'Weather - Precipitation': [
        'precip_total', 'precip_vegetative', 'precip_reproductive',
        'precip_grainfill', 'precip_mean_weekly', 'precip_max_weekly',
        'precip_std', 'weeks_dry', 'weeks_very_dry', 'weeks_wet',
        'water_stress_reproductive'
    ],
    'Weather - Anomalies': [
        'gdd_anomaly', 'precip_anomaly_mm', 'precip_anomaly_pct', 'temp_anomaly'
    ],
    'Weather - Other': [
        'rh_mean', 'rh_reproductive', 'weeks_high_humidity',
        'temp_early_vs_late', 'precip_early_vs_late'
    ],
    'Area': ['Abandonment_Rate', 'Harvest_Efficiency'],
    'Geographic': ['State_Encoded']
}

# Default values for missing data
DEFAULT_VALUES = {
    'Soil_AWC': 15.0,
    'Soil_Clay_Pct': 25.0,
    'Soil_pH': 6.5,
    'Soil_Organic_Matter_Pct': 3.0,
    'Abandonment_Rate': 0.05,
    'Harvest_Efficiency': 1.0,
}

# UI Configuration
PAGE_CONFIG = {
    'page_title': 'US Corn Yield Prediction',
    'page_icon': '🌽',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#2ca02c',
    'accent': '#ff7f0e',
    'background': '#f8f9fa',
    'text': '#212529',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
}

