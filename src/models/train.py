"""
Complete ML Training Pipeline for Corn Yield Prediction
========================================================
Trains and evaluates 5 different models:
1. Baseline: Historical Average
2. Linear Regression (Ridge with regularization)
3. Random Forest Regressor
4. XGBoost Regressor
5. Gradient Boosting Regressor

Includes:
- Train/Validation/Test splits (70/15/15)
- Cross-validation (5-fold)
- Hyperparameter tuning
- Feature importance analysis
- Error analysis
- Model comparison
- Model persistence

Author: Corn Yield Prediction Project
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

print("="*70)
print("CORN YIELD PREDICTION - ML TRAINING PIPELINE")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CV_FOLDS = 5

# Define project root (3 levels up from this file: src/models/train.py -> src/models -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
TABLES_DIR = RESULTS_DIR / 'tables'

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")

data_path = DATA_DIR / 'modeling_dataset_final.csv'
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found at {data_path}. Please run data preparation script first.")

df = pd.read_csv(data_path)
print(f"  ✓ Loaded {len(df):,} records")
print(f"  ✓ Features: {len(df.columns)}")

# ============================================================================
# 2. FEATURE ENGINEERING & SELECTION
# ============================================================================
print("\n[2/8] Feature engineering...")

# Create lag features for historical yield
df = df.sort_values(['State', 'County', 'Year'])
df['Yield_Lag1'] = df.groupby(['State', 'County'])['Yield_BU_ACRE'].shift(1)
df['Yield_Lag2'] = df.groupby(['State', 'County'])['Yield_BU_ACRE'].shift(2)
df['Yield_3yr_Avg'] = df.groupby(['State', 'County'])['Yield_BU_ACRE'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
)

# Drop rows with NaN in lag features (first years)
df = df.dropna(subset=['Yield_Lag1'])

print(f"  ✓ Created lag features")
print(f"  ✓ Records after feature engineering: {len(df):,}")

# Define feature columns
exclude_cols = [
    'Yield_BU_ACRE',  # Target
    'State', 'County',  # Identifiers (will encode)
    'Year',  # Already captured in features
    'State ANSI', 'County ANSI', 'Ag District',  # Redundant IDs
    'Area_Planted_ACRES', 'Area_Harvested_ACRES', 'Production_BU',  # Don't use for prediction
]

# Encode State as categorical
state_encoder = {state: idx for idx, state in enumerate(df['State'].unique())}
df['State_Encoded'] = df['State'].map(state_encoder)

# Select features
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"  ✓ Selected {len(feature_cols)} features")

# Separate features and target
X = df[feature_cols]
y = df['Yield_BU_ACRE']

print(f"\n  Feature Groups:")
print(f"    Historical: Yield_Lag1, Yield_Lag2, Yield_3yr_Avg")
print(f"    Soil: Soil_AWC, Soil_Clay_Pct, Soil_pH, Soil_Organic_Matter_Pct")
print(f"    Weather: gdd_total, precip_total, temp_mean_season, etc. (34 features)")
print(f"    Other: Abandonment_Rate, Harvest_Efficiency, State_Encoded")

# ============================================================================
# 3. TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n[3/8] Creating train/validation/test splits...")

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Second split: separate validation set from training
val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_STATE
)

print(f"  ✓ Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  ✓ Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"  ✓ Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================
print("\n[4/8] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"  ✓ Features scaled (StandardScaler)")

# ============================================================================
# 5. MODEL TRAINING & EVALUATION
# ============================================================================
print("\n[5/8] Training models...")
print("="*70)

models = {}
predictions = {}
metrics = {}

# ----------------------------------------------------------------------------
# MODEL 1: BASELINE - Historical Average
# ----------------------------------------------------------------------------
print("\n[Model 1/5] Baseline: Historical Average")
print("-" * 70)

# Use 3-year average as baseline prediction
baseline_pred_train = X_train['Yield_3yr_Avg']
baseline_pred_val = X_val['Yield_3yr_Avg']
baseline_pred_test = X_test['Yield_3yr_Avg']

# Metrics
baseline_metrics = {
    'train_mae': mean_absolute_error(y_train, baseline_pred_train),
    'train_rmse': np.sqrt(mean_squared_error(y_train, baseline_pred_train)),
    'train_r2': r2_score(y_train, baseline_pred_train),
    'val_mae': mean_absolute_error(y_val, baseline_pred_val),
    'val_rmse': np.sqrt(mean_squared_error(y_val, baseline_pred_val),),
    'val_r2': r2_score(y_val, baseline_pred_val),
    'test_mae': mean_absolute_error(y_test, baseline_pred_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, baseline_pred_test)),
    'test_r2': r2_score(y_test, baseline_pred_test)
}

print(f"  Training   - MAE: {baseline_metrics['train_mae']:.2f}, RMSE: {baseline_metrics['train_rmse']:.2f}, R²: {baseline_metrics['train_r2']:.3f}")
print(f"  Validation - MAE: {baseline_metrics['val_mae']:.2f}, RMSE: {baseline_metrics['val_rmse']:.2f}, R²: {baseline_metrics['val_r2']:.3f}")
print(f"  Test       - MAE: {baseline_metrics['test_mae']:.2f}, RMSE: {baseline_metrics['test_rmse']:.2f}, R²: {baseline_metrics['test_r2']:.3f}")

metrics['Baseline'] = baseline_metrics
predictions['Baseline'] = {'train': baseline_pred_train, 'val': baseline_pred_val, 'test': baseline_pred_test}

# ----------------------------------------------------------------------------
# MODEL 2: RIDGE REGRESSION
# ----------------------------------------------------------------------------
print("\n[Model 2/5] Ridge Regression with Hyperparameter Tuning")
print("-" * 70)

# Hyperparameter tuning
param_grid_ridge = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

ridge_model = Ridge(random_state=RANDOM_STATE)
grid_search_ridge = GridSearchCV(
    ridge_model, param_grid_ridge, cv=CV_FOLDS, 
    scoring='r2', n_jobs=-1, verbose=0
)
grid_search_ridge.fit(X_train_scaled, y_train)

print(f"  ✓ Best parameters: {grid_search_ridge.best_params_}")
print(f"  ✓ Best CV R²: {grid_search_ridge.best_score_:.3f}")

# Train final model
ridge_best = grid_search_ridge.best_estimator_
ridge_pred_train = ridge_best.predict(X_train_scaled)
ridge_pred_val = ridge_best.predict(X_val_scaled)
ridge_pred_test = ridge_best.predict(X_test_scaled)

ridge_metrics = {
    'train_mae': mean_absolute_error(y_train, ridge_pred_train),
    'train_rmse': np.sqrt(mean_squared_error(y_train, ridge_pred_train)),
    'train_r2': r2_score(y_train, ridge_pred_train),
    'val_mae': mean_absolute_error(y_val, ridge_pred_val),
    'val_rmse': np.sqrt(mean_squared_error(y_val, ridge_pred_val)),
    'val_r2': r2_score(y_val, ridge_pred_val),
    'test_mae': mean_absolute_error(y_test, ridge_pred_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, ridge_pred_test)),
    'test_r2': r2_score(y_test, ridge_pred_test)
}

print(f"  Training   - MAE: {ridge_metrics['train_mae']:.2f}, RMSE: {ridge_metrics['train_rmse']:.2f}, R²: {ridge_metrics['train_r2']:.3f}")
print(f"  Validation - MAE: {ridge_metrics['val_mae']:.2f}, RMSE: {ridge_metrics['val_rmse']:.2f}, R²: {ridge_metrics['val_r2']:.3f}")
print(f"  Test       - MAE: {ridge_metrics['test_mae']:.2f}, RMSE: {ridge_metrics['test_rmse']:.2f}, R²: {ridge_metrics['test_r2']:.3f}")

models['Ridge'] = ridge_best
metrics['Ridge'] = ridge_metrics
predictions['Ridge'] = {'train': ridge_pred_train, 'val': ridge_pred_val, 'test': ridge_pred_test}

# ----------------------------------------------------------------------------
# MODEL 3: RANDOM FOREST
# ----------------------------------------------------------------------------
print("\n[Model 3/5] Random Forest with Hyperparameter Tuning")
print("-" * 70)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

rf_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
grid_search_rf = GridSearchCV(
    rf_model, param_grid_rf, cv=CV_FOLDS,
    scoring='r2', n_jobs=-1, verbose=1
)

print("  Training Random Forest (this may take 5-10 minutes)...")
grid_search_rf.fit(X_train, y_train)

print(f"  ✓ Best parameters: {grid_search_rf.best_params_}")
print(f"  ✓ Best CV R²: {grid_search_rf.best_score_:.3f}")

rf_best = grid_search_rf.best_estimator_
rf_pred_train = rf_best.predict(X_train)
rf_pred_val = rf_best.predict(X_val)
rf_pred_test = rf_best.predict(X_test)

rf_metrics = {
    'train_mae': mean_absolute_error(y_train, rf_pred_train),
    'train_rmse': np.sqrt(mean_squared_error(y_train, rf_pred_train)),
    'train_r2': r2_score(y_train, rf_pred_train),
    'val_mae': mean_absolute_error(y_val, rf_pred_val),
    'val_rmse': np.sqrt(mean_squared_error(y_val, rf_pred_val)),
    'val_r2': r2_score(y_val, rf_pred_val),
    'test_mae': mean_absolute_error(y_test, rf_pred_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, rf_pred_test)),
    'test_r2': r2_score(y_test, rf_pred_test)
}

print(f"  Training   - MAE: {rf_metrics['train_mae']:.2f}, RMSE: {rf_metrics['train_rmse']:.2f}, R²: {rf_metrics['train_r2']:.3f}")
print(f"  Validation - MAE: {rf_metrics['val_mae']:.2f}, RMSE: {rf_metrics['val_rmse']:.2f}, R²: {rf_metrics['val_r2']:.3f}")
print(f"  Test       - MAE: {rf_metrics['test_mae']:.2f}, RMSE: {rf_metrics['test_rmse']:.2f}, R²: {rf_metrics['test_r2']:.3f}")

models['RandomForest'] = rf_best
metrics['RandomForest'] = rf_metrics
predictions['RandomForest'] = {'train': rf_pred_train, 'val': rf_pred_val, 'test': rf_pred_test}

# ----------------------------------------------------------------------------
# MODEL 4: XGBOOST
# ----------------------------------------------------------------------------
print("\n[Model 4/5] XGBoost with Hyperparameter Tuning")
print("-" * 70)

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
grid_search_xgb = GridSearchCV(
    xgb_model, param_grid_xgb, cv=CV_FOLDS,
    scoring='r2', n_jobs=-1, verbose=1
)

print("  Training XGBoost (this may take 10-15 minutes)...")
grid_search_xgb.fit(X_train, y_train)

print(f"  ✓ Best parameters: {grid_search_xgb.best_params_}")
print(f"  ✓ Best CV R²: {grid_search_xgb.best_score_:.3f}")

xgb_best = grid_search_xgb.best_estimator_
xgb_pred_train = xgb_best.predict(X_train)
xgb_pred_val = xgb_best.predict(X_val)
xgb_pred_test = xgb_best.predict(X_test)

xgb_metrics = {
    'train_mae': mean_absolute_error(y_train, xgb_pred_train),
    'train_rmse': np.sqrt(mean_squared_error(y_train, xgb_pred_train)),
    'train_r2': r2_score(y_train, xgb_pred_train),
    'val_mae': mean_absolute_error(y_val, xgb_pred_val),
    'val_rmse': np.sqrt(mean_squared_error(y_val, xgb_pred_val)),
    'val_r2': r2_score(y_val, xgb_pred_val),
    'test_mae': mean_absolute_error(y_test, xgb_pred_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, xgb_pred_test)),
    'test_r2': r2_score(y_test, xgb_pred_test)
}

print(f"  Training   - MAE: {xgb_metrics['train_mae']:.2f}, RMSE: {xgb_metrics['train_rmse']:.2f}, R²: {xgb_metrics['train_r2']:.3f}")
print(f"  Validation - MAE: {xgb_metrics['val_mae']:.2f}, RMSE: {xgb_metrics['val_rmse']:.2f}, R²: {xgb_metrics['val_r2']:.3f}")
print(f"  Test       - MAE: {xgb_metrics['test_mae']:.2f}, RMSE: {xgb_metrics['test_rmse']:.2f}, R²: {xgb_metrics['test_r2']:.3f}")

models['XGBoost'] = xgb_best
metrics['XGBoost'] = xgb_metrics
predictions['XGBoost'] = {'train': xgb_pred_train, 'val': xgb_pred_val, 'test': xgb_pred_test}

# ----------------------------------------------------------------------------
# MODEL 5: GRADIENT BOOSTING
# ----------------------------------------------------------------------------
print("\n[Model 5/5] Gradient Boosting with Hyperparameter Tuning")
print("-" * 70)

param_grid_gb = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

gb_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
grid_search_gb = GridSearchCV(
    gb_model, param_grid_gb, cv=CV_FOLDS,
    scoring='r2', n_jobs=-1, verbose=1
)

print("  Training Gradient Boosting (this may take 10-15 minutes)...")
grid_search_gb.fit(X_train, y_train)

print(f"  ✓ Best parameters: {grid_search_gb.best_params_}")
print(f"  ✓ Best CV R²: {grid_search_gb.best_score_:.3f}")

gb_best = grid_search_gb.best_estimator_
gb_pred_train = gb_best.predict(X_train)
gb_pred_val = gb_best.predict(X_val)
gb_pred_test = gb_best.predict(X_test)

gb_metrics = {
    'train_mae': mean_absolute_error(y_train, gb_pred_train),
    'train_rmse': np.sqrt(mean_squared_error(y_train, gb_pred_train)),
    'train_r2': r2_score(y_train, gb_pred_train),
    'val_mae': mean_absolute_error(y_val, gb_pred_val),
    'val_rmse': np.sqrt(mean_squared_error(y_val, gb_pred_val)),
    'val_r2': r2_score(y_val, gb_pred_val),
    'test_mae': mean_absolute_error(y_test, gb_pred_test),
    'test_rmse': np.sqrt(mean_squared_error(y_test, gb_pred_test)),
    'test_r2': r2_score(y_test, gb_pred_test)
}

print(f"  Training   - MAE: {gb_metrics['train_mae']:.2f}, RMSE: {gb_metrics['train_rmse']:.2f}, R²: {gb_metrics['train_r2']:.3f}")
print(f"  Validation - MAE: {gb_metrics['val_mae']:.2f}, RMSE: {gb_metrics['val_rmse']:.2f}, R²: {gb_metrics['val_r2']:.3f}")
print(f"  Test       - MAE: {gb_metrics['test_mae']:.2f}, RMSE: {gb_metrics['test_rmse']:.2f}, R²: {gb_metrics['test_r2']:.3f}")

models['GradientBoosting'] = gb_best
metrics['GradientBoosting'] = gb_metrics
predictions['GradientBoosting'] = {'train': gb_pred_train, 'val': gb_pred_val, 'test': gb_pred_test}

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================
print("\n" + "="*70)
print("[6/8] MODEL COMPARISON")
print("="*70)

# Create comparison DataFrame
comparison_data = []
for model_name, model_metrics in metrics.items():
    comparison_data.append({
        'Model': model_name,
        'Train_R2': model_metrics['train_r2'],
        'Train_MAE': model_metrics['train_mae'],
        'Train_RMSE': model_metrics['train_rmse'],
        'Val_R2': model_metrics['val_r2'],
        'Val_MAE': model_metrics['val_mae'],
        'Val_RMSE': model_metrics['val_rmse'],
        'Test_R2': model_metrics['test_r2'],
        'Test_MAE': model_metrics['test_mae'],
        'Test_RMSE': model_metrics['test_rmse']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test_R2', ascending=False)

print("\nTest Set Performance (sorted by R²):")
print(comparison_df[['Model', 'Test_R2', 'Test_MAE', 'Test_RMSE']].to_string(index=False))

print("\n" + "="*70)
print("BEST MODEL (based on Test R²):")
best_model_name = comparison_df.iloc[0]['Model']
best_model_r2 = comparison_df.iloc[0]['Test_R2']
best_model_mae = comparison_df.iloc[0]['Test_MAE']
print(f"  {best_model_name}")
print(f"  R² = {best_model_r2:.3f}")
print(f"  MAE = {best_model_mae:.2f} BU/ACRE")
print("="*70)

# Save comparison table
comparison_path = TABLES_DIR / 'model_comparison.csv'
comparison_df.to_csv(comparison_path, index=False)
print(f"\n✓ Saved: {comparison_path}")

# ============================================================================
# 7. FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
print("\n[7/8] Feature Importance Analysis...")

# XGBoost Feature Importance
if 'XGBoost' in models:
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': models['XGBoost'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 Most Important Features (XGBoost):")
    print(feature_importance.head(15).to_string(index=False))
    
    importance_path = TABLES_DIR / 'feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"\n✓ Saved: {importance_path}")

# ============================================================================
# 8. SAVE MODELS
# ============================================================================
print("\n[8/8] Saving models...")

# Save all models
for model_name, model in models.items():
    filename = MODELS_DIR / f'{model_name.lower()}_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved: {filename}")

# Save scaler
scaler_path = MODELS_DIR / 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ Saved: {scaler_path}")

# Save feature columns
cols_path = MODELS_DIR / 'feature_columns.pkl'
with open(cols_path, 'wb') as f:
    pickle.dump(list(X_train.columns), f)
print(f"  ✓ Saved: {cols_path}")

# Save state encoder
state_encoder_path = MODELS_DIR / 'state_encoder.pkl'
with open(state_encoder_path, 'wb') as f:
    pickle.dump(state_encoder, f)
print(f"  ✓ Saved: {state_encoder_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\n✓ Trained 5 models")
print(f"✓ Best model: {best_model_name} (R² = {best_model_r2:.3f})")
print(f"✓ Improvement over baseline: {(best_model_r2 - baseline_metrics['test_r2']):.3f}")
print(f"✓ All models saved to {MODELS_DIR}")
print(f"✓ Comparison table: {comparison_path}")
print(f"✓ Feature importance: {importance_path}")
print("\n" + "="*70 + "\n")

