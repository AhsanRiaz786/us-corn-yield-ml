# Models Directory

This directory contains trained machine learning models and associated artifacts.

## Contents

```
models/
├── xgboost_model.pkl              # XGBoost model (best performer)
├── gradientboosting_model.pkl     # Scikit-learn Gradient Boosting
├── randomforest_model.pkl         # Random Forest Regressor
├── ridge_model.pkl                # Ridge Regression (baseline)
├── scaler.pkl                     # StandardScaler for feature normalization
└── feature_columns.pkl            # Ordered list of feature names
```

## Model Files

### 1. xgboost_model.pkl

**Algorithm:** XGBoost Regressor  
**Performance:** R² = 0.863, MAE = 11.22 BU/ACRE  
**Size:** ~15 MB  
**Training Time:** ~45 minutes

Best performing model. Recommended for production deployment.

**Hyperparameters:**
- n_estimators: 400
- max_depth: 10
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.6

### 2. gradientboosting_model.pkl

**Algorithm:** Scikit-learn Gradient Boosting  
**Performance:** R² = 0.859, MAE = 11.38 BU/ACRE  
**Size:** ~8 MB  
**Training Time:** ~38 minutes

Close second to XGBoost. Good alternative with standard scikit-learn API.

### 3. randomforest_model.pkl

**Algorithm:** Random Forest Regressor  
**Performance:** R² = 0.844, MAE = 12.14 BU/ACRE  
**Size:** ~25 MB  
**Training Time:** ~28 minutes

Solid performance with faster training. Good for rapid iteration.

### 4. ridge_model.pkl

**Algorithm:** Ridge Regression  
**Performance:** R² = 0.713, MAE = 16.94 BU/ACRE  
**Size:** <1 MB  
**Training Time:** ~2 minutes

Fast and interpretable baseline. Useful when computational resources limited.

## Auxiliary Files

### scaler.pkl

**Type:** StandardScaler (scikit-learn)  
**Purpose:** Feature normalization (mean=0, std=1)  
**Must be used:** Before prediction with any trained model

**Usage:**
```python
import joblib
scaler = joblib.load('models/scaler.pkl')
X_scaled = scaler.transform(X)
```

### feature_columns.pkl

**Type:** Python list of strings  
**Purpose:** Defines expected feature order for model input  
**Length:** 50+ features

**Usage:**
```python
import joblib
features = joblib.load('models/feature_columns.pkl')
X_ordered = X[features]  # Ensure correct column order
```

## Loading Models

### Quick Start

```python
import joblib
import pandas as pd

# Load model and artifacts
model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')
features = joblib.load('models/feature_columns.pkl')

# Prepare input data
X = pd.DataFrame({...})  # Your feature data
X_ordered = X[features]
X_scaled = scaler.transform(X_ordered)

# Make prediction
predictions = model.predict(X_scaled)
```

### Using Prediction Utility

```python
from src.models.predict import YieldPredictor

# Initialize predictor
predictor = YieldPredictor(model_name='xgboost')

# Make predictions
predictions = predictor.predict(X)
```

## Model Training

To retrain all models from scratch:

```bash
python scripts/03_train_models.py
```

This will:
1. Load processed data
2. Create train/validation/test splits
3. Train all 5 models with hyperparameter tuning
4. Save trained models to this directory
5. Generate performance reports

**Note:** Training requires the final modeling dataset (`data/processed/modeling_dataset_final.csv`)

## Model Validation

### Train/Validation/Test Split

- **Training:** 1981-2015 (70% of data)
- **Validation:** 2016-2019 (15% of data)
- **Test:** 2020-2023 (15% of data)

Time-based splitting ensures no data leakage and reflects real-world deployment.

### Cross-Validation

- **Method:** 3-fold cross-validation during hyperparameter tuning
- **Metric:** Negative Mean Absolute Error
- **Purpose:** Prevent overfitting to validation set

## Model Comparison

| Model | R² | MAE | RMSE | Size | Speed |
|-------|-----|-----|------|------|-------|
| XGBoost | 0.863 | 11.22 | 15.59 | 15 MB | Medium |
| Gradient Boosting | 0.859 | 11.38 | 15.82 | 8 MB | Medium |
| Random Forest | 0.844 | 12.14 | 16.62 | 25 MB | Fast |
| Ridge | 0.713 | 16.94 | 22.51 | <1 MB | Very Fast |

## Feature Importance

Top 10 features (XGBoost model):

1. Yield_lag1 (32.1%) - Previous year yield
2. Yield_lag2 (10.8%) - Two years prior
3. Yield_lag3 (10.2%) - Three years prior
4. Heat_stress_days (4.7%) - Days >35°C
5. Tmax_max (3.8%) - Peak temperature
6. GDD_total (3.2%) - Growing degree days
7. Precip_anomaly (2.9%) - Rainfall deviation
8. Area_planted (2.1%) - Acres planted
9. Abandonment_rate (1.8%) - Harvested/planted
10. AWC_avg (1.6%) - Soil water capacity

See `results/tables/feature_importance.csv` for complete rankings.

## Deployment Considerations

### Memory Requirements

- **XGBoost:** ~100 MB RAM (loaded)
- **Random Forest:** ~150 MB RAM (loaded)
- **Ridge:** ~10 MB RAM (loaded)

### Prediction Speed

- **Ridge:** ~0.1 ms per prediction (10,000/sec)
- **Random Forest:** ~1 ms per prediction (1,000/sec)
- **XGBoost:** ~2 ms per prediction (500/sec)

All models fast enough for real-time API deployment.

### Model Updates

Models should be retrained annually when new harvest data becomes available. This ensures:
- Capture of recent yield trends
- Adaptation to changing climate patterns
- Integration of latest agricultural practices

### Versioning

Current models trained: November 2025

When retraining, save models with version suffix:
```
xgboost_model_v2.pkl
xgboost_model_2025.pkl
```

## Error Analysis

Detailed error analysis available in:
- `results/error_analysis_summary.txt`
- `results/tables/error_by_year.csv`
- `results/tables/error_by_state.csv`
- `results/figures/error_distribution.png`

Key findings:
- Best performance in Corn Belt states (IA, IL, IN)
- Higher errors during extreme weather events (2012 drought)
- Relative error decreases with yield level (4-10%)

## Model Limitations

1. **Extrapolation:** Poor performance beyond training data range
2. **Extreme events:** Limited training examples of severe droughts/floods
3. **Spatial bias:** Optimized for high-production regions
4. **Temporal assumptions:** Assumes stationary climate (no climate change trends)

See `docs/model_performance.md` for comprehensive limitations discussion.

## File Formats

All model files are Python pickle format (`.pkl`):
- Created with: `joblib.dump()`
- Loaded with: `joblib.load()`
- Python version: 3.11+
- Compatible with: scikit-learn 1.3+, xgboost 2.0+

## Backup

Model files should be backed up before retraining. Training is computationally expensive (~2 hours total).

## Questions?

For model implementation details, see:
- `src/models/train.py` - Training pipeline
- `src/models/predict.py` - Prediction utilities
- `docs/model_performance.md` - Comprehensive performance analysis

