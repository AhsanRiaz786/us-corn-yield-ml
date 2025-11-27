# US Corn Yield Prediction System

A machine learning system for predicting county-level corn yields across the United States using multi-source data integration and ensemble modeling techniques.

## Overview

This project implements a comprehensive ML pipeline that integrates heterogeneous datasets from multiple sources to predict corn yields at the county level. The system achieves state-of-the-art performance (R² = 0.863) by combining historical agricultural statistics, weather data, and soil properties.

## Performance Summary

| Model | Test R² | MAE (BU/ACRE) | RMSE (BU/ACRE) |
|-------|---------|---------------|----------------|
| XGBoost | 0.863 | 11.22 | 15.59 |
| Gradient Boosting | 0.859 | 11.38 | 15.82 |
| Random Forest | 0.844 | 12.14 | 16.62 |
| Ridge Regression | 0.713 | 16.94 | 22.51 |
| Baseline (3-year avg) | 0.628 | 19.44 | 25.64 |

The best model (XGBoost) achieves a mean absolute error of 11.22 BU/ACRE, representing approximately 6.4% relative error on average yields.

## Data Sources

The system integrates data from four primary sources:

1. **USDA NASS QuickStats** - County-level corn statistics (1981-2023)
   - Corn yield (BU/ACRE)
   - Area planted (ACRES)
   - Area harvested (ACRES)
   - Total production (BU)

2. **NASA POWER Agroclimatology** - Daily weather data aggregated to weekly intervals
   - Temperature (min, max, mean)
   - Precipitation
   - Relative humidity

3. **USDA NRCS Soil Data Access** - Soil properties by county
   - Available water capacity (AWC)
   - Clay content percentage
   - Soil pH
   - Organic matter percentage

4. **US Census TIGER/Line** - County geographic centroids for spatial queries

Total dataset: 82,436 county-year observations spanning 1981-2023 across 2,635 counties.

## Repository Structure

```
us-corn-yield/
├── data/
│   ├── raw/              # Original downloaded data
│   └── processed/        # Cleaned and merged datasets
├── src/
│   ├── data_collection/  # Data download scripts
│   ├── preprocessing/    # Data cleaning and merging
│   ├── features/         # Feature engineering
│   ├── models/          # Model training and evaluation
│   └── visualization/    # Plotting utilities
├── models/              # Trained model artifacts (.pkl)
├── results/
│   ├── figures/         # Generated plots
│   └── tables/          # Performance metrics (CSV)
├── notebooks/           # Analysis notebooks
├── app/                 # Streamlit dashboard
├── scripts/             # Pipeline automation
└── docs/                # Additional documentation
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or conda package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/AhsanRiaz786/us-corn-yield-ml
cd us-corn-yield

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start: Interactive Dashboard

Launch the Streamlit dashboard to explore predictions interactively:

```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:
- County-level yield predictions
- Interactive maps
- Model performance visualization
- What-if scenario simulation

### Running the Full Pipeline

Execute the complete data collection, preprocessing, and training pipeline:

```bash
# Run all steps sequentially
bash scripts/run_full_pipeline.sh

# Or run individual steps
python scripts/01_download_all_data.py
python scripts/02_prepare_data.py
python scripts/03_train_models.py
python scripts/04_evaluate_models.py
```

### Using Trained Models

```python
import joblib
import pandas as pd

# Load model and preprocessing artifacts
model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

# Prepare input data
input_data = pd.DataFrame({...})  # Your county-year features
input_scaled = scaler.transform(input_data[feature_columns])

# Generate prediction
predicted_yield = model.predict(input_scaled)
```

## Methodology

### Feature Engineering

The system creates 50+ features including:

**Historical Features**
- Lagged yields (1-3 years)
- Rolling averages
- Year-over-year changes

**Weather Features** (Growing season: April-September)
- Growing Degree Days (GDD)
- Heat stress indicators
- Precipitation totals and anomalies
- Temperature extremes

**Soil Features**
- Available water capacity
- Soil pH
- Clay content
- Organic matter

**Derived Features**
- Abandonment rate (area harvested / area planted)
- Production per acre
- Temporal trends

### Model Training

**Data Split Strategy**
- Training: 1981-2015 (70%)
- Validation: 2016-2019 (15%)
- Test: 2020-2023 (15%)

Time-based splitting prevents data leakage and reflects real-world deployment.

**Hyperparameter Optimization**
- RandomizedSearchCV for efficient parameter space exploration
- 3-fold cross-validation
- Parallel processing for computational efficiency

**Models Evaluated**
1. Baseline: 3-year moving average
2. Ridge Regression: L2 regularization
3. Random Forest: Ensemble of decision trees
4. XGBoost: Gradient boosting with regularization
5. Gradient Boosting: Sequential ensemble learning

## Key Findings

### Feature Importance (XGBoost)

Top 10 most influential features:
1. Yield lag 1 year (32.1%)
2. Yield lag 2 years (10.8%)
3. Yield lag 3 years (10.2%)
4. Heat stress days (4.7%)
5. Maximum temperature (3.8%)
6. GDD total (3.2%)
7. Precipitation anomaly (2.9%)
8. Area planted (2.1%)
9. Abandonment rate (1.8%)
10. Soil AWC (1.6%)

Historical yield patterns account for 53% of total feature importance, emphasizing the persistence of agricultural productivity.

### Error Analysis

**Temporal Patterns**
- Best performance: 2020-2023 (MAE = 10.88)
- Worst performance: 2012 (MAE = 16.63) - severe drought year
- Model struggles during extreme weather events

**Spatial Patterns**
- Corn Belt states (IA, IL, NE): MAE = 8-10 BU/ACRE
- Peripheral states: MAE = 15-20 BU/ACRE
- Higher errors in low-yield, high-variability regions

**Yield Level Patterns**
- High yield counties (>180 BU/ACRE): Best predictions
- Low yield counties (<120 BU/ACRE): Higher relative errors
- Model optimized for typical production conditions

## Limitations and Future Work

**Current Limitations**
- Temporal resolution: Annual predictions only
- Spatial resolution: County-level aggregation
- Weather data: Centroid-based, not area-weighted
- Soil data: Static properties, no seasonal dynamics

**Future Enhancements**
- Incorporate satellite imagery (NDVI, LAI)
- Add economic features (fertilizer prices, crop insurance)
- Implement deep learning architectures (LSTMs, CNNs)
- Develop sub-county prediction capabilities
- Add real-time prediction API
- Extend to other crops (soybeans, wheat)

## Documentation

- `docs/data_sources.md` - Detailed data source documentation
- `docs/feature_engineering.md` - Feature creation methodology
- `docs/model_performance.md` - Comprehensive results analysis

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{riaz2025cornyield,
  author = {Riaz, Ahsan},
  title = {US Corn Yield Prediction System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/AhsanRiaz786/us-corn-yield-ml}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Data sources:
- USDA National Agricultural Statistics Service (NASS)
- NASA POWER Project
- USDA Natural Resources Conservation Service (NRCS)
- US Census Bureau TIGER/Line Shapefiles

## Contact

Ahsan Riaz - CS 245 Machine Learning, Fall 2025

For questions or collaborations, please open an issue on GitHub.

