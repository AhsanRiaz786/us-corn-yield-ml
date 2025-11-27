# Quick Setup Guide

This guide provides step-by-step instructions for setting up and running the US Corn Yield Prediction System.

## Prerequisites

- Python 3.11 or higher
- 600 MB free disk space for data
- Internet connection for data download

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/us-corn-yield.git
cd us-corn-yield
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage Options

### Option A: Use Pre-Trained Models (Quick Start)

If models and data are already available:

```bash
# Launch dashboard immediately
streamlit run app/streamlit_app.py
```

### Option B: Complete Pipeline from Scratch

#### Step 1: Download Data (2-4 hours)

```bash
python scripts/01_download_all_data.py
```

Downloads:
- USDA NASS corn statistics
- NASA POWER weather data
- USDA NRCS soil properties
- County geographic coordinates

#### Step 2: Prepare Data (10 minutes)

```bash
python scripts/02_prepare_data.py
```

Performs:
- Dataset merging
- Feature engineering
- Data cleaning

#### Step 3: Train Models (30-60 minutes)

```bash
python scripts/03_train_models.py
```

Trains:
- Baseline model
- Ridge Regression
- Random Forest
- XGBoost
- Gradient Boosting

#### Step 4: Evaluate Models (5 minutes)

```bash
python scripts/04_evaluate_models.py
```

Generates:
- Error analysis
- Performance visualizations
- Summary statistics

#### Step 5: Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Option C: Automated Pipeline

Run everything at once:

```bash
bash scripts/run_full_pipeline.sh
```

Total time: 3-5 hours

## Verification

After setup, verify the structure:

```bash
# Check data files
ls data/processed/modeling_dataset_final.csv

# Check trained models
ls models/*.pkl

# Check results
ls results/figures/*.png
```

## Dashboard Features

Once running, the dashboard provides:

1. **Overview**: System metrics and model comparison
2. **Predictions**: County-level yield forecasting
3. **Model Performance**: Detailed accuracy analysis
4. **What-If Analysis**: Scenario simulation
5. **Data Explorer**: Interactive data browsing

## Troubleshooting

### Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Missing Data

```bash
# Re-download specific datasets
python src/data_collection/get_yield_by_county.py
```

### Model Loading Errors

```bash
# Retrain models
python scripts/03_train_models.py
```

### Dashboard Port Conflict

```bash
# Use different port
streamlit run app/streamlit_app.py --server.port 8502
```

## Development Setup

For development and modifications:

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install jupyter ipykernel black pylint

# Launch Jupyter for analysis
jupyter notebook notebooks/
```

## Directory Structure Reference

```
us-corn-yield/
├── data/              # Data files (raw and processed)
├── models/            # Trained model artifacts
├── results/           # Figures and performance tables
├── src/               # Source code modules
├── app/               # Streamlit dashboard
├── scripts/           # Pipeline automation
├── notebooks/         # Analysis notebooks
├── docs/              # Detailed documentation
└── tests/             # Unit tests
```

## Next Steps

After setup:

1. Explore the dashboard to understand model capabilities
2. Review `docs/` for detailed methodology
3. Examine `notebooks/` for analysis examples
4. Modify features in `src/features/` for experimentation
5. Retrain models with new data when available

## Support

For issues or questions:

1. Check `docs/` directory for detailed documentation
2. Review error messages and logs
3. Verify all dependencies installed correctly
4. Ensure sufficient disk space and memory

## Performance Expectations

On modern hardware (M1/M2 Mac, 16GB RAM):

- Data download: 2-4 hours
- Data processing: 10 minutes
- Model training: 30-60 minutes
- Dashboard launch: <10 seconds

Lower-spec systems may require more time for model training.

## Data Privacy

All data used is publicly available from US government sources. No proprietary or sensitive information is included.

## License

MIT License - see LICENSE file for details.

