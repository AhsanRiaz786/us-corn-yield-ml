# Streamlit Dashboard - US Corn Yield Prediction

## Overview

Professional, comprehensive Proof of Concept (POC) dashboard for the US Corn Yield Prediction project. The dashboard demonstrates the complete ML pipeline, model capabilities, and data insights through an interactive web interface.

## Features

### Pages

1. **Home** - Landing page with project overview and key metrics
2. **Yield Prediction** - Interactive yield prediction interface (main feature)
3. **Data Exploration** - Interactive EDA with filters and visualizations
4. **Scenarios** - What-if analysis for different weather conditions
5. **Model Insights** - Deep dive into model performance and analysis
6. **About** - Methodology and documentation

### Key Capabilities

- **Real-time Predictions**: Make yield predictions for any county and year
- **Historical Context**: View historical yields and trends
- **Custom Inputs**: Override weather and soil parameters for custom scenarios
- **Interactive Visualizations**: Explore data with Plotly charts
- **Scenario Analysis**: Compare multiple what-if scenarios
- **Model Comparison**: Compare performance across all trained models
- **Feature Importance**: Explore which factors drive predictions

## Requirements

All dependencies are listed in `requirements.txt` in the project root. Key packages:

- Streamlit >= 1.28.0
- Plotly >= 5.14.0
- Pandas, NumPy
- scikit-learn, XGBoost
- Joblib

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Data and Models**:
   - Data file: `data/processed/modeling_dataset_final.csv`
   - Models: `models/xgboost_model.pkl`, `models/scaler.pkl`, etc.
   - Feature columns: `models/feature_columns.pkl`

3. **Run the Application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   
   Or from the project root:
   ```bash
   streamlit run streamlit_app.py
   ```

## Directory Structure

```
app/
├── streamlit_app.py           # Main entry point
├── config.py                  # Configuration constants
├── pages/
│   ├── 1_Yield_Prediction.py  # Main prediction interface
│   ├── 2_Data_Exploration.py  # EDA visualizations
│   ├── 3_Scenarios.py         # What-if scenario analysis
│   ├── 4_Model_Insights.py    # Model performance analysis
│   └── 5_About.py             # Methodology documentation
├── utils/
│   ├── data_loader.py         # Data loading & caching
│   ├── model_loader.py        # Model loading utilities
│   ├── predictions.py         # Prediction logic
│   └── visualizations.py      # Reusable plot functions
└── README.md                  # This file
```

## Usage Guide

### Making a Prediction

1. Navigate to **Yield Prediction** page
2. Select a **State** and **County** from dropdowns
3. Choose a **Year** for prediction
4. (Optional) Override weather or soil data in **Advanced Options**
5. Click **Generate Prediction**
6. View results with confidence intervals and historical comparison

### Exploring Data

1. Navigate to **Data Exploration** page
2. Use sidebar filters to select states, counties, and year range
3. Explore different tabs:
   - Geographic Analysis
   - Temporal Trends
   - Yield Distribution
   - Weather Impact
4. View filtered data table at bottom

### Running Scenarios

1. Navigate to **Scenarios** page
2. Set base scenario (state, county, year)
3. Select scenario type:
   - Preset scenarios (Drought, Optimal, Extreme Heat, Excessive Rain)
   - Custom scenario with manual parameter adjustment
4. Compare results across scenarios

## Configuration

Configuration constants are defined in `app/config.py`:

- Model paths and metadata
- Feature categories
- Default values
- UI colors and styling
- Paths to data and results directories

## Performance Notes

- Data loading is cached using `@st.cache_data`
- Model loading is cached using `@st.cache_resource`
- Initial load may take a few seconds
- Large datasets are filtered efficiently

## Troubleshooting

**Error: Data file not found**
- Ensure `data/processed/modeling_dataset_final.csv` exists
- Run data preparation scripts first

**Error: Model file not found**
- Ensure models are trained and saved in `models/` directory
- Run model training script first

**Import errors**
- Ensure you're running from project root or `app/` directory
- Check that all dependencies are installed
- Verify Python path includes project root

**Slow performance**
- First run will be slower due to caching
- Subsequent runs should be faster
- Large filters may take time to process

## Deployment

The app is ready for deployment on:
- Streamlit Cloud
- Local server
- Docker container

For Streamlit Cloud:
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set working directory to `app/` or project root
4. Ensure data files are accessible (or include download step)

## Future Enhancements

Potential additions (out of scope for POC):
- User authentication
- Saved predictions
- Export reports
- API endpoints
- Real-time data updates
- Mobile-responsive optimizations

## License

See project root LICENSE file.

## Support

For issues or questions, refer to the main project README or contact the project maintainer.

