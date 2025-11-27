#!/bin/bash
# Run Streamlit dashboard for US Corn Yield Prediction

# Navigate to project root
cd "$(dirname "$0")"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit is not installed."
    echo "Please install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Check if data file exists
if [ ! -f "data/processed/modeling_dataset_final.csv" ]; then
    echo "Warning: Data file not found at data/processed/modeling_dataset_final.csv"
    echo "Please ensure data is prepared before running the dashboard."
fi

# Check if models exist
if [ ! -f "models/xgboost_model.pkl" ]; then
    echo "Warning: Model files not found in models/ directory"
    echo "Please train models before running the dashboard."
fi

# Run Streamlit app
echo "Starting Streamlit dashboard..."
echo "The app will open in your default web browser."
echo "Working directory: $(pwd)"
echo ""

# Run from project root with explicit path
streamlit run app/streamlit_app.py --server.headless=false

