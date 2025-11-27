#!/bin/bash

# Complete pipeline execution script for US Corn Yield Prediction System
# This script runs the entire workflow from data download to model evaluation

set -e

echo "================================================================================"
echo "US CORN YIELD PREDICTION SYSTEM - FULL PIPELINE"
echo "================================================================================"
echo ""
echo "This script will execute the complete pipeline:"
echo "  1. Download all data sources"
echo "  2. Prepare and merge datasets"
echo "  3. Train machine learning models"
echo "  4. Evaluate models and generate visualizations"
echo ""
echo "Estimated total time: 3-5 hours"
echo "================================================================================"
echo ""

read -p "Continue with full pipeline execution? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Pipeline execution cancelled."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo ""
echo "================================================================================"
echo "STEP 1/4: DATA DOWNLOAD"
echo "================================================================================"
python scripts/01_download_all_data.py
if [ $? -ne 0 ]; then
    echo "Error in data download step. Exiting."
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 2/4: DATA PREPARATION"
echo "================================================================================"
python scripts/02_prepare_data.py
if [ $? -ne 0 ]; then
    echo "Error in data preparation step. Exiting."
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 3/4: MODEL TRAINING"
echo "================================================================================"
python scripts/03_train_models.py
if [ $? -ne 0 ]; then
    echo "Error in model training step. Exiting."
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 4/4: MODEL EVALUATION"
echo "================================================================================"
python scripts/04_evaluate_models.py
if [ $? -ne 0 ]; then
    echo "Error in model evaluation step. Exiting."
    exit 1
fi

echo ""
echo "================================================================================"
echo "PIPELINE EXECUTION COMPLETE"
echo "================================================================================"
echo ""
echo "All steps completed successfully!"
echo ""
echo "Outputs:"
echo "  - Data: data/processed/modeling_dataset_final.csv"
echo "  - Models: models/*.pkl"
echo "  - Results: results/tables/*.csv"
echo "  - Figures: results/figures/*.png"
echo ""
echo "To launch the interactive dashboard:"
echo "  streamlit run app/streamlit_app.py"
echo ""

