"""
Train all machine learning models.

Trains baseline, classical ML, and ensemble models with hyperparameter tuning.
Saves trained models and performance metrics.

Requires: modeling_dataset_final.csv from 02_prepare_data.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import train


def main():
    """Execute model training pipeline."""
    print("=" * 80)
    print("US CORN YIELD PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 80)
    print()
    
    print("Training 5 models with hyperparameter optimization:")
    print("  1. Baseline (3-year moving average)")
    print("  2. Ridge Regression")
    print("  3. Random Forest")
    print("  4. XGBoost")
    print("  5. Gradient Boosting")
    print()
    print("Estimated time: 30-60 minutes")
    print("=" * 80)
    print()
    
    try:
        train.main()
    except Exception as e:
        print(f"Error during model training: {e}")
        return
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("=" * 80)
    print("\nTrained models saved to: models/")
    print("Performance metrics saved to: results/tables/model_comparison.csv")
    print("\nNext step: Run 04_evaluate_models.py for detailed error analysis")


if __name__ == "__main__":
    main()

