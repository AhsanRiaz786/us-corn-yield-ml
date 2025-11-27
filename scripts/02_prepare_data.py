"""
Prepare and merge all datasets for modeling.

This script:
1. Merges corn statistics (yield, area, production)
2. Engineers weather features from raw data
3. Integrates soil properties
4. Creates final modeling dataset

Requires: All data from 01_download_all_data.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import merge_datasets
from src.features import engineer_weather_features
from src.preprocessing import merge_all_data


def main():
    """Execute data preparation pipeline."""
    print("=" * 80)
    print("US CORN YIELD PREDICTION - DATA PREPARATION PIPELINE")
    print("=" * 80)
    print()
    
    print("[1/3] Merging corn statistics datasets...")
    print("-" * 80)
    try:
        merge_datasets.main()
        print("Corn datasets merged successfully.")
    except Exception as e:
        print(f"Error merging corn datasets: {e}")
        return
    
    print("\n[2/3] Engineering weather features...")
    print("-" * 80)
    try:
        engineer_weather_features.main()
        print("Weather features created successfully.")
    except Exception as e:
        print(f"Error engineering weather features: {e}")
        return
    
    print("\n[3/3] Creating final modeling dataset...")
    print("-" * 80)
    try:
        merge_all_data.main()
        print("Final dataset created successfully.")
    except Exception as e:
        print(f"Error creating final dataset: {e}")
        return
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print("\nFinal dataset saved to: data/processed/modeling_dataset_final.csv")
    print("\nNext step: Run 03_train_models.py to train ML models")


if __name__ == "__main__":
    main()

