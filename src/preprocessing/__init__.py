"""
Data preprocessing and integration module.

Handles merging, cleaning, and preparation of datasets for modeling.
Integrates corn statistics, weather features, and soil properties into
a unified modeling dataset.
"""

from pathlib import Path

# Define data directories
RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
]

