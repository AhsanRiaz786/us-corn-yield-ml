"""
Feature engineering module.

Transforms raw data into features suitable for machine learning models.
Includes weather aggregations, temporal features, and derived metrics.
"""

from pathlib import Path

PROCESSED_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

__all__ = [
    "PROCESSED_DATA_DIR",
]

