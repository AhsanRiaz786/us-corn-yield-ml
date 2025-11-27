"""
Data collection module for US corn yield prediction.

This module provides utilities for downloading data from multiple sources:
- USDA NASS QuickStats: Corn yield, area, and production statistics
- NASA POWER: Daily weather data (temperature, precipitation, humidity)
- USDA NRCS: Soil properties (AWC, pH, clay content, organic matter)
- US Census: County geographic centroids
"""

from pathlib import Path

# Define data output directory
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "DATA_DIR",
]

