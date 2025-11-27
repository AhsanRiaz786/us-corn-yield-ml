"""
Download all required data from external sources.

This script orchestrates the complete data collection pipeline:
1. USDA NASS QuickStats: Corn yield, area, production
2. US Census: County centroids for spatial queries
3. NASA POWER: Weather data (weekly aggregated)
4. USDA NRCS: Soil properties

Run this script first to set up the project data.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection import get_yield_by_county
from src.data_collection import get_area_planted_by_county
from src.data_collection import get_area_harvested_by_county
from src.data_collection import get_production_by_county
from src.data_collection import get_county_centroids
from src.data_collection import get_weather_data
from src.data_collection import get_soil_data


def main():
    """Execute complete data download pipeline."""
    print("=" * 80)
    print("US CORN YIELD PREDICTION - DATA DOWNLOAD PIPELINE")
    print("=" * 80)
    print()
    
    print("This will download data from multiple sources:")
    print("  1. USDA NASS QuickStats (corn statistics)")
    print("  2. US Census TIGER/Line (county centroids)")
    print("  3. NASA POWER (weather data)")
    print("  4. USDA NRCS (soil properties)")
    print()
    print("Estimated time: 2-4 hours depending on network speed")
    print("=" * 80)
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    print("\n[1/7] Downloading corn yield data...")
    print("-" * 80)
    try:
        get_yield_by_county.main()
        print("Yield data download complete.")
    except Exception as e:
        print(f"Error downloading yield data: {e}")
        return
    
    print("\n[2/7] Downloading area planted data...")
    print("-" * 80)
    try:
        get_area_planted_by_county.main()
        print("Area planted data download complete.")
    except Exception as e:
        print(f"Error downloading area planted data: {e}")
        return
    
    print("\n[3/7] Downloading area harvested data...")
    print("-" * 80)
    try:
        get_area_harvested_by_county.main()
        print("Area harvested data download complete.")
    except Exception as e:
        print(f"Error downloading area harvested data: {e}")
        return
    
    print("\n[4/7] Downloading production data...")
    print("-" * 80)
    try:
        get_production_by_county.main()
        print("Production data download complete.")
    except Exception as e:
        print(f"Error downloading production data: {e}")
        return
    
    print("\n[5/7] Downloading county centroids...")
    print("-" * 80)
    try:
        get_county_centroids.main()
        print("County centroids download complete.")
    except Exception as e:
        print(f"Error downloading centroids: {e}")
        return
    
    print("\n[6/7] Downloading weather data (this will take time)...")
    print("-" * 80)
    try:
        get_weather_data.main()
        print("Weather data download complete.")
    except Exception as e:
        print(f"Error downloading weather data: {e}")
        return
    
    print("\n[7/7] Downloading soil data...")
    print("-" * 80)
    try:
        get_soil_data.main()
        print("Soil data download complete.")
    except Exception as e:
        print(f"Error downloading soil data: {e}")
        return
    
    print("\n" + "=" * 80)
    print("DATA DOWNLOAD COMPLETE")
    print("=" * 80)
    print("\nAll data files saved to: data/raw/")
    print("\nNext step: Run 02_prepare_data.py to process and merge datasets")


if __name__ == "__main__":
    main()

