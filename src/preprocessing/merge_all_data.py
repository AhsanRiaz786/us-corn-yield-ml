"""
Merge All Data: Corn + Soil + Weather
======================================
Creates the final modeling dataset by merging:
1. Corn data (modeling_dataset_complete.csv)
2. Soil data (county_soil_aggregates.csv)
3. Weather features (weather_features_county_year.csv)

Output: modeling_dataset_final.csv

Author: Corn Yield Prediction Project
Date: November 2025
"""

import pandas as pd
import numpy as np

print("="*70)
print("MERGING CORN + SOIL + WEATHER DATA")
print("="*70)

# ============================================================================
# LOAD DATASETS
# ============================================================================
print("\n[1/5] Loading datasets...")

# Load corn data
corn_data = pd.read_csv('modeling_dataset_complete.csv')
print(f"  ✓ Corn data: {len(corn_data):,} records")
print(f"    Years: {int(corn_data['Year'].min())}-{int(corn_data['Year'].max())}")

# Load soil data  
soil_data = pd.read_csv('county_soil_aggregates.csv')
print(f"  ✓ Soil data: {len(soil_data):,} counties")

# Load weather features
weather_data = pd.read_csv('weather_features_county_year.csv')
print(f"  ✓ Weather features: {len(weather_data):,} county-years")
print(f"    Years: {int(weather_data['Year'].min())}-{int(weather_data['Year'].max())}")

# ============================================================================
# PREPARE SOIL DATA
# ============================================================================
print("\n[2/5] Preparing soil data...")

# Rename soil columns to match merge
soil_data = soil_data.rename(columns={
    'AVG_AWC': 'Soil_AWC',
    'AVG_CLAY': 'Soil_Clay_Pct',
    'AVG_PH': 'Soil_pH',
    'AVG_OM': 'Soil_Organic_Matter_Pct'
})

# Select only needed columns
soil_cols = ['State', 'County', 'Soil_AWC', 'Soil_Clay_Pct', 'Soil_pH', 'Soil_Organic_Matter_Pct']
soil_data = soil_data[soil_cols]

# Fill missing values with state averages
print(f"  Calculating state-level averages for missing counties...")
state_averages = soil_data.groupby('State')[['Soil_AWC', 'Soil_Clay_Pct', 'Soil_pH', 'Soil_Organic_Matter_Pct']].mean()

# For each state, fill missing values
for state in soil_data['State'].unique():
    mask = soil_data['State'] == state
    for col in ['Soil_AWC', 'Soil_Clay_Pct', 'Soil_pH', 'Soil_Organic_Matter_Pct']:
        soil_data.loc[mask, col] = soil_data.loc[mask, col].fillna(state_averages.loc[state, col])

# Fill any remaining NaN with national averages
national_avg = {
    'Soil_AWC': 0.15,
    'Soil_Clay_Pct': 25.0,
    'Soil_pH': 6.5,
    'Soil_Organic_Matter_Pct': 2.5
}
soil_data = soil_data.fillna(national_avg)

print(f"  ✓ Soil data prepared: {len(soil_data):,} counties")

# ============================================================================
# MERGE CORN + SOIL
# ============================================================================
print("\n[3/5] Merging corn data with soil...")

# Merge on State and County (soil is static - same for all years)
merged_data = corn_data.merge(
    soil_data,
    on=['State', 'County'],
    how='left'
)

# Check merge
before_rows = len(corn_data)
after_rows = len(merged_data)
print(f"  ✓ Merged successfully")
print(f"    Before: {before_rows:,} records")
print(f"    After:  {after_rows:,} records")
print(f"    Soil data coverage: {merged_data['Soil_AWC'].notna().sum() / len(merged_data) * 100:.1f}%")

# Fill any remaining missing soil values with national averages
for col, val in national_avg.items():
    merged_data[col] = merged_data[col].fillna(val)

# ============================================================================
# MERGE WITH WEATHER
# ============================================================================
print("\n[4/5] Merging with weather data...")

# Merge on State, County, and Year
final_data = merged_data.merge(
    weather_data,
    on=['State', 'County', 'Year'],
    how='inner'  # Only keep records where we have weather data
)

print(f"  ✓ Merged successfully")
print(f"    Before: {len(merged_data):,} records")
print(f"    After:  {len(final_data):,} records")
print(f"    Records with weather: {len(final_data) / len(merged_data) * 100:.1f}%")

# ============================================================================
# FINAL DATASET STATS
# ============================================================================
print("\n[5/5] Finalizing dataset...")

# Sort by State, County, Year
final_data = final_data.sort_values(['State', 'County', 'Year'])

# Save
final_data.to_csv('modeling_dataset_final.csv', index=False)

print(f"  ✓ Saved: modeling_dataset_final.csv")
print(f"  ✓ Total records: {len(final_data):,}")
print(f"  ✓ Total features: {len(final_data.columns)}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL DATASET SUMMARY")
print("="*70)

print(f"\nDataset: modeling_dataset_final.csv")
print(f"Records: {len(final_data):,}")
print(f"Features: {len(final_data.columns)}")

print(f"\nCoverage:")
print(f"  States: {final_data['State'].nunique()}")
print(f"  Counties: {final_data[['State', 'County']].drop_duplicates().shape[0]:,}")
print(f"  Years: {int(final_data['Year'].min())}-{int(final_data['Year'].max())}")

print(f"\nFeature Breakdown:")
print(f"  Corn variables: ~10 (yield, area, production, etc.)")
print(f"  Soil variables: 4 (AWC, Clay%, pH, OM%)")
print(f"  Weather variables: 34 (GDD, precip, temp, stress, anomalies)")
print(f"  Identifiers: 3 (State, County, Year)")

print(f"\nData Quality:")
print(f"  Missing values: {final_data.isnull().sum().sum()}")
print(f"  Complete records: {(~final_data.isnull().any(axis=1)).sum():,}")

print(f"\nSample data:")
display_cols = ['State', 'County', 'Year', 'Yield', 'gdd_total', 'precip_total', 'Soil_AWC']
print(final_data[display_cols].head(3).to_string(index=False))

import os
size_mb = os.path.getsize('modeling_dataset_final.csv') / 1024 / 1024
print(f"\nFile size: {size_mb:.1f} MB")

print("\n" + "="*70)
print("✓ DATA MERGE COMPLETE!")
print("="*70)
print(f"\nYou now have the complete modeling dataset!")
print(f"  • {len(final_data):,} county-year records")
print(f"  • {len(final_data.columns)} total features")
print(f"  • Ready for model training!")
print("\n" + "="*70)
print("Next Steps:")
print("  1. Train baseline models")
print("  2. Train advanced models (RF, XGBoost, Neural Net)")
print("  3. Compare performance")
print("  4. Build dashboard")
print("="*70 + "\n")

