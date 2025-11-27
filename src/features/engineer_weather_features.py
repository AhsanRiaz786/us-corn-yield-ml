"""
Engineer Features from Weekly Weather Data
===========================================
Aggregate weekly weather data to county-year level features.

Input:  weather_data_weekly.csv (~3M rows)
Output: weather_features_county_year.csv (~87K rows)

Features: ~25-30 weather features per county-year

Author: Corn Yield Prediction Project
Date: November 2025
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ENGINEERING FEATURES FROM WEEKLY WEATHER DATA")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'weather_data_weekly.csv'
OUTPUT_FILE = 'weather_features_county_year.csv'

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/5] Loading weekly weather data...")

try:
    weather_weekly = pd.read_csv(INPUT_FILE)
    print(f"  ✓ Loaded {len(weather_weekly):,} weekly records")
except FileNotFoundError:
    print(f"  ✗ ERROR: {INPUT_FILE} not found!")
    print(f"  Please run download_nasa_power_weather.py first")
    exit(1)

print(f"  ✓ Counties: {len(weather_weekly[['State', 'County']].drop_duplicates()):,}")
print(f"  ✓ Years: {int(weather_weekly['Year'].min())}-{int(weather_weekly['Year'].max())}")

# ============================================================================
# DEFINE GROWTH PERIODS
# ============================================================================
print("\n[2/5] Defining corn growth periods...")

def assign_growth_period(week):
    """
    Assign growth period based on week number (Week 1 = early April)
    
    Approximate corn growth stages:
    - Weeks 1-4: Planting/Emergence (April)
    - Weeks 5-8: Vegetative Early (May)
    - Weeks 9-12: Vegetative Late (Early June)
    - Weeks 13-16: Reproductive (Late June - Early July) **CRITICAL**
    - Weeks 17-20: Grain Fill Early (Late July)
    - Weeks 21-26: Grain Fill Late + Maturity (August-September)
    """
    if week <= 4:
        return 'Planting'
    elif week <= 8:
        return 'Vegetative_Early'
    elif week <= 12:
        return 'Vegetative_Late'
    elif week <= 16:
        return 'Reproductive'  # MOST CRITICAL
    elif week <= 20:
        return 'GrainFill_Early'
    else:
        return 'GrainFill_Late'

weather_weekly['growth_period'] = weather_weekly['Week'].apply(assign_growth_period)

print(f"  ✓ Growth periods assigned")

# ============================================================================
# AGGREGATE TO COUNTY-YEAR FEATURES
# ============================================================================
print("\n[3/5] Aggregating to county-year features...")
print("  (Creating ~30 features per county-year...)")

grouped = weather_weekly.groupby(['State', 'County', 'Year'])

features_list = []

for (state, county, year), group in tqdm(grouped, desc="County-Years"):
    
    features = {
        'State': state,
        'County': county,
        'Year': year
    }
    
    # ========================================================================
    # TEMPERATURE FEATURES
    # ========================================================================
    
    # Growing Degree Days (GDD) - already calculated per week
    features['gdd_total'] = group['gdd_week'].sum()
    features['gdd_vegetative'] = group[group['growth_period'].str.contains('Vegetative')]['gdd_week'].sum()
    features['gdd_reproductive'] = group[group['growth_period'] == 'Reproductive']['gdd_week'].sum()
    features['gdd_grainfill'] = group[group['growth_period'].str.contains('GrainFill')]['gdd_week'].sum()
    
    # Temperature averages
    features['temp_mean_season'] = group['tavg'].mean()
    features['temp_max_season'] = group['tmax'].max()
    features['temp_min_season'] = group['tmin'].min()
    
    # Critical period temps (Reproductive = weeks 13-16)
    repro = group[group['growth_period'] == 'Reproductive']
    if len(repro) > 0:
        features['temp_mean_reproductive'] = repro['tavg'].mean()
        features['temp_max_reproductive'] = repro['tmax'].max()
    else:
        features['temp_mean_reproductive'] = np.nan
        features['temp_max_reproductive'] = np.nan
    
    # Heat stress (weeks with high temps)
    features['weeks_heat_stress'] = (group['tmax'] > 32).sum()
    features['weeks_extreme_heat'] = (group['tmax'] > 35).sum()
    
    # Temperature variability
    features['temp_std_season'] = group['tavg'].std()
    features['temp_range_avg'] = group['temp_range'].mean()
    
    # ========================================================================
    # PRECIPITATION FEATURES
    # ========================================================================
    
    # Total precipitation
    features['precip_total'] = group['prcp'].sum()
    features['precip_vegetative'] = group[group['growth_period'].str.contains('Vegetative')]['prcp'].sum()
    features['precip_reproductive'] = repro['prcp'].sum() if len(repro) > 0 else np.nan
    features['precip_grainfill'] = group[group['growth_period'].str.contains('GrainFill')]['prcp'].sum()
    
    # Precipitation patterns
    features['precip_mean_weekly'] = group['prcp'].mean()
    features['precip_max_weekly'] = group['prcp'].max()
    features['precip_std'] = group['prcp'].std()
    
    # Dry weeks (< 10mm per week)
    features['weeks_dry'] = (group['prcp'] < 10).sum()
    features['weeks_very_dry'] = (group['prcp'] < 5).sum()
    
    # Wet weeks (> 40mm per week)
    features['weeks_wet'] = (group['prcp'] > 40).sum()
    
    # ========================================================================
    # WATER STRESS INDICATORS
    # ========================================================================
    
    # Combined stress metrics
    if len(repro) > 0 and not pd.isna(features['temp_mean_reproductive']):
        # Water deficit in critical period: low precip + high temp
        repro_precip = repro['prcp'].sum()
        repro_temp = features['temp_mean_reproductive']
        features['water_stress_reproductive'] = (32 - repro_temp) / (repro_precip / 10 + 1)
    else:
        features['water_stress_reproductive'] = np.nan
    
    # Heat-moisture stress
    features['heat_moisture_stress'] = features['weeks_heat_stress'] / (features['precip_total'] / 100 + 1)
    
    # ========================================================================
    # HUMIDITY FEATURES
    # ========================================================================
    
    features['rh_mean'] = group['rh'].mean()
    features['rh_reproductive'] = repro['rh'].mean() if len(repro) > 0 else np.nan
    
    # High humidity can promote disease
    features['weeks_high_humidity'] = (group['rh'] > 80).sum()
    
    # ========================================================================
    # TEMPORAL PATTERNS
    # ========================================================================
    
    # Early vs late season comparison
    early_weeks = group[group['Week'] <= 13]
    late_weeks = group[group['Week'] > 13]
    
    if len(early_weeks) > 0 and len(late_weeks) > 0:
        features['temp_early_vs_late'] = early_weeks['tavg'].mean() - late_weeks['tavg'].mean()
        features['precip_early_vs_late'] = early_weeks['prcp'].sum() - late_weeks['prcp'].sum()
    else:
        features['temp_early_vs_late'] = np.nan
        features['precip_early_vs_late'] = np.nan
    
    features_list.append(features)

df_features = pd.DataFrame(features_list)

print(f"\n  ✓ Created {len(df_features):,} county-year records")
print(f"  ✓ Features per record: {len(df_features.columns) - 3}")

# ============================================================================
# CALCULATE ANOMALIES
# ============================================================================
print("\n[4/5] Calculating climate anomalies...")

# 30-year normal period (1991-2020)
normal_data = df_features[(df_features['Year'] >= 1991) & (df_features['Year'] <= 2020)]

county_normals = normal_data.groupby(['State', 'County']).agg({
    'gdd_total': 'mean',
    'precip_total': 'mean',
    'temp_mean_season': 'mean',
    'precip_reproductive': 'mean'
}).reset_index()

county_normals.columns = ['State', 'County', 'gdd_normal', 'precip_normal', 
                          'temp_normal', 'precip_repro_normal']

# Merge
df_features = df_features.merge(county_normals, on=['State', 'County'], how='left')

# Calculate anomalies
df_features['gdd_anomaly'] = df_features['gdd_total'] - df_features['gdd_normal']
df_features['precip_anomaly_mm'] = df_features['precip_total'] - df_features['precip_normal']
df_features['precip_anomaly_pct'] = (
    (df_features['precip_total'] - df_features['precip_normal']) / 
    (df_features['precip_normal'] + 1) * 100  # Add 1 to avoid division by zero
)
df_features['temp_anomaly'] = df_features['temp_mean_season'] - df_features['temp_normal']

# Drop normal columns (not needed for modeling)
df_features = df_features.drop(['gdd_normal', 'precip_normal', 'temp_normal', 'precip_repro_normal'], axis=1)

print(f"  ✓ Anomalies calculated")

# ============================================================================
# HANDLE MISSING VALUES
# ============================================================================
print("\n[5/5] Finalizing dataset...")

# Fill NaN values with 0 (these are mostly from missing reproductive period data)
df_features = df_features.fillna(0)

# Sort
df_features = df_features.sort_values(['State', 'County', 'Year'])

# Save
df_features.to_csv(OUTPUT_FILE, index=False)

print(f"  ✓ Saved: {OUTPUT_FILE}")
print(f"  ✓ Records: {len(df_features):,}")
print(f"  ✓ Features: {len(df_features.columns) - 3}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING SUMMARY")
print("="*70)

print(f"\nOutput: {OUTPUT_FILE}")
print(f"Records: {len(df_features):,}")
print(f"Years: {int(df_features['Year'].min())}-{int(df_features['Year'].max())}")
print(f"Total features: {len(df_features.columns) - 3}")

print(f"\nFeature categories:")
print(f"  Temperature: ~12 features")
print(f"  Precipitation: ~10 features")
print(f"  Stress indicators: ~5 features")
print(f"  Anomalies: ~4 features")

print(f"\nSample statistics:")
sample_cols = ['gdd_total', 'precip_total', 'weeks_heat_stress', 'precip_reproductive']
print(df_features[sample_cols].describe().to_string())

print(f"\nSample records:")
display_cols = ['State', 'County', 'Year', 'gdd_total', 'precip_total', 'temp_mean_season']
print(df_features[display_cols].head(3).to_string(index=False))

import os
size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
print(f"\nFile size: {size_mb:.1f} MB")

print("\n" + "="*70)
print("✓ WEATHER FEATURE ENGINEERING COMPLETE!")
print("="*70)
print(f"\nNext: Merge weather + soil with corn data")
print("="*70 + "\n")

