"""
Download Weekly Aggregated Weather Data from NASA POWER
========================================================
Downloads daily data but immediately aggregates to weekly summaries.

Weekly aggregation: Much more manageable dataset
- Daily: 22M rows (2,800 counties × 43 years × 183 days)  
- Weekly: 3.1M rows (2,800 counties × 43 years × 26 weeks) ✅

Growing Season: April-September = ~26 weeks
Years: 1981-2023

Input:  county_centroids.csv
Output: weather_data_weekly.csv (~3M rows instead of 22M!)

Author: Corn Yield Prediction Project
Date: November 2025
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time
from tqdm import tqdm
import os

print("="*70)
print("DOWNLOADING WEEKLY WEATHER DATA FROM NASA POWER")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'county_centroids.csv'
OUTPUT_FILE = 'weather_data_weekly.csv'
CHECKPOINT_FILE = 'weekly_weather_checkpoint.csv'

# NASA POWER API
API_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Weather parameters
PARAMETERS = [
    'T2M',           # Temperature at 2m (°C)
    'T2M_MAX',       # Max Temperature (°C)
    'T2M_MIN',       # Min Temperature (°C)
    'PRECTOTCORR',   # Precipitation (mm/day)
    'RH2M'           # Relative Humidity (%)
]

# Year range
START_YEAR = 1981
END_YEAR = 2023

CHECKPOINT_FREQ = 50  # Save every 50 counties

# ============================================================================
# LOAD COUNTIES
# ============================================================================
print("\n[1/5] Loading county centroids...")

try:
    counties = pd.read_csv(INPUT_FILE)
    print(f"  ✓ Loaded {len(counties):,} counties")
except FileNotFoundError:
    print(f"  ✗ ERROR: {INPUT_FILE} not found!")
    exit(1)

# ============================================================================
# CHECK WHAT'S ALREADY DOWNLOADED
# ============================================================================
print("\n[2/5] Checking existing data...")

completed = set()

# Check what counties are in the actual data file
if os.path.exists(OUTPUT_FILE):
    print(f"  ℹ Found existing data file, analyzing...")
    existing_df = pd.read_csv(OUTPUT_FILE)
    
    # Get unique counties that have data
    counties_with_data = existing_df[['State', 'County']].drop_duplicates()
    completed = set(zip(counties_with_data['State'], counties_with_data['County']))
    
    print(f"  ✓ Found data for {len(completed):,} counties")
    
    # Verify data completeness for each county
    print(f"  ℹ Verifying data completeness...")
    n_years = END_YEAR - START_YEAR + 1
    expected_weeks_per_county = n_years * 26  # 26 weeks per year
    
    # Count records per county
    county_counts = existing_df.groupby(['State', 'County']).size()
    
    # Counties with insufficient data (less than 80% of expected)
    min_acceptable = int(expected_weeks_per_county * 0.8)
    incomplete_counties = county_counts[county_counts < min_acceptable]
    
    if len(incomplete_counties) > 0:
        print(f"  ⚠ Found {len(incomplete_counties)} counties with incomplete data (< 80%)")
        # Remove incomplete counties from completed set so they get re-downloaded
        for (state, county) in incomplete_counties.index:
            completed.discard((state, county))
        print(f"  → Will re-download these counties")
else:
    print(f"  ℹ No existing data file found, starting fresh")

# Mark counties as completed or remaining
counties['completed'] = counties.apply(
    lambda row: (row['State'], row['County']) in completed, axis=1
)
counties_remaining = counties[~counties['completed']]

print(f"  ✓ Counties with complete data: {len(completed):,}")
print(f"  → Counties to download: {len(counties_remaining):,}")

# Load or initialize checkpoint
if os.path.exists(CHECKPOINT_FILE):
    checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
    completed_info = checkpoint_df.to_dict('records')
else:
    completed_info = [{'State': state, 'County': county} for state, county in completed]

# ============================================================================
# ESTIMATE
# ============================================================================
print("\n[3/5] Estimating download...")

n_counties = len(counties_remaining)
n_years = END_YEAR - START_YEAR + 1
weeks_per_year = 26  # Apr-Sept = ~26 weeks
expected_rows = n_counties * n_years * weeks_per_year

print(f"  Counties to download: {n_counties:,}")
print(f"  Years: {n_years} ({START_YEAR}-{END_YEAR})")
print(f"  Weeks per year: {weeks_per_year}")
print(f"  Expected new rows: {expected_rows:,}")
print(f"  Estimated time: {n_counties * 2 / 60:.1f} minutes")

if n_counties == 0:
    print("\n✓ All counties already have complete data!")
    print("="*70 + "\n")
    exit(0)

# ============================================================================
# DOWNLOAD AND AGGREGATE FUNCTION
# ============================================================================

def download_and_aggregate_weekly(lat, lon, start_year, end_year):
    """
    Download daily data and immediately aggregate to weekly summaries.
    Returns weekly aggregated DataFrame.
    """
    
    params_str = ','.join(PARAMETERS)
    url = f"{API_BASE}?parameters={params_str}&community=AG&longitude={lon}&latitude={lat}&start={start_year}0101&end={end_year}1231&format=JSON"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if 'properties' not in data or 'parameter' not in data['properties']:
            return None
        
        param_data = data['properties']['parameter']
        
        # Convert to daily DataFrame
        daily_records = []
        for date_str in param_data['T2M'].keys():
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            # Filter growing season only (Apr-Sept)
            if month < 4 or month > 9:
                continue
            
            daily_records.append({
                'Date': pd.to_datetime(f"{year}-{month:02d}-{day:02d}"),
                'Year': year,
                'Month': month,
                'tmin': param_data['T2M_MIN'][date_str],
                'tmax': param_data['T2M_MAX'][date_str],
                'tavg': param_data['T2M'][date_str],
                'prcp': param_data['PRECTOTCORR'][date_str],
                'rh': param_data['RH2M'][date_str]
            })
        
        if len(daily_records) == 0:
            return None
        
        df_daily = pd.DataFrame(daily_records)
        
        # ====================================================================
        # AGGREGATE TO WEEKLY
        # ====================================================================
        
        # Create week number (starting from April 1 = Week 1)
        df_daily['DayOfYear'] = df_daily['Date'].dt.dayofyear
        
        # April 1 is approximately day 91 of year
        april_1_doy = 91
        df_daily['DaysSinceApril1'] = df_daily['DayOfYear'] - april_1_doy
        df_daily['Week'] = (df_daily['DaysSinceApril1'] // 7) + 1
        
        # Group by Year and Week
        weekly = df_daily.groupby(['Year', 'Week']).agg({
            'tmin': 'mean',           # Average weekly min temp
            'tmax': 'mean',           # Average weekly max temp
            'tavg': 'mean',           # Average weekly mean temp
            'prcp': 'sum',            # Total weekly precipitation
            'rh': 'mean',             # Average weekly humidity
            'Date': 'min'             # Week start date
        }).reset_index()
        
        # Add derived metrics
        weekly['temp_range'] = weekly['tmax'] - weekly['tmin']
        weekly['heat_stress_score'] = (weekly['tmax'] > 32).astype(int)
        weekly['gdd_week'] = ((weekly['tavg'].clip(upper=30) - 10).clip(lower=0) * 7)  # GDD for the week
        
        return weekly
        
    except Exception as e:
        return None

# ============================================================================
# DOWNLOAD DATA
# ============================================================================
print("\n[4/5] Downloading and aggregating to weekly data...")
print("  Downloading daily → immediately aggregating to weekly")
print("  Progress saved every 50 counties\n")

all_data = []
error_count = 0
success_count = 0
counties_this_session = 0

pbar = tqdm(total=len(counties_remaining), desc="Counties", unit="county")

for idx, row in counties_remaining.iterrows():
    state = row['State']
    county = row['County']
    lat = row['Latitude']
    lon = row['Longitude']
    
    pbar.set_description(f"{state[:15]:<15} {county[:20]:<20}")
    
    try:
        # Download and aggregate
        weekly_df = download_and_aggregate_weekly(lat, lon, START_YEAR, END_YEAR)
        
        if weekly_df is not None and len(weekly_df) > 0:
            # Add identifiers
            weekly_df['State'] = state
            weekly_df['County'] = county
            weekly_df['Latitude'] = lat
            weekly_df['Longitude'] = lon
            
            all_data.append(weekly_df)
            success_count += 1
        else:
            error_count += 1
            
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted! Saving progress...")
        if len(all_data) > 0:
            df_temp = pd.concat(all_data, ignore_index=True)
            # Append to existing file if it exists
            if os.path.exists(OUTPUT_FILE):
                df_temp.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            else:
                df_temp.to_csv(OUTPUT_FILE, index=False)
        # Save checkpoint (this already has all completed counties)
        pd.DataFrame(completed_info).to_csv(CHECKPOINT_FILE, index=False)
        print("  ✓ Progress saved")
        exit(0)
        
    except Exception as e:
        error_count += 1
        continue
    
    # Add to completed list
    completed_info.append({'State': state, 'County': county})
    counties_this_session += 1
    
    # Checkpoint save - APPEND data, OVERWRITE checkpoint
    if counties_this_session % CHECKPOINT_FREQ == 0:
        if len(all_data) > 0:
            df_temp = pd.concat(all_data, ignore_index=True)
            
            # Append to existing file (add header only if file doesn't exist)
            if os.path.exists(OUTPUT_FILE):
                df_temp.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            else:
                df_temp.to_csv(OUTPUT_FILE, index=False)
            
            # Update checkpoint file with ALL completed counties (overwrite)
            pd.DataFrame(completed_info).to_csv(CHECKPOINT_FILE, index=False)
            
            # Clear memory
            all_data = []
        
        pbar.set_postfix({'Success': success_count, 'Errors': error_count})
    
    pbar.update(1)
    time.sleep(0.2)

pbar.close()

# ============================================================================
# FINAL SAVE - Save any remaining data
# ============================================================================
print("\n[5/5] Saving final data...")

if len(all_data) > 0:
    df_final = pd.concat(all_data, ignore_index=True)
    
    # Append to existing file
    if os.path.exists(OUTPUT_FILE):
        df_final.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"  ✓ Saved {len(df_final):,} remaining records")

# Save final checkpoint
if len(completed_info) > 0:
    pd.DataFrame(completed_info).to_csv(CHECKPOINT_FILE, index=False)

# ============================================================================
# VALIDATION
# ============================================================================
print("\n" + "="*70)
print("VALIDATION")
print("="*70)

try:
    df = pd.read_csv(OUTPUT_FILE)
    print(f"Total weekly records: {len(df):,}")
    print(f"Unique counties: {len(df[['State', 'County']].drop_duplicates()):,}")
    print(f"Years: {int(df['Year'].min())}-{int(df['Year'].max())}")
    
    # Check completeness
    n_years = END_YEAR - START_YEAR + 1
    expected_weeks_per_county = n_years * 26
    county_counts = df.groupby(['State', 'County']).size()
    
    complete_counties = (county_counts >= expected_weeks_per_county * 0.8).sum()
    incomplete_counties = (county_counts < expected_weeks_per_county * 0.8).sum()
    
    print(f"\nCompleteness check:")
    print(f"  ✓ Complete counties (≥80% data): {complete_counties:,}")
    if incomplete_counties > 0:
        print(f"  ⚠ Incomplete counties (<80% data): {incomplete_counties:,}")
        print(f"    → Run script again to retry these counties")
    
    print(f"\nData quality:")
    print(f"  Temp range: {df['tmin'].min():.1f}°C to {df['tmax'].max():.1f}°C")
    print(f"  Weekly precip: {df['prcp'].min():.1f} to {df['prcp'].max():.1f} mm/week")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    
    print(f"\nSample data:")
    print(df[['State', 'County', 'Year', 'Week', 'tavg', 'prcp', 'gdd_week']].head(5).to_string(index=False))
    
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"\nFile size: {size_mb:.1f} MB")
    
except Exception as e:
    print(f"⚠ Validation error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("DOWNLOAD SUMMARY")
print("="*70)
print(f"Successful this session: {success_count:,} counties")
print(f"Failed this session: {error_count:,} counties")
if success_count + error_count > 0:
    print(f"Success rate: {success_count/(success_count+error_count)*100:.1f}%")

print(f"Total completed: {len(completed_info):,} counties")

if error_count > 0:
    print(f"\n⚠ {error_count} counties failed to download")
    print(f"  → Simply run this script again to retry failed counties")

print(f"\n📊 Data Reduction:")
print(f"  If we used daily: ~22 million rows")
print(f"  Using weekly: reduced by ~7x ✅")

print(f"\nOutput: {OUTPUT_FILE}")

print("\n" + "="*70)
print("✓ WEEKLY WEATHER DATA DOWNLOAD COMPLETE!")
print("="*70)
print(f"\nNext: Run engineer_weekly_weather_features.py")
print("="*70 + "\n")