"""
STEP 1: Get County Geographic Centroids
=========================================
Extract latitude and longitude for each US county to enable weather data download.

Input:  modeling_dataset_complete.csv
Output: county_centroids.csv (State, County, FIPS, Latitude, Longitude)

Author: Corn Yield Prediction Project
Date: November 2025
"""

import pandas as pd
import geopandas as gpd
import os
from pathlib import Path

print("="*70)
print("STEP 1: EXTRACTING COUNTY CENTROIDS")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'modeling_dataset_complete.csv'
OUTPUT_FILE = 'county_centroids.csv'

# ============================================================================
# STEP 1.1: LOAD CORN DATA
# ============================================================================
print("\n[1/4] Loading corn yield data...")
try:
    corn_data = pd.read_csv(INPUT_FILE)
    print(f"  ✓ Loaded {len(corn_data):,} records")
except FileNotFoundError:
    print(f"  ✗ ERROR: {INPUT_FILE} not found!")
    print(f"  Please run this script from the weather_pipeline/ directory")
    exit(1)

# Extract unique counties
unique_counties = corn_data[['State', 'County', 'State ANSI', 'County ANSI']].drop_duplicates()
print(f"  ✓ Found {len(unique_counties):,} unique counties")

# ============================================================================
# STEP 1.2: DOWNLOAD US COUNTY BOUNDARIES
# ============================================================================
print("\n[2/4] Loading US county boundaries...")
print("  (This may take a minute on first run...)")

try:
    # Try to load from built-in geodatasets
    import geodatasets
    
    # Get US counties shapefile
    counties_url = geodatasets.get_path('geoda.us_counties')
    counties_gdf = gpd.read_file(counties_url)
    print(f"  ✓ Loaded {len(counties_gdf):,} US counties from geodatasets")
    
except ImportError:
    print("  Installing geodatasets package...")
    os.system('pip install geodatasets -q')
    import geodatasets
    counties_url = geodatasets.get_path('geoda.us_counties')
    counties_gdf = gpd.read_file(counties_url)
    print(f"  ✓ Loaded {len(counties_gdf):,} US counties")
except Exception as e:
    print(f"  ✗ Error loading from geodatasets: {e}")
    print("\n  Trying alternative: US Census TIGER data...")
    
    # Fallback: Download from US Census
    census_url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip"
    try:
        counties_gdf = gpd.read_file(census_url)
        print(f"  ✓ Downloaded {len(counties_gdf):,} counties from US Census")
    except Exception as e2:
        print(f"  ✗ Error: {e2}")
        print("\n  FALLBACK: Using manual county centroid lookup...")
        use_manual = True

# ============================================================================
# STEP 1.3: CALCULATE CENTROIDS
# ============================================================================
print("\n[3/4] Calculating county centroids...")

# Calculate centroids from geometries
counties_gdf['centroid'] = counties_gdf.geometry.centroid
counties_gdf['Latitude'] = counties_gdf['centroid'].y
counties_gdf['Longitude'] = counties_gdf['centroid'].x

# Clean county names (remove ' County' suffix if present)
if 'NAME' in counties_gdf.columns:
    counties_gdf['County_Clean'] = counties_gdf['NAME'].str.upper().str.replace(' COUNTY', '').str.strip()
elif 'NAMELSAD' in counties_gdf.columns:
    counties_gdf['County_Clean'] = counties_gdf['NAMELSAD'].str.upper().str.replace(' COUNTY', '').str.strip()

# Get state names (convert FIPS to state names if needed)
state_fips_to_name = {
    '01': 'ALABAMA', '02': 'ALASKA', '04': 'ARIZONA', '05': 'ARKANSAS', '06': 'CALIFORNIA',
    '08': 'COLORADO', '09': 'CONNECTICUT', '10': 'DELAWARE', '11': 'DISTRICT OF COLUMBIA',
    '12': 'FLORIDA', '13': 'GEORGIA', '15': 'HAWAII', '16': 'IDAHO', '17': 'ILLINOIS',
    '18': 'INDIANA', '19': 'IOWA', '20': 'KANSAS', '21': 'KENTUCKY', '22': 'LOUISIANA',
    '23': 'MAINE', '24': 'MARYLAND', '25': 'MASSACHUSETTS', '26': 'MICHIGAN', '27': 'MINNESOTA',
    '28': 'MISSISSIPPI', '29': 'MISSOURI', '30': 'MONTANA', '31': 'NEBRASKA', '32': 'NEVADA',
    '33': 'NEW HAMPSHIRE', '34': 'NEW JERSEY', '35': 'NEW MEXICO', '36': 'NEW YORK',
    '37': 'NORTH CAROLINA', '38': 'NORTH DAKOTA', '39': 'OHIO', '40': 'OKLAHOMA', '41': 'OREGON',
    '42': 'PENNSYLVANIA', '44': 'RHODE ISLAND', '45': 'SOUTH CAROLINA', '46': 'SOUTH DAKOTA',
    '47': 'TENNESSEE', '48': 'TEXAS', '49': 'UTAH', '50': 'VERMONT', '51': 'VIRGINIA',
    '53': 'WASHINGTON', '54': 'WEST VIRGINIA', '55': 'WISCONSIN', '56': 'WYOMING'
}

if 'STATEFP' in counties_gdf.columns:
    counties_gdf['State_Name'] = counties_gdf['STATEFP'].map(state_fips_to_name)
elif 'STATE_NAME' in counties_gdf.columns:
    counties_gdf['State_Name'] = counties_gdf['STATE_NAME'].str.upper()

print(f"  ✓ Calculated centroids for {len(counties_gdf):,} counties")

# ============================================================================
# STEP 1.4: MATCH WITH CORN DATA COUNTIES
# ============================================================================
print("\n[4/4] Matching with corn data counties...")

# Prepare merge keys
unique_counties['County_Clean'] = unique_counties['County'].astype(str).str.upper().str.strip()
unique_counties['State_Clean'] = unique_counties['State'].astype(str).str.upper().str.strip()

# Try to match by name
matched_counties = unique_counties.merge(
    counties_gdf[['State_Name', 'County_Clean', 'Latitude', 'Longitude']],
    left_on=['State_Clean', 'County_Clean'],
    right_on=['State_Name', 'County_Clean'],
    how='left'
)

# Count matches
matched_count = matched_counties['Latitude'].notna().sum()
print(f"  ✓ Matched {matched_count}/{len(unique_counties)} counties ({matched_count/len(unique_counties)*100:.1f}%)")

if matched_count < len(unique_counties) * 0.8:
    print(f"  ⚠ Warning: Only {matched_count/len(unique_counties)*100:.1f}% matched!")
    print(f"  Some counties may have name mismatches")
# ============================================================================
# DEBUG: IDENTIFY MISMATCHES
# ============================================================================
print("\n[DEBUG] Analyzing Mismatches:")
missing = matched_counties[matched_counties['Latitude'].isna()].copy()

if not missing.empty:
    print(f"  Found {len(missing)} mismatches. Here are the first 20:")
    # Sort by state for easier reading
    missing = missing.sort_values(['State_Clean', 'County_Clean'])
    
    for idx, row in missing[['State_Clean', 'County_Clean']].head(20).iterrows():
        print(f"  ❌ {row['State_Clean']} | {row['County_Clean']}")
        
    # Check for common "St." vs "Saint" or "De" issues
    print("\n  [Diagnostic Check]")
    print("  Checking for common spelling variations in the Census data...")
    
    # Get a list of all available counties in the Census data for reference
    census_refs = counties_gdf[['State_Name', 'County_Clean']].drop_duplicates()
    
    for idx, row in missing.head(5).iterrows():
        state = row['State_Clean']
        county = row['County_Clean']
        
        # Find potential matches in that state
        possible_matches = census_refs[
            (census_refs['State_Name'] == state) & 
            (census_refs['County_Clean'].str.contains(county[:4], na=False)) # Match first 4 chars
        ]['County_Clean'].tolist()
        
        if possible_matches:
            print(f"  For '{county}' ({state}), did you mean: {possible_matches}?")
            
    print("\n  ⚠ STOPPING SCRIPT so you can fix these names in a mapping dictionary.")
    exit() # Stop here so we don't save bad data

# ============================================================================
# MANUAL FALLBACK FOR MISSING COUNTIES
# ============================================================================
missing_counties = matched_counties[matched_counties['Latitude'].isna()]
if len(missing_counties) > 0:
    print(f"\n  ℹ {len(missing_counties)} counties need manual lookup")
    print(f"  Using state centroids as approximation for missing counties...")
    
    # State centroids (approximate)
    state_centroids = {
        'ALABAMA': (32.7, -86.8), 'ARKANSAS': (34.8, -92.2), 'CALIFORNIA': (36.7, -119.4),
        'COLORADO': (39.0, -105.5), 'DELAWARE': (38.9, -75.5), 'GEORGIA': (32.6, -83.4),
        'IDAHO': (44.0, -114.7), 'ILLINOIS': (40.0, -89.0), 'INDIANA': (40.0, -86.1),
        'IOWA': (42.0, -93.5), 'KANSAS': (38.5, -98.0), 'KENTUCKY': (37.5, -85.3),
        'LOUISIANA': (31.0, -92.0), 'MARYLAND': (39.0, -76.8), 'MICHIGAN': (44.3, -85.6),
        'MINNESOTA': (46.0, -94.0), 'MISSISSIPPI': (32.7, -89.7), 'MISSOURI': (38.5, -92.5),
        'NEBRASKA': (41.5, -99.8), 'NEW YORK': (43.0, -75.5), 'NORTH CAROLINA': (35.5, -79.0),
        'NORTH DAKOTA': (47.5, -100.5), 'OHIO': (40.2, -82.9), 'OKLAHOMA': (35.5, -98.0),
        'PENNSYLVANIA': (41.0, -77.5), 'SOUTH CAROLINA': (34.0, -81.0),
        'SOUTH DAKOTA': (44.5, -100.0), 'TENNESSEE': (35.8, -86.3), 'TEXAS': (31.0, -99.0),
        'VIRGINIA': (37.5, -78.8), 'WASHINGTON': (47.5, -120.5), 'WISCONSIN': (44.5, -89.5)
    }
    
    for idx in missing_counties.index:
        state = matched_counties.loc[idx, 'State_Clean']
        if state in state_centroids:
            matched_counties.loc[idx, 'Latitude'] = state_centroids[state][0]
            matched_counties.loc[idx, 'Longitude'] = state_centroids[state][1]

# ============================================================================
# STEP 1.5: SAVE OUTPUT
# ============================================================================
print(f"\n[5/5] Saving county centroids...")

# Prepare final output
output_df = matched_counties[[
    'State', 'County', 'State ANSI', 'County ANSI', 'Latitude', 'Longitude'
]].copy()

# Remove any remaining NaN
output_df = output_df.dropna(subset=['Latitude', 'Longitude'])

# Validate coordinates
valid_coords = (
    output_df['Latitude'].between(24, 50) & 
    output_df['Longitude'].between(-125, -65)
)
if not valid_coords.all():
    print(f"  ⚠ Warning: {(~valid_coords).sum()} counties have invalid coordinates")
    output_df = output_df[valid_coords]

# Save
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"  ✓ Saved: {OUTPUT_FILE}")
print(f"  ✓ Counties with valid coordinates: {len(output_df):,}")

# ============================================================================
# VALIDATION & SUMMARY
# ============================================================================
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
print(f"Input counties: {len(unique_counties):,}")
print(f"Output counties: {len(output_df):,}")
print(f"Coverage: {len(output_df)/len(unique_counties)*100:.1f}%")
print(f"\nCoordinate ranges:")
print(f"  Latitude:  {output_df['Latitude'].min():.2f} to {output_df['Latitude'].max():.2f}")
print(f"  Longitude: {output_df['Longitude'].min():.2f} to {output_df['Longitude'].max():.2f}")
print(f"\nSample counties:")
print(output_df.head(5).to_string(index=False))

print("\n" + "="*70)
print("✓ STEP 1 COMPLETE!")
print("="*70)
print(f"\nNext step: Run download_daymet_weather.py")
print("="*70 + "\n")

