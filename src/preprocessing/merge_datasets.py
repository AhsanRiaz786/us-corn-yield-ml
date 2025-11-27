"""
Merge USDA NASS Corn County Data
Combines: Yield, Area Planted, Area Harvested, and Production
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("USDA NASS CORN DATA MERGER")
print("="*70)

# ============================================================================
# STEP 1: LOAD ALL DATASETS
# ============================================================================
print("\n[1/7] Loading datasets...")

try:
    yield_df = pd.read_csv('us_corn_yield_county_all_years.csv')
    print(f"  ✓ Yield: {len(yield_df):,} rows")
except FileNotFoundError:
    print("  ✗ ERROR: us_corn_yield_county_all_years.csv not found!")
    exit(1)

try:
    planted_df = pd.read_csv('us_corn_area_planted_county_all_years.csv')
    print(f"  ✓ Area Planted: {len(planted_df):,} rows")
except FileNotFoundError:
    print("  ✗ ERROR: us_corn_area_planted_county_all_years.csv not found!")
    exit(1)

try:
    harvested_df = pd.read_csv('us_corn_area_harvested_county_all_years.csv')
    print(f"  ✓ Area Harvested: {len(harvested_df):,} rows")
except FileNotFoundError:
    print("  ✗ ERROR: us_corn_area_harvested_county_all_years.csv not found!")
    exit(1)

try:
    production_df = pd.read_csv('us_corn_production_county_all_years.csv')
    print(f"  ✓ Production: {len(production_df):,} rows")
except FileNotFoundError:
    print("  ✗ ERROR: us_corn_production_county_all_years.csv not found!")
    exit(1)

# ============================================================================
# STEP 2: CLEAN AND STANDARDIZE DATA
# ============================================================================
print("\n[2/7] Cleaning data...")

def clean_dataframe(df, value_name):
    """Clean and prepare a dataframe for merging"""
    df = df.copy()
    
    # Convert Value column from string to numeric (remove commas)
    df['Value'] = df['Value'].astype(str).str.replace(',', '')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # Create unique key: State_County_Year
    df['County_Clean'] = df['County'].astype(str).str.strip()
    df['State_Clean'] = df['State'].astype(str).str.strip().str.upper()
    df['Year_Clean'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Create merge key
    df['merge_key'] = (df['State_Clean'] + '_' + 
                       df['County_Clean'] + '_' + 
                       df['Year_Clean'].astype(str))
    
    # Select and rename columns
    result = df[['merge_key', 'Year_Clean', 'State_Clean', 'State ANSI', 
                 'County_Clean', 'County ANSI', 'Ag District', 'Value']].copy()
    
    result = result.rename(columns={
        'Year_Clean': 'Year',
        'State_Clean': 'State',
        'County_Clean': 'County',
        'Value': value_name
    })
    
    # Remove duplicates (keep first occurrence)
    result = result.drop_duplicates(subset=['merge_key'], keep='first')
    
    return result

yield_clean = clean_dataframe(yield_df, 'Yield_BU_ACRE')
planted_clean = clean_dataframe(planted_df, 'Area_Planted_ACRES')
harvested_clean = clean_dataframe(harvested_df, 'Area_Harvested_ACRES')
production_clean = clean_dataframe(production_df, 'Production_BU')

print(f"  ✓ Yield cleaned: {len(yield_clean):,} unique records")
print(f"  ✓ Area Planted cleaned: {len(planted_clean):,} unique records")
print(f"  ✓ Area Harvested cleaned: {len(harvested_clean):,} unique records")
print(f"  ✓ Production cleaned: {len(production_clean):,} unique records")

# ============================================================================
# STEP 3: MERGE ALL DATASETS
# ============================================================================
print("\n[3/7] Merging datasets...")

# Start with yield (most complete dataset typically)
master = yield_clean.copy()

# Merge with planted
master = master.merge(
    planted_clean[['merge_key', 'Area_Planted_ACRES']], 
    on='merge_key', 
    how='outer'
)

# Merge with harvested
master = master.merge(
    harvested_clean[['merge_key', 'Area_Harvested_ACRES']], 
    on='merge_key', 
    how='outer'
)

# Merge with production
master = master.merge(
    production_clean[['merge_key', 'Production_BU']], 
    on='merge_key', 
    how='outer'
)

# Fill in missing State/County/Year info from other datasets
for col in ['Year', 'State', 'County', 'State ANSI', 'County ANSI', 'Ag District']:
    if master[col].isna().any():
        # Try to fill from other dataframes
        for df in [planted_clean, harvested_clean, production_clean]:
            if col in df.columns:
                fill_data = df.set_index('merge_key')[col]
                master[col] = master[col].fillna(master['merge_key'].map(fill_data))

print(f"  ✓ Merged dataset: {len(master):,} total records")

# ============================================================================
# STEP 4: CALCULATE DERIVED FEATURES
# ============================================================================
print("\n[4/7] Calculating derived features...")

# Calculate Production_Calculated (should match Production_BU if data is consistent)
master['Production_Calculated'] = (master['Yield_BU_ACRE'] * 
                                   master['Area_Harvested_ACRES'])

# Abandonment Rate (key stress indicator!)
master['Abandonment_Rate'] = (
    (master['Area_Planted_ACRES'] - master['Area_Harvested_ACRES']) / 
    master['Area_Planted_ACRES']
).clip(lower=0, upper=1)  # Constraint: between 0 and 1

# Harvest Efficiency (percentage of planted area that was harvested)
master['Harvest_Efficiency'] = (
    master['Area_Harvested_ACRES'] / master['Area_Planted_ACRES']
).clip(upper=1)  # Can't harvest more than planted

# Data completeness flags
master['Has_Yield'] = master['Yield_BU_ACRE'].notna()
master['Has_Planted'] = master['Area_Planted_ACRES'].notna()
master['Has_Harvested'] = master['Area_Harvested_ACRES'].notna()
master['Has_Production'] = master['Production_BU'].notna()
master['Complete_Record'] = (master['Has_Yield'] & 
                             master['Has_Planted'] & 
                             master['Has_Harvested'] & 
                             master['Has_Production'])

print(f"  ✓ Derived features calculated")

# ============================================================================
# STEP 5: DATA VALIDATION
# ============================================================================
print("\n[5/7] Validating data quality...")

# Check Production consistency
master['Production_Error'] = np.abs(
    master['Production_BU'] - master['Production_Calculated']
)
master['Production_Error_Pct'] = (
    master['Production_Error'] / master['Production_BU'].replace(0, np.nan) * 100
)

# Identify records with large discrepancies (>10% error)
large_errors = master[
    (master['Production_Error_Pct'] > 10) & 
    master['Complete_Record']
]

print(f"  Data Quality Checks:")
print(f"    Records with all 4 metrics: {master['Complete_Record'].sum():,}")
print(f"    Records with >10% production error: {len(large_errors):,}")

if len(large_errors) > 0:
    print(f"    WARNING: {len(large_errors)} records have inconsistent production values!")

# ============================================================================
# STEP 6: CREATE SUMMARY STATISTICS
# ============================================================================
print("\n[6/7] Generating summary statistics...")

print(f"\n{'='*70}")
print("DATASET SUMMARY")
print(f"{'='*70}")
print(f"Total records: {len(master):,}")
print(f"Complete records (all 4 metrics): {master['Complete_Record'].sum():,}")
print(f"Year range: {int(master['Year'].min())} - {int(master['Year'].max())}")
print(f"Number of states: {master['State'].nunique()}")
print(f"Number of unique counties: {master.groupby(['State', 'County']).ngroups}")

print(f"\nData Availability:")
print(f"  Records with Yield: {master['Has_Yield'].sum():,} ({master['Has_Yield'].mean()*100:.1f}%)")
print(f"  Records with Area Planted: {master['Has_Planted'].sum():,} ({master['Has_Planted'].mean()*100:.1f}%)")
print(f"  Records with Area Harvested: {master['Has_Harvested'].sum():,} ({master['Has_Harvested'].mean()*100:.1f}%)")
print(f"  Records with Production: {master['Has_Production'].sum():,} ({master['Has_Production'].mean()*100:.1f}%)")

# Summary statistics for complete records only
complete_df = master[master['Complete_Record']].copy()

if len(complete_df) > 0:
    print(f"\n{'='*70}")
    print("STATISTICS FOR COMPLETE RECORDS")
    print(f"{'='*70}")
    
    stats_df = complete_df[['Yield_BU_ACRE', 'Area_Planted_ACRES', 
                             'Area_Harvested_ACRES', 'Production_BU', 
                             'Abandonment_Rate']].describe()
    print(stats_df)
    
    # Top producing states
    print(f"\n{'='*70}")
    print("TOP 10 CORN PRODUCING STATES (by average annual production)")
    print(f"{'='*70}")
    state_prod = complete_df.groupby('State')['Production_BU'].mean().sort_values(ascending=False)
    for i, (state, prod) in enumerate(state_prod.head(10).items(), 1):
        print(f"  {i:2d}. {state:20s} {prod:>15,.0f} BU")
    
    # Abandonment rate analysis
    print(f"\n{'='*70}")
    print("ABANDONMENT RATE ANALYSIS")
    print(f"{'='*70}")
    print(f"  Mean abandonment rate: {complete_df['Abandonment_Rate'].mean()*100:.2f}%")
    print(f"  Median abandonment rate: {complete_df['Abandonment_Rate'].median()*100:.2f}%")
    print(f"  Records with >10% abandonment: {(complete_df['Abandonment_Rate'] > 0.1).sum():,}")
    print(f"  Records with >20% abandonment: {(complete_df['Abandonment_Rate'] > 0.2).sum():,}")
    
    # Worst abandonment years (stress years)
    print(f"\n{'='*70}")
    print("TOP 10 WORST ABANDONMENT YEARS (drought/stress indicators)")
    print(f"{'='*70}")
    yearly_abandon = complete_df.groupby('Year')['Abandonment_Rate'].mean().sort_values(ascending=False)
    for i, (year, rate) in enumerate(yearly_abandon.head(10).items(), 1):
        print(f"  {i:2d}. {int(year)} - {rate*100:.2f}%")

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================
print(f"\n[7/7] Saving datasets...")

# Save full master dataset
master_output = master[[
    'Year', 'State', 'State ANSI', 'County', 'County ANSI', 'Ag District',
    'Yield_BU_ACRE', 'Area_Planted_ACRES', 'Area_Harvested_ACRES', 
    'Production_BU', 'Production_Calculated', 'Production_Error',
    'Abandonment_Rate', 'Harvest_Efficiency',
    'Has_Yield', 'Has_Planted', 'Has_Harvested', 'Has_Production', 'Complete_Record'
]].copy()

master_output = master_output.sort_values(['State', 'County', 'Year'])
master_output.to_csv('master_corn_county_data.csv', index=False)
print(f"  ✓ Full dataset saved: master_corn_county_data.csv")
print(f"    Records: {len(master_output):,}")

# Save modeling dataset (complete records only)
modeling_df = complete_df[[
    'Year', 'State', 'State ANSI', 'County', 'County ANSI', 'Ag District',
    'Yield_BU_ACRE', 'Area_Planted_ACRES', 'Area_Harvested_ACRES', 
    'Production_BU', 'Abandonment_Rate', 'Harvest_Efficiency'
]].copy()

modeling_df = modeling_df.sort_values(['State', 'County', 'Year'])
modeling_df.to_csv('modeling_dataset_complete.csv', index=False)
print(f"  ✓ Modeling dataset (complete records): modeling_dataset_complete.csv")
print(f"    Records: {len(modeling_df):,}")

# Save summary report
with open('data_merge_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("USDA NASS CORN DATA MERGE REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total records in master dataset: {len(master):,}\n")
    f.write(f"Complete records (all 4 metrics): {master['Complete_Record'].sum():,}\n")
    f.write(f"Year range: {int(master['Year'].min())} - {int(master['Year'].max())}\n")
    f.write(f"Number of states: {master['State'].nunique()}\n")
    f.write(f"Number of unique counties: {master.groupby(['State', 'County']).ngroups}\n")
    f.write(f"\nFiles created:\n")
    f.write(f"  - master_corn_county_data.csv (all records)\n")
    f.write(f"  - modeling_dataset_complete.csv (complete records only)\n")
    f.write(f"  - data_merge_report.txt (this file)\n")

print(f"  ✓ Summary report saved: data_merge_report.txt")

print(f"\n{'='*70}")
print("✓ MERGE COMPLETE!")
print(f"{'='*70}")
print(f"\nNext Steps:")
print(f"  1. Open modeling_dataset_complete.csv to start analysis")
print(f"  2. Identify your modeling timeframe (e.g., 1980-2024)")
print(f"  3. Begin collecting weather data for those years")
print(f"{'='*70}\n")