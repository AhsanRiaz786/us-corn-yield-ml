import pandas as pd
import numpy as np

print("="*70)
print("CLEANING MODELING SET")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = 'modeling_dataset_complete.csv' 
OUTPUT_FILE = 'modeling_dataset_complete.csv'

# ============================================================================
# STEP 1: LOAD MODELING SET
# ============================================================================
print(f"\n[1/3] Loading modeling set ({INPUT_FILE})...")
try:
    modeling_set = pd.read_csv(INPUT_FILE)
    initial_count = len(modeling_set)
    print(f"  ✓ Loaded {initial_count:,} rows")
except FileNotFoundError:
    print(f"  ✗ ERROR: {INPUT_FILE} not found.")
    print("  Please check the filename or path.")
    exit(1)

# ============================================================================
# STEP 2: REMOVE "OTHER/COMBINED" COUNTIES
# ============================================================================
print("\n[2/3] Removing invalid counties...")

# Standardize to uppercase for filtering
modeling_set['County'] = modeling_set['County'].astype(str).str.upper().str.strip()
modeling_set['State'] = modeling_set['State'].astype(str).str.upper().str.strip()

# Filter out rows containing "OTHER" or "COMBINED"
valid_rows = ~modeling_set['County'].str.contains('OTHER', case=False) & \
             ~modeling_set['County'].str.contains('COMBINED', case=False)

modeling_set_clean = modeling_set[valid_rows].copy()
dropped_count = initial_count - len(modeling_set_clean)

print(f"  ✓ Dropped {dropped_count:,} rows (Other/Combined counties)")

# ============================================================================
# STEP 3: FIX COUNTY NAMES (FINAL MAPPING)
# ============================================================================
print("\n[3/3] Standardizing county names...")

def fix_county_names(row):
    state = row['State']
    county = row['County']
    
    # --- 1. STATE-SPECIFIC FIXES ---
    
    # Virginia Independent Cities (e.g., "Chesapeake City" -> "Chesapeake")
    if state == 'VIRGINIA' and county.endswith(' CITY'):
        return county.replace(' CITY', '')

    # Indiana uses 2 L's ("Vermillion"), Illinois uses 1 L ("Vermilion")
    if state == 'INDIANA' and county == 'VERMILION':
        return 'VERMILLION'
        
    # Illinois Specifics (Census spelling vs USDA spelling)
    if state == 'ILLINOIS':
        if county == 'DEWITT': return 'DE WITT'   # Needs space
        if county == 'LA SALLE': return 'LASALLE' # No space
        if county == 'JO DAVIESS': return 'JO DAVIESS'

    # Louisiana Specifics
    if state == 'LOUISIANA':
        if county == 'DESOTO': return 'DE SOTO'   # Needs space
        if county == 'LA SALLE': return 'LASALLE' # No space

    # Wisconsin Specifics
    if state == 'WISCONSIN':
        if county == 'LACROSSE': return 'LA CROSSE' # Needs space
        if county == 'FON DU LAC': return 'FOND DU LAC'
    if state == 'VIRGINIA':
        if county == 'CHARLES': return 'CHARLES CITY'
        if county == 'JAMES': return 'JAMES CITY'

    # Oklahoma/Mississippi Le Flore
    if state == 'OKLAHOMA' and county == 'LEFLORE': return 'LE FLORE'
    
    # New Mexico
    if state == 'NEW MEXICO' and 'DONA ANA' in county: return 'DOÑA ANA'
    
    # South Dakota Historical Mergers
    if state == 'SOUTH DAKOTA':
        if county in ['WASHABAUGH', 'WASHINGTON']: return 'JACKSON'
        if county == 'SHANNON': return 'OGLALA LAKOTA'

    # --- 2. GLOBAL TEXT FIXES ---
    
    # Standardize Saint -> St.
    if county.startswith('SAINT '):
        county = county.replace('SAINT ', 'ST. ')
    elif county.startswith('ST ') and not county.startswith('ST. '):
        county = county.replace('ST ', 'ST. ')

    # Global Corrections Dictionary
    corrections = {
        'DE KALB': 'DEKALB',          # Most states use combined
        'DU PAGE': 'DUPAGE',
        'LA PORTE': 'LAPORTE',
        'LA MOURE': 'LAMOURE',
        'LAPAZ': 'LA PAZ',
        'O BRIEN': "O'BRIEN",
        'OBRIEN': "O'BRIEN",
        'ST MARYS': "ST. MARY'S",     # Handle missing apostrophe AND dot
        'ST. MARYS': "ST. MARY'S",    # Handle missing apostrophe
        'QUEEN ANNES': "QUEEN ANNE'S",
        'PRINCE GEORGES': "PRINCE GEORGE'S",
        'STE GENEVIEVE': 'STE. GENEVIEVE',
        'DE WITT': 'DE WITT',         # Ensure we don't accidentally compress this
        'DE SOTO': 'DE SOTO'          # Ensure we don't accidentally compress this
    }
    
    if county in corrections:
        return corrections[county]
        
    return county

# Apply the cleaning function row by row
modeling_set_clean['County'] = modeling_set_clean.apply(fix_county_names, axis=1)

print("  ✓ Applied final spelling corrections")

# ============================================================================
# SAVE OUTPUT
# ============================================================================
print("\n" + "="*70)
print("SAVING OUTPUT")
print("="*70)

modeling_set_clean.to_csv(OUTPUT_FILE, index=False)
print(f"  ✓ Saved cleaned file: {OUTPUT_FILE}")
print(f"  ✓ Final Row Count: {len(modeling_set_clean):,}")

print("\nNEXT STEPS:")
print(f"1. Open 'get_county_centroids.py'")
print(f"2. Ensure: INPUT_FILE = '{OUTPUT_FILE}'")
print(f"3. Run 'get_county_centroids.py'")