# Feature Engineering Documentation

This document describes all features created for the corn yield prediction models, including their rationale, computation methods, and importance.

## Overview

The final modeling dataset contains 50+ features across five categories:
1. **Historical Features** (8 features): Past yield patterns
2. **Weather Features** (33 features): Growing season climate
3. **Soil Features** (4 features): Physical soil properties
4. **Area Features** (3 features): Planting and harvesting metrics
5. **Temporal Features** (2 features): Year and geographic identifiers

## 1. Historical Features

Historical yield patterns are the strongest predictors, capturing technological trends, farm management practices, and location-specific productivity.

### Lag Features

| Feature | Description | Computation | Importance |
|---------|-------------|-------------|------------|
| `Yield_lag1` | Previous year's yield | `shift(1)` | 32.1% |
| `Yield_lag2` | Two years prior yield | `shift(2)` | 10.8% |
| `Yield_lag3` | Three years prior yield | `shift(3)` | 10.2% |

**Rationale:** Yield exhibits strong autocorrelation due to persistent factors (soil quality, infrastructure, expertise).

**Implementation:**
```python
df = df.sort_values(['State_FIPS', 'County_FIPS', 'Year'])
df['Yield_lag1'] = df.groupby(['State_FIPS', 'County_FIPS'])['Yield'].shift(1)
df['Yield_lag2'] = df.groupby(['State_FIPS', 'County_FIPS'])['Yield'].shift(2)
df['Yield_lag3'] = df.groupby(['State_FIPS', 'County_FIPS'])['Yield'].shift(3)
```

### Rolling Statistics

| Feature | Description | Computation |
|---------|-------------|-------------|
| `Yield_roll3` | 3-year rolling mean | `rolling(3).mean()` |
| `Yield_roll5` | 5-year rolling mean | `rolling(5).mean()` |

**Rationale:** Smooths inter-annual variability, captures medium-term trends.

### Year-over-Year Changes

| Feature | Description | Computation |
|---------|-------------|-------------|
| `Yield_change1` | Annual yield change | `Yield - Yield_lag1` |
| `Yield_pct_change1` | Percentage change | `(Yield - Yield_lag1) / Yield_lag1` |

**Rationale:** Captures momentum and recovery from poor years.

### Trend Features

| Feature | Description | Computation |
|---------|-------------|-------------|
| `Years_since_1981` | Temporal trend proxy | `Year - 1981` |

**Rationale:** Captures technological improvement over time (genetic advances, precision agriculture).

## 2. Weather Features

Weather during the growing season (April-September) critically determines corn yield through water availability, temperature stress, and developmental timing.

### Growing Degree Days (GDD)

**Definition:** Accumulated heat units above base temperature required for crop development.

| Feature | Description | Formula |
|---------|-------------|---------|
| `GDD_total` | Season-total GDD | `Σ max(0, (Tmax + Tmin)/2 - 10°C)` |
| `GDD_early` | April-May GDD | Same, early season |
| `GDD_mid` | June-July GDD | Same, mid season |
| `GDD_late` | August-September GDD | Same, late season |

**Rationale:** Corn requires ~2,700-3,000 GDD from planting to maturity. Timing of accumulation affects yield potential.

**Implementation:**
```python
def calculate_gdd(tmax, tmin, base_temp=10):
    """Calculate Growing Degree Days."""
    tavg = (tmax + tmin) / 2
    return max(0, tavg - base_temp)

df['GDD_daily'] = df.apply(lambda x: calculate_gdd(x['T2M_MAX'], x['T2M_MIN']), axis=1)
df['GDD_total'] = df.groupby(['State_FIPS', 'County_FIPS', 'Year'])['GDD_daily'].sum()
```

### Heat Stress Indicators

**Definition:** Days or cumulative exposure exceeding temperature thresholds harmful to corn.

| Feature | Description | Threshold | Impact |
|---------|-------------|-----------|--------|
| `Heat_stress_days` | Days with Tmax > 35°C | 35°C | Pollen viability loss |
| `Extreme_heat_days` | Days with Tmax > 38°C | 38°C | Photosynthesis inhibition |
| `Hot_nights` | Nights with Tmin > 24°C | 24°C | Respiration losses |

**Rationale:** High temperatures during flowering (July) drastically reduce kernel set and yield.

**Feature Importance:** Heat stress days rank 4th overall (4.7% importance).

### Temperature Statistics

| Feature | Description | Aggregation |
|---------|-------------|-------------|
| `Tmax_mean` | Mean maximum temperature | Growing season average |
| `Tmax_max` | Peak temperature | Growing season maximum |
| `Tmin_mean` | Mean minimum temperature | Growing season average |
| `Tmin_min` | Coldest night | Growing season minimum |
| `Tavg_mean` | Mean daily temperature | Growing season average |

**Rationale:** Overall thermal environment affects growth rate and yield potential.

### Precipitation Features

**Absolute Metrics:**

| Feature | Description | Unit |
|---------|-------------|------|
| `Precip_total` | Total growing season rainfall | mm |
| `Precip_early` | April-May rainfall | mm |
| `Precip_mid` | June-July rainfall (critical) | mm |
| `Precip_late` | August-September rainfall | mm |

**Anomaly Metrics:**

| Feature | Description | Computation |
|---------|-------------|-------------|
| `Precip_anomaly` | Deviation from county mean | `Precip_total - county_mean` |
| `Precip_mid_anomaly` | Mid-season deviation | `Precip_mid - county_mean_mid` |

**Rationale:** Corn water demand peaks during flowering and grain fill (June-August). Anomalies matter more than absolute values due to local adaptation.

**Feature Importance:** Precipitation anomaly ranks 7th (2.9%).

### Drought Indicators

| Feature | Description | Threshold |
|---------|-------------|-----------|
| `Dry_days` | Days with precipitation < 1mm | <1mm |
| `Wet_days` | Days with precipitation > 10mm | >10mm |
| `Max_dry_spell` | Longest consecutive dry period | days |

**Rationale:** Duration of water stress periods affects yield more than total rainfall.

### Humidity and Evapotranspiration Proxy

| Feature | Description | Unit |
|---------|-------------|------|
| `RH_mean` | Mean relative humidity | % |
| `RH_min` | Minimum relative humidity | % |
| `VPD_approx` | Vapor pressure deficit estimate | kPa |

**Rationale:** Low humidity increases evaporative demand and water stress.

### Temperature-Precipitation Interactions

| Feature | Description | Computation |
|---------|-------------|-------------|
| `Heat_drought_index` | Combined stress | `(Heat_stress_days × 10) / Precip_mid` |
| `GDD_per_mm` | Growth efficiency | `GDD_total / Precip_total` |

**Rationale:** Simultaneous heat and drought stress have multiplicative negative effects.

## 3. Soil Features

Soil properties determine water holding capacity, nutrient availability, and root zone conditions.

### Available Water Capacity (AWC)

**Definition:** Volume of water available to plants between field capacity and permanent wilting point.

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `AWC_avg` | County average AWC | cm/cm | 0.05-0.25 |

**Rationale:** Higher AWC buffers against drought stress. Critical in rain-fed production.

**Feature Importance:** 1.6% (10th overall)

### Clay Content

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `Clay_avg` | County average clay % | % | 5-60% |

**Rationale:** 
- Moderate clay (20-35%): Optimal water retention
- High clay (>40%): Poor drainage, compaction risk
- Low clay (<15%): Low water retention

### Soil pH

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `pH_avg` | County average pH | pH units | 4.5-8.5 |

**Rationale:** Corn optimal pH is 6.0-7.0. Affects nutrient availability (P, micronutrients).

### Organic Matter

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `OM_avg` | County average organic matter % | % | 0.5-10% |

**Rationale:** Improves water retention, nutrient supply, soil structure. Higher in historically productive regions.

### Soil Feature Limitations

- **Static values:** No temporal variation modeled
- **Aggregation:** County-level masks within-county heterogeneity
- **Survey timing:** Data from different years (1980s-2020s)

## 4. Area Features

Planting and harvesting metrics reflect farmer expectations and realized conditions.

### Core Metrics

| Feature | Description | Unit |
|---------|-------------|------|
| `Area_planted` | Total acres planted to corn | ACRES |
| `Area_harvested` | Total acres harvested | ACRES |
| `Production` | Total production | BU |

### Derived Features

| Feature | Description | Computation | Interpretation |
|---------|-------------|-------------|----------------|
| `Abandonment_rate` | Unharvested area fraction | `1 - (Area_harvested / Area_planted)` | Weather damage, profitability |
| `Production_per_acre` | Redundant with yield | `Production / Area_harvested` | Should equal Yield |

**Abandonment Rate Rationale:** High abandonment indicates severe stress (drought, flood, hail) or economic factors. Strong negative correlation with yield.

**Feature Importance:** Abandonment rate ranks 9th (1.8%).

## 5. Temporal and Geographic Features

### Year

| Feature | Description | Use |
|---------|-------------|-----|
| `Year` | Calendar year | Captures technology trends |

**Rationale:** Yield has increased ~1.8 BU/ACRE/year since 1980 due to genetic improvement and management advances.

### State Identifier

| Feature | Description | Use |
|---------|-------------|-----|
| `State` | State name or FIPS | Regional fixed effects |

**Rationale:** Captures unmeasured regional factors (soil types, climate zones, extension services).

## Feature Selection and Preprocessing

### Correlation Analysis

High-correlation pairs (r > 0.9) examined for redundancy:
- `Production` and `Yield`: Perfectly correlated (Production = Yield × Area)
- `Tavg_mean` and `GDD_total`: High correlation (r = 0.85)
- `Precip_total` and `Wet_days`: Moderate correlation (r = 0.72)

**Decision:** Keep both members if they capture different aspects (e.g., total vs distribution).

### Handling Missing Values

**Strategy:**
- Historical lags: Missing for first 1-3 years per county (drop these rows)
- Weather data: Interpolate short gaps (<3 days), drop county-years with >10% missing
- Soil data: County mean imputation or drop if critical feature

**Final dataset:** 82,436 complete records after missing value treatment.

### Feature Scaling

**Method:** StandardScaler (mean=0, std=1)

**Applied to:** All features except binary indicators

**Rationale:** 
- Tree-based models: Scaling not required but doesn't hurt
- Linear models: Essential for regularization to work properly
- Improves convergence for gradient-based optimization

## Feature Importance Results

**XGBoost Model (Best Performer):**

Rank | Feature | Importance (%) | Category |
-----|---------|----------------|----------|
1 | Yield_lag1 | 32.1 | Historical |
2 | Yield_lag2 | 10.8 | Historical |
3 | Yield_lag3 | 10.2 | Historical |
4 | Heat_stress_days | 4.7 | Weather |
5 | Tmax_max | 3.8 | Weather |
6 | GDD_total | 3.2 | Weather |
7 | Precip_anomaly | 2.9 | Weather |
8 | Area_planted | 2.1 | Area |
9 | Abandonment_rate | 1.8 | Area |
10 | AWC_avg | 1.6 | Soil |

**Key Insights:**
- Historical features dominate (53.1% combined)
- Weather features are second most important (20-25%)
- Soil features have modest but consistent contribution (5-7%)
- Area features capture stress signals (3.9%)

## Feature Engineering Pipeline

**Script:** `src/features/engineer_weather_features.py`

**Workflow:**
1. Load raw daily/weekly weather data
2. Define growing season window (April 1 - September 30)
3. Calculate GDD for each day
4. Count stress days (heat, drought)
5. Aggregate to seasonal totals and means
6. Compute anomalies relative to county historical means
7. Create interaction terms
8. Output county-year feature matrix

**Runtime:** ~5 minutes for 120,000 county-year-week records

## Future Feature Engineering

**Potential Additions:**
- **Satellite imagery:** NDVI, LAI, soil moisture
- **Economic factors:** Fertilizer prices, crop insurance enrollment
- **Management practices:** Tillage, cover crops (if data available)
- **Pest/disease:** Occurrence data for major corn pests
- **Soil moisture:** From models or satellite (SMAP, SMOS)
- **Irrigation:** Flag for irrigated vs rain-fed counties

## References

1. Nielsen, R.L. (2020). Heat Stress and Corn Production. Purdue Extension.
2. Shaw, R.H. (1988). Climate requirement. In Corn and Corn Improvement.
3. USDA NRCS. (2024). Soil Health and Water-Holding Capacity.
4. Lobell, D.B. et al. (2014). Greater sensitivity to drought accompanies maize yield increase in the U.S. Midwest. *Science*, 344(6183), 516-519.

## Code Example

See `notebooks/02_feature_engineering.ipynb` for detailed walkthrough with visualizations.

**Quick reference:**
```python
from src.features.engineer_weather_features import create_weather_features

# Generate all weather features
weather_features = create_weather_features(
    weather_data_path="data/raw/weather_data_weekly.csv",
    output_path="data/processed/weather_features_county_year.csv"
)
```

