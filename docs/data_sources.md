# Data Sources Documentation

This document provides detailed information about all data sources used in the US Corn Yield Prediction System.

## Overview

The system integrates data from four primary sources, creating a comprehensive dataset spanning 1981-2023 across 2,635 US counties. The final modeling dataset contains 82,436 county-year observations with 50+ features.

## 1. USDA NASS QuickStats

**Source:** United States Department of Agriculture - National Agricultural Statistics Service  
**URL:** https://quickstats.nass.usda.gov/  
**License:** Public domain (US Government data)  
**Access Method:** HTTP POST requests to internal API endpoints

### Data Retrieved

| Dataset | Description | Unit | Years | Records |
|---------|-------------|------|-------|---------|
| Corn Yield | County-level corn grain yield | BU/ACRE | 1981-2023 | ~80,000 |
| Area Planted | Total area planted to corn | ACRES | 1981-2023 | ~80,000 |
| Area Harvested | Actual harvested corn area | ACRES | 1981-2023 | ~80,000 |
| Production | Total corn grain production | BU | 1981-2023 | ~80,000 |

### Coverage

- **Temporal:** 1981-2023 (43 years)
- **Spatial:** 2,635 counties across major corn-producing states
- **Geographic Scope:** Continental United States, focused on Corn Belt

### Key States Included

Primary corn-producing states (90%+ of data):
- Iowa (IA)
- Illinois (IL)
- Nebraska (NE)
- Minnesota (MN)
- Indiana (IN)
- South Dakota (SD)
- Kansas (KS)
- Ohio (OH)
- Wisconsin (WI)
- Missouri (MO)
- Michigan (MI)
- North Dakota (ND)
- Kentucky (KY)

### Data Quality Notes

- Only published county-level estimates included
- Data suppressed for counties with <3 farms to protect privacy
- Survey-based estimates with associated sampling error
- More complete coverage in recent years (2000+)
- Some counties have gaps in time series

### Collection Method

Custom Python scraper using USDA QuickStats internal API:
1. POST to `/api/get_constraints` for query structure
2. POST to `/uuid/encode` to generate download UUID
3. GET `/data/spreadsheet/{uuid}.csv` for CSV export

## 2. NASA POWER Agroclimatology

**Source:** NASA Prediction of Worldwide Energy Resources  
**URL:** https://power.larc.nasa.gov/  
**License:** CC0 1.0 Universal (Public Domain)  
**Access Method:** RESTful API (JSON)

### Data Retrieved

| Variable | Description | Unit | Temporal Resolution |
|----------|-------------|------|---------------------|
| T2M | Temperature at 2 meters | °C | Daily → Weekly avg |
| T2M_MAX | Maximum temperature | °C | Daily → Weekly max |
| T2M_MIN | Minimum temperature | °C | Daily → Weekly min |
| PRECTOTCORR | Precipitation (corrected) | mm/day | Daily → Weekly sum |
| RH2M | Relative humidity at 2m | % | Daily → Weekly avg |

### Coverage

- **Temporal:** 1981-2023, growing season only (April-September)
- **Spatial:** County centroids (2,635 locations)
- **Resolution:** 0.5° × 0.625° grid cells

### Growing Season Definition

Weather data aggregated for critical growing period:
- **Start:** April 1 (planting preparation)
- **End:** September 30 (harvest completion)
- **Duration:** 26 weeks per year

### Data Processing

Raw daily data aggregated to weekly intervals to reduce dimensionality:
- Temperature: Mean, min, max computed
- Precipitation: Sum computed
- Relative humidity: Mean computed

Final dataset: ~120,000 county-year-week records

### Data Quality Notes

- Satellite-based reanalysis data (not direct measurements)
- Spatial resolution moderate (50-60 km grid cells)
- Point-based (county centroid) rather than area-weighted
- Reliable temporal consistency across full time period
- Validated against ground station data

### Collection Method

Python script querying NASA POWER API:
```python
base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    "start": "YYYYMMDD",
    "end": "YYYYMMDD",
    "latitude": lat,
    "longitude": lon,
    "community": "ag",
    "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M",
    "format": "JSON"
}
```

## 3. USDA NRCS Soil Data Access

**Source:** USDA Natural Resources Conservation Service  
**URL:** https://sdmdataaccess.sc.egov.usda.gov/  
**Database:** gSSURGO (Gridded Soil Survey Geographic Database)  
**License:** Public domain (US Government data)  
**Access Method:** SQL queries via REST API

### Data Retrieved

| Property | Description | Unit | Aggregation |
|----------|-------------|------|-------------|
| AWC | Available water capacity | cm/cm | County average |
| Clay% | Clay content | % | County average |
| pH | Soil pH (1:1 water) | pH units | County average |
| OM% | Organic matter | % | County average |

### Coverage

- **Temporal:** Static properties (2020 snapshot)
- **Spatial:** 2,500+ counties with valid soil data
- **Vertical:** Top 150 cm of soil profile

### Aggregation Method

County-level values computed by:
1. Querying all map units within county boundary
2. Averaging component properties weighted by area
3. Depth-weighting horizons within each component

### Data Quality Notes

- County-level aggregates mask within-county variability
- Based on SSURGO surveys (1:12,000 to 1:63,360 scale)
- Survey dates vary by county (1980s-2020s)
- ~100 counties missing due to data gaps or urban areas
- Static properties do not capture temporal changes

### Collection Method

SQL queries via Soil Data Access API:
```sql
SELECT 
    co.areasymbol,
    AVG(ch.awc_r) as awc_avg,
    AVG(ch.claytotal_r) as clay_avg,
    AVG(ch.ph1to1h2o_r) as ph_avg,
    AVG(ch.om_r) as om_avg
FROM component AS co
INNER JOIN chorizon AS ch ON ch.cokey = co.cokey
WHERE co.areasymbol = 'US{state_fips}{county_fips}'
GROUP BY co.areasymbol
```

## 4. US Census TIGER/Line Shapefiles

**Source:** United States Census Bureau  
**URL:** https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html  
**License:** Public domain (US Government data)  
**Access Method:** Shapefile download and processing

### Data Retrieved

| Data Type | Description | Use |
|-----------|-------------|-----|
| County boundaries | Polygon geometries | Spatial intersection |
| County centroids | Point coordinates (lat/lon) | Weather queries |
| FIPS codes | State + County identifiers | Data merging |

### Coverage

- **Temporal:** 2020 vintage used
- **Spatial:** All 3,143 US counties

### Processing

County centroids computed using:
```python
gdf = gpd.read_file("tl_2020_us_county.shp")
gdf['centroid'] = gdf.geometry.centroid
gdf['latitude'] = gdf.centroid.y
gdf['longitude'] = gdf.centroid.x
```

### Data Quality Notes

- Centroids approximate center of county area
- Does not account for corn-growing area distribution
- FIPS codes stable over time (minimal changes 1981-2023)

## Data Integration

### Merge Keys

Primary keys for joining datasets:
- **State FIPS Code** (2 digits, e.g., "17" for Illinois)
- **County FIPS Code** (3 digits, e.g., "019" for Champaign)
- **Year** (4 digits, e.g., "2020")

### Integration Process

1. **Corn Statistics Merge:**
   - Join yield, area, production by State FIPS + County FIPS + Year
   - Create derived features (abandonment rate, production density)

2. **Weather Feature Integration:**
   - Engineer features from daily/weekly weather data
   - Aggregate to county-year level
   - Join by State FIPS + County FIPS + Year

3. **Soil Properties Integration:**
   - Static properties joined by State FIPS + County FIPS
   - Same values replicated across all years for each county

4. **Temporal Feature Engineering:**
   - Create lag features (1-3 years) using historical data
   - Requires sorting by county and year

### Final Dataset Structure

**Dimensions:** 82,436 rows × 50 columns

**Feature Categories:**
- Target variable: Yield (BU/ACRE)
- Historical features: 8 (lags, changes, trends)
- Weather features: 33 (GDD, stress, precipitation)
- Soil features: 4 (AWC, clay, pH, OM)
- Area features: 3 (planted, harvested, abandonment)
- Temporal features: 2 (year, state)

## Data Limitations

### Temporal

- Annual resolution only (no within-season predictions)
- Historical data more sparse for early years (1980s)
- Climate change trends not explicitly modeled

### Spatial

- County-level aggregation masks field-level variability
- Weather data point-based (centroid) not area-weighted
- Soil data static, no temporal dynamics

### Coverage

- Focus on major corn-producing regions
- Limited data for marginal production areas
- Missing data for some county-year combinations

### Measurement

- Survey-based estimates have sampling uncertainty
- Satellite weather data has spatial resolution limits
- Soil surveys conducted at different times

## Data Access Scripts

All data collection scripts located in `src/data_collection/`:
- `get_yield_by_county.py`
- `get_area_planted_by_county.py`
- `get_area_harvested_by_county.py`
- `get_production_by_county.py`
- `get_county_centroids.py`
- `get_weather_data.py`
- `get_soil_data.py`

Run `scripts/01_download_all_data.py` to execute complete download pipeline.

## References

1. USDA NASS. (2024). Quick Stats Database. https://quickstats.nass.usda.gov/
2. NASA POWER Project. (2024). Agroclimatology Daily Data. https://power.larc.nasa.gov/
3. USDA NRCS. (2024). Soil Data Access. https://sdmdataaccess.sc.egov.usda.gov/
4. US Census Bureau. (2020). TIGER/Line Shapefiles. https://www.census.gov/

## Citation

When using this data pipeline, please cite:

```bibtex
@dataset{riaz2025corndata,
  author = {Riaz, Ahsan},
  title = {US County-Level Corn Yield Multi-Source Dataset},
  year = {2025},
  note = {Integrated dataset from USDA NASS, NASA POWER, USDA NRCS, and US Census}
}
```

