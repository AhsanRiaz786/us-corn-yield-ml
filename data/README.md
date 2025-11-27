# Data Directory

This directory contains all data files used in the US Corn Yield Prediction System.

## Directory Structure

```
data/
├── raw/                    # Original downloaded data (not version controlled)
│   ├── us_corn_yield_county_all_years.csv
│   ├── us_corn_area_planted_county_all_years.csv
│   ├── us_corn_area_harvested_county_all_years.csv
│   ├── us_corn_production_county_all_years.csv
│   ├── county_centroids.csv
│   ├── weather_data_weekly.csv
│   └── county_soil_aggregates.csv
│
└── processed/              # Cleaned and merged datasets
    ├── master_corn_county_data.csv
    ├── weather_features_county_year.csv
    ├── modeling_dataset_complete.csv
    └── modeling_dataset_final.csv
```

## Raw Data Files

### USDA NASS QuickStats

1. **us_corn_yield_county_all_years.csv**
   - Source: USDA NASS QuickStats
   - Content: County-level corn grain yield (BU/ACRE)
   - Years: 1981-2023
   - Records: ~80,000

2. **us_corn_area_planted_county_all_years.csv**
   - Source: USDA NASS QuickStats
   - Content: Total acres planted to corn
   - Years: 1981-2023
   - Records: ~80,000

3. **us_corn_area_harvested_county_all_years.csv**
   - Source: USDA NASS QuickStats
   - Content: Total acres harvested for grain
   - Years: 1981-2023
   - Records: ~80,000

4. **us_corn_production_county_all_years.csv**
   - Source: USDA NASS QuickStats
   - Content: Total corn grain production (BU)
   - Years: 1981-2023
   - Records: ~80,000

### Geographic Data

5. **county_centroids.csv**
   - Source: US Census TIGER/Line Shapefiles
   - Content: County centroid coordinates (latitude, longitude)
   - Records: 3,143 counties
   - Columns: State_FIPS, County_FIPS, Latitude, Longitude

### Weather Data

6. **weather_data_weekly.csv**
   - Source: NASA POWER Agroclimatology API
   - Content: Weekly aggregated weather data (April-September)
   - Variables: Temperature (min/max/mean), Precipitation, Relative humidity
   - Records: ~120,000 county-year-week observations
   - Size: ~450 MB

### Soil Data

7. **county_soil_aggregates.csv**
   - Source: USDA NRCS Soil Data Access (gSSURGO)
   - Content: County-aggregated soil properties
   - Variables: AWC, Clay%, pH, Organic matter%
   - Records: ~2,500 counties
   - Coverage: Counties with valid soil survey data

## Processed Data Files

### Intermediate Files

1. **master_corn_county_data.csv**
   - Description: Merged USDA NASS datasets (yield, area, production)
   - Records: ~85,000 county-year observations
   - Features: Yield, Area_Planted, Area_Harvested, Production, Abandonment_rate

2. **weather_features_county_year.csv**
   - Description: Engineered weather features at county-year level
   - Records: ~82,000 county-year observations
   - Features: 33 weather metrics (GDD, heat stress, precipitation, etc.)
   - Size: ~55 MB

### Final Modeling Dataset

3. **modeling_dataset_complete.csv**
   - Description: Complete records after initial merge
   - Records: ~85,000 observations
   - Features: Corn statistics + Weather + Soil properties

4. **modeling_dataset_final.csv** ⭐
   - Description: Final dataset used for model training
   - Records: 82,436 county-year observations
   - Features: 50+ features including historical lags
   - Columns:
     - Identifiers: State, County_FIPS, Year
     - Target: Yield_BU_ACRE
     - Historical: Yield lags (1-3 years), rolling averages, changes
     - Weather: GDD, heat stress, precipitation, temperature extremes
     - Soil: AWC, Clay%, pH, Organic matter%
     - Area: Planted, Harvested, Abandonment rate
   - Size: ~45 MB
   - **This is the primary input for model training**

## Data Download

Raw data files are not included in version control due to size. To download all data:

```bash
python scripts/01_download_all_data.py
```

This will download data from all sources and save to `data/raw/`.

Estimated download time: 2-4 hours depending on network speed.

## Data Processing

To regenerate processed files from raw data:

```bash
python scripts/02_prepare_data.py
```

This will:
1. Merge corn statistics datasets
2. Engineer weather features
3. Integrate soil properties
4. Create lag features
5. Output final modeling dataset

Processing time: ~10 minutes

## Data Quality

### Coverage

- **Temporal:** 1981-2023 (43 years)
- **Spatial:** 2,635 counties with corn production
- **Completeness:** 82,436 complete records (98.5% of potential county-years)

### Missing Data

- Early years (1981-1983): Fewer counties reporting
- Soil data: ~100 counties missing (urban/non-agricultural areas)
- Weather data: <0.1% missing (interpolated or dropped)

### Data Validation

All processed datasets undergo validation:
- Range checks on all numeric fields
- Temporal consistency checks (year sequences)
- Spatial consistency checks (FIPS codes valid)
- Cross-dataset validation (area planted ≥ area harvested)

## Data Citation

When using this data, please cite the original sources:

```bibtex
@dataset{usda2024quickstats,
  author = {{USDA NASS}},
  title = {Quick Stats Database},
  year = {2024},
  url = {https://quickstats.nass.usda.gov/}
}

@dataset{nasa2024power,
  author = {{NASA POWER Project}},
  title = {Agroclimatology Daily Data},
  year = {2024},
  url = {https://power.larc.nasa.gov/}
}

@dataset{usda2024soil,
  author = {{USDA NRCS}},
  title = {Soil Data Access - gSSURGO},
  year = {2024},
  url = {https://sdmdataaccess.sc.egov.usda.gov/}
}
```

## File Formats

All data files are in CSV format with the following conventions:
- Encoding: UTF-8
- Delimiter: Comma (,)
- Line endings: Unix (LF)
- Missing values: Empty string or NaN
- Date format: YYYY for years, YYYY-MM-DD for full dates

## Storage Requirements

Total storage space required:

- Raw data: ~500 MB
- Processed data: ~100 MB
- **Total:** ~600 MB

Ensure sufficient disk space before running data download scripts.

## Backup and Archival

Raw data files should be backed up after initial download to avoid repeated downloads. Processed files can be regenerated from raw data.

## Questions?

For questions about data sources, processing, or quality, see:
- `docs/data_sources.md` for detailed source documentation
- `docs/feature_engineering.md` for processing methodology
- `README.md` in project root for overall system documentation

