"""
Data loading and caching utilities for the Streamlit dashboard.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st
from app.config import DATA_FILE, PROJECT_ROOT


@st.cache_data
def load_data():
    """
    Load the main modeling dataset with caching.
    
    Returns:
        pd.DataFrame: The complete modeling dataset
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)
    return df


@st.cache_data
def get_unique_states(df):
    """Get sorted list of unique states."""
    return sorted(df['State'].unique().tolist())


@st.cache_data
def get_counties_by_state(df, state):
    """Get sorted list of counties for a given state."""
    if state is None:
        return []
    state_data = df[df['State'] == state]
    return sorted(state_data['County'].unique().tolist())


@st.cache_data
def get_year_range(df):
    """Get minimum and maximum years in dataset."""
    return int(df['Year'].min()), int(df['Year'].max())


@st.cache_data
def get_historical_yields(df, state, county, current_year=None):
    """
    Get historical yield data for a county.
    
    Args:
        df: Full dataset
        state: State name
        county: County name
        current_year: Current year (to exclude from results)
    
    Returns:
        pd.DataFrame: Historical yields sorted by year
    """
    county_data = df[(df['State'] == state) & (df['County'] == county)].copy()
    
    if current_year is not None:
        county_data = county_data[county_data['Year'] < current_year]
    
    county_data = county_data.sort_values('Year')
    
    return county_data[['Year', 'Yield_BU_ACRE']].copy()


@st.cache_data
def get_county_soil_data(df, state, county):
    """
    Get soil properties for a county.
    
    Returns:
        dict: Soil properties or defaults if not found
    """
    county_data = df[(df['State'] == state) & (df['County'] == county)]
    
    if len(county_data) == 0:
        return None
    
    soil_cols = ['Soil_AWC', 'Soil_Clay_Pct', 'Soil_pH', 'Soil_Organic_Matter_Pct']
    
    soil_data = {}
    for col in soil_cols:
        values = county_data[col].dropna()
        if len(values) > 0:
            soil_data[col] = float(values.iloc[0])
        else:
            soil_data[col] = None
    
    return soil_data


@st.cache_data
def get_latest_county_data(df, state, county):
    """
    Get the most recent year's data for a county.
    
    Returns:
        pd.Series: Latest year's data row
    """
    county_data = df[(df['State'] == state) & (df['County'] == county)]
    
    if len(county_data) == 0:
        return None
    
    latest = county_data.loc[county_data['Year'].idxmax()]
    return latest


@st.cache_data
def filter_data(df, states=None, counties=None, year_min=None, year_max=None):
    """
    Filter dataset based on criteria.
    
    Args:
        df: Full dataset
        states: List of states or None
        counties: List of counties or None
        year_min: Minimum year or None
        year_max: Maximum year or None
    
    Returns:
        pd.DataFrame: Filtered dataset
    """
    filtered = df.copy()
    
    if states is not None and len(states) > 0:
        filtered = filtered[filtered['State'].isin(states)]
    
    if counties is not None and len(counties) > 0:
        filtered = filtered[filtered['County'].isin(counties)]
    
    if year_min is not None:
        filtered = filtered[filtered['Year'] >= year_min]
    
    if year_max is not None:
        filtered = filtered[filtered['Year'] <= year_max]
    
    return filtered


@st.cache_data
def get_state_statistics(df, state=None):
    """
    Get yield statistics for state(s).
    
    Args:
        df: Dataset
        state: State name or None for all states
    
    Returns:
        dict: Statistics (mean, median, std, count)
    """
    if state is not None:
        data = df[df['State'] == state]['Yield_BU_ACRE']
    else:
        data = df['Yield_BU_ACRE']
    
    return {
        'mean': float(data.mean()),
        'median': float(data.median()),
        'std': float(data.std()),
        'min': float(data.min()),
        'max': float(data.max()),
        'count': int(len(data))
    }


@st.cache_data
def get_dataset_summary(df):
    """Get overall dataset summary statistics."""
    return {
        'total_records': len(df),
        'num_states': df['State'].nunique(),
        'num_counties': df.groupby(['State', 'County']).ngroups,
        'year_range': (int(df['Year'].min()), int(df['Year'].max())),
        'avg_yield': float(df['Yield_BU_ACRE'].mean()),
        'num_features': len(df.columns)
    }

