"""
Data Exploration Page - Interactive EDA with filters and visualizations.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from app.utils import (
    load_data, get_unique_states, get_counties_by_state,
    get_year_range, filter_data
)
from app.utils.visualizations import (
    plot_yield_trend, plot_yield_by_state, plot_yield_distribution,
    plot_scatter_with_trend, plot_choropleth_map
)


st.set_page_config(page_title="Data Exploration", layout="wide")
st.title("Data Exploration")

# Load data
try:
    df = load_data()
    states = get_unique_states(df)
    year_min, year_max = get_year_range(df)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

# State filter
selected_states = st.sidebar.multiselect(
    "Select States",
    states,
    default=[]
)

# County filter (if states selected)
selected_counties = []
if selected_states:
    all_counties = []
    for state in selected_states:
        counties = get_counties_by_state(df, state)
        all_counties.extend(counties)
    
    if all_counties:
        selected_counties = st.sidebar.multiselect(
            "Select Counties (optional)",
            sorted(set(all_counties)),
            default=[]
        )

# Year range filter
st.sidebar.subheader("Year Range")
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

# Apply filters
filtered_df = filter_data(
    df,
    states=selected_states if selected_states else None,
    counties=selected_counties if selected_counties else None,
    year_min=year_range[0],
    year_max=year_range[1]
)

# Display filtered dataset info
st.info(f"Showing {len(filtered_df):,} records")

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs([
    "Geographic Analysis",
    "Temporal Trends",
    "Yield Distribution",
    "Weather Impact"
])

with tab1:
    st.subheader("Geographic Analysis")
    
    # Interactive Map
    st.write(f"**County-Level Yield Map ({year_range[1]})**")
    map_fig = plot_choropleth_map(filtered_df, year_range[1])
    st.plotly_chart(map_fig, use_container_width=True)
    
    # Top states
    st.write("**Top 15 States by Average Yield**")
    fig = plot_yield_by_state(filtered_df, top_n=15)
    st.plotly_chart(fig, use_container_width=True)
    
    # State statistics
    if selected_states:
        st.subheader("Selected States Statistics")
        state_stats = filtered_df.groupby('State')['Yield_BU_ACRE'].agg(['mean', 'std', 'count']).round(2)
        state_stats.columns = ['Mean Yield', 'Std Dev', 'Count']
        st.dataframe(state_stats, use_container_width=True)

with tab2:
    st.subheader("Temporal Trends")
    
    # National trend
    st.write("**National Average Yield Trend**")
    fig = plot_yield_trend(filtered_df, title="National Yield Trend")
    st.plotly_chart(fig, use_container_width=True)
    
    # State comparison
    if selected_states:
        st.write("**State Comparison**")
        state_trends = filtered_df[filtered_df['State'].isin(selected_states)].groupby(['State', 'Year'])['Yield_BU_ACRE'].mean().reset_index()
        
        import plotly.express as px
        fig = px.line(
            state_trends,
            x='Year',
            y='Yield_BU_ACRE',
            color='State',
            title='Yield Trends by State',
            labels={'Yield_BU_ACRE': 'Yield (BU/ACRE)', 'Year': 'Year'}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Yield Distribution")
    
    # Distribution histogram
    fig = plot_yield_distribution(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{filtered_df['Yield_BU_ACRE'].mean():.1f} BU/ACRE")
    with col2:
        st.metric("Median", f"{filtered_df['Yield_BU_ACRE'].median():.1f} BU/ACRE")
    with col3:
        st.metric("Std Dev", f"{filtered_df['Yield_BU_ACRE'].std():.1f} BU/ACRE")
    with col4:
        st.metric("Range", f"{filtered_df['Yield_BU_ACRE'].max() - filtered_df['Yield_BU_ACRE'].min():.1f} BU/ACRE")

with tab4:
    st.subheader("Weather Impact Analysis")
    
    # Check available weather columns
    weather_cols = [col for col in filtered_df.columns if any(x in col.lower() for x in ['gdd', 'temp', 'precip', 'heat'])]
    
    if weather_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("X-axis Variable", weather_cols[:10])
        
        with col2:
            st.write("Y-axis: Yield (BU/ACRE)")
        
        if x_var:
            fig = plot_scatter_with_trend(
                filtered_df[x_var],
                filtered_df['Yield_BU_ACRE'],
                x_var,
                'Yield_BU_ACRE',
                f'{x_var} vs Yield'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Weather features not available in filtered data")

# Data table
st.markdown("---")
st.subheader("Data Table")
st.dataframe(filtered_df.head(1000), use_container_width=True, height=400)

