"""
Visualization utilities for the Streamlit dashboard.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from app.config import COLORS


def plot_choropleth_map(df, year):
    """
    Plot county-level yield map for a specific year.
    
    Args:
        df: DataFrame containing yield data
        year: Year to visualize
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Filter for year
    map_df = df[df['Year'] == year].copy()
    
    # Create FIPS code if not exists (State ANSI + County ANSI)
    # Ensure ANSI codes are string and padded
    if 'State ANSI' in map_df.columns and 'County ANSI' in map_df.columns:
        map_df['State ANSI'] = map_df['State ANSI'].astype(str).str.zfill(2)
        map_df['County ANSI'] = map_df['County ANSI'].astype(str).str.zfill(3)
        map_df['FIPS'] = map_df['State ANSI'] + map_df['County ANSI']
    
    # Use Plotly Express Choropleth
    # Note: This requires a GeoJSON file for US counties. 
    # Plotly has a built-in one but it might be slow or require internet.
    # We will use 'usa-states' scope for performance if county geojson is tricky,
    # but let's try county level first with built-in geojson.
    
    from urllib.request import urlopen
    import json
    
    # For robust offline/fast usage, we might want to use a scattergeo if geojson fails,
    # but let's try the standard plotly way first.
    
    fig = px.choropleth(
        map_df,
        locations='FIPS',
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        color='Yield_BU_ACRE',
        scope="usa",
        hover_name='County',
        hover_data=['State', 'Yield_BU_ACRE'],
        color_continuous_scale="Greens",
        title=f'Corn Yield by County ({year})'
    )
    
    fig.update_layout(
        margin={"r":0,"t":30,"l":0,"b":0},
        plot_bgcolor='white',
        geo=dict(
            lakecolor='rgb(255, 255, 255)',
        )
    )
    
    return fig

def plot_yield_trend(df, state=None, county=None, title=None):
    """
    Plot yield trend over time.
    
    Args:
        df: DataFrame with Year and Yield_BU_ACRE columns
        state: State name for filtering (optional)
        county: County name for filtering (optional)
        title: Plot title (optional)
    
    Returns:
        plotly.graph_objects.Figure
    """
    plot_df = df.copy()
    
    if state is not None:
        plot_df = plot_df[plot_df['State'] == state]
    if county is not None:
        plot_df = plot_df[plot_df['County'] == county]
    
    plot_df = plot_df.groupby('Year')['Yield_BU_ACRE'].mean().reset_index()
    
    fig = px.line(
        plot_df, 
        x='Year', 
        y='Yield_BU_ACRE',
        title=title or 'Average Yield Over Time',
        labels={'Yield_BU_ACRE': 'Yield (BU/ACRE)', 'Year': 'Year'},
        color_discrete_sequence=[COLORS['primary']]
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        hovermode='x unified'
    )
    
    return fig


def plot_yield_by_state(df, top_n=15):
    """Plot average yield by state (top N)."""
    state_avg = df.groupby('State')['Yield_BU_ACRE'].mean().sort_values(ascending=False).head(top_n)
    
    fig = px.bar(
        x=state_avg.values,
        y=state_avg.index,
        orientation='h',
        title=f'Top {top_n} States by Average Yield',
        labels={'x': 'Average Yield (BU/ACRE)', 'y': 'State'},
        color=state_avg.values,
        color_continuous_scale='Greens'
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        yaxis=dict(autorange='reversed'),
        showlegend=False
    )
    
    return fig


def plot_yield_distribution(df, bins=50):
    """Plot yield distribution histogram."""
    fig = px.histogram(
        df,
        x='Yield_BU_ACRE',
        nbins=bins,
        title='Yield Distribution',
        labels={'Yield_BU_ACRE': 'Yield (BU/ACRE)', 'count': 'Frequency'},
        color_discrete_sequence=[COLORS['primary']]
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    return fig


def plot_scatter_with_trend(x, y, xlabel, ylabel, title):
    """Plot scatter plot with trend line."""
    df_plot = pd.DataFrame({xlabel: x, ylabel: y})
    df_plot = df_plot.dropna()
    
    fig = px.scatter(
        df_plot,
        x=xlabel,
        y=ylabel,
        title=title,
        trendline='ols',
        labels={xlabel: xlabel, ylabel: ylabel}
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    return fig


def plot_historical_vs_predicted(historical_data, predicted_yield, predicted_year):
    """
    Plot historical yields with prediction.
    
    Args:
        historical_data: DataFrame with Year and Yield_BU_ACRE columns
        predicted_yield: Predicted yield value
        predicted_year: Year for prediction
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data['Year'],
        y=historical_data['Yield_BU_ACRE'],
        mode='lines+markers',
        name='Historical',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6)
    ))
    
    # Prediction
    fig.add_trace(go.Scatter(
        x=[predicted_year],
        y=[predicted_yield],
        mode='markers',
        name='Predicted',
        marker=dict(size=15, color=COLORS['success'], symbol='star')
    ))
    
    fig.update_layout(
        title='Historical Yields and Prediction',
        xaxis_title='Year',
        yaxis_title='Yield (BU/ACRE)',
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    return fig


def plot_model_comparison(comparison_data):
    """Plot model comparison metrics."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('R² Score', 'MAE (BU/ACRE)', 'RMSE (BU/ACRE)'),
        horizontal_spacing=0.1
    )
    
    metrics = ['r2', 'mae', 'rmse']
    for i, metric in enumerate(metrics, 1):
        fig.add_trace(
            go.Bar(
                x=comparison_data['model'],
                y=comparison_data[metric],
                name=metric.upper(),
                marker_color=COLORS['primary']
            ),
            row=1, col=i
        )
    
    fig.update_layout(
        title='Model Performance Comparison',
        showlegend=False,
        height=400
    )
    
    return fig

