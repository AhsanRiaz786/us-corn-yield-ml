"""
Scenarios Page - What-if analysis for different weather and condition scenarios.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from app.utils import (
    load_data, get_unique_states, get_counties_by_state,
    get_year_range, predict_yield, get_latest_county_data
)
from app.config import MODEL_METADATA
import plotly.graph_objects as go


st.set_page_config(page_title="Scenarios", layout="wide")
st.title("Scenario Analysis")

st.write("Analyze the impact of different weather and growing conditions on corn yield predictions.")

# Load data
try:
    df = load_data()
    states = get_unique_states(df)
    year_min, year_max = get_year_range(df)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar - Base scenario setup
st.sidebar.header("Base Scenario")

selected_state = st.sidebar.selectbox("State", states)
if selected_state:
    counties = get_counties_by_state(df, selected_state)
    selected_county = st.sidebar.selectbox("County", counties) if counties else None
else:
    selected_county = None

selected_year = st.sidebar.number_input(
    "Year",
    min_value=year_min + 4,
    max_value=year_max + 10,
    value=year_max + 1
)

selected_model = st.sidebar.selectbox(
    "Model",
    list(MODEL_METADATA.keys()),
    format_func=lambda x: MODEL_METADATA[x]['name'],
    index=0
)

# Get base scenario
if selected_state and selected_county:
    try:
        base_result = predict_yield(
            selected_state,
            selected_county,
            selected_year,
            model_name=selected_model,
            df=df
        )
        
        st.session_state['base_scenario'] = {
            'result': base_result,
            'state': selected_state,
            'county': selected_county,
            'year': selected_year
        }
    except Exception as e:
        st.error(f"Error creating base scenario: {str(e)}")
        st.stop()
else:
    st.info("Please select a state and county to begin scenario analysis.")
    st.stop()

# Main content
base = st.session_state.get('base_scenario')

if base:
    st.subheader(f"Base Scenario: {base['county']}, {base['state']} ({base['year']})")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Base Prediction", f"{base['result']['predicted_yield']:.1f} BU/ACRE")
    with col2:
        st.metric("Lower Bound", f"{base['result']['confidence_lower']:.1f} BU/ACRE")
    with col3:
        st.metric("Upper Bound", f"{base['result']['confidence_upper']:.1f} BU/ACRE")
    
    st.markdown("---")
    
    # Scenario selection
    st.header("Select Scenario")
    
    scenario_type = st.radio(
        "Scenario Type",
        ["Preset Scenarios", "Custom Scenario"],
        horizontal=True
    )
    
    scenarios = {}
    
    if scenario_type == "Preset Scenarios":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Drought Scenario")
            st.write("Reduced precipitation and increased heat stress")
            
            if st.button("Run Drought Scenario", key="drought"):
                base_features = base['result']['features_used']
                scenarios['Drought'] = {
                    # Precipitation reductions (40% less rain)
                    'precip_total': base_features.get('precip_total', 500) * 0.6,
                    'precip_vegetative': base_features.get('precip_vegetative', 150) * 0.6,
                    'precip_reproductive': base_features.get('precip_reproductive', 150) * 0.6,
                    'precip_grainfill': base_features.get('precip_grainfill', 150) * 0.6,
                    'precip_mean_weekly': base_features.get('precip_mean_weekly', 20) * 0.6,
                    'precip_max_weekly': base_features.get('precip_max_weekly', 40) * 0.6,
                    # Precipitation anomalies (negative = below normal)
                    'precip_anomaly_mm': -150,  # 150mm below normal
                    'precip_anomaly_pct': -40.0,  # 40% below normal
                    # Dry conditions
                    'weeks_dry': base_features.get('weeks_dry', 3) * 2.5,  # More dry weeks
                    'weeks_very_dry': base_features.get('weeks_very_dry', 1) * 3.0,
                    'weeks_wet': 0,  # No wet weeks during drought
                    # Water stress
                    'water_stress_reproductive': 3.0,  # High water stress
                    # Heat stress (drought often comes with heat)
                    'weeks_heat_stress': base_features.get('weeks_heat_stress', 2) * 2.5,
                    'weeks_extreme_heat': base_features.get('weeks_extreme_heat', 0.5) * 3.0,
                    'heat_moisture_stress': base_features.get('heat_moisture_stress', 1.0) * 2.5,  # Combined stress
                    # Temperature increases (drought = hotter)
                    'temp_mean_season': base_features.get('temp_mean_season', 20) + 2.0,
                    'temp_max_season': base_features.get('temp_max_season', 28) + 2.5,
                    'temp_mean_reproductive': base_features.get('temp_mean_reproductive', 22) + 2.0,
                    'temp_anomaly': 2.0,  # 2°C above normal
                    'gdd_anomaly': base_features.get('gdd_anomaly', 0) + 150,  # More GDD
                }
        
        with col2:
            st.subheader("Optimal Weather")
            st.write("Ideal growing conditions")
            
            if st.button("Run Optimal Scenario", key="optimal"):
                base_features = base['result']['features_used']
                scenarios['Optimal'] = {
                    # Optimal = conditions close to county normal (near-zero anomalies)
                    # Keep precipitation near normal levels (small positive deviation is OK)
                    'precip_total': base_features.get('precip_total', 500) * 1.05,  # 5% above normal
                    'precip_vegetative': base_features.get('precip_vegetative', 150) * 1.05,
                    'precip_reproductive': base_features.get('precip_reproductive', 150) * 1.08,  # Slightly extra during critical stage
                    'precip_grainfill': base_features.get('precip_grainfill', 150) * 1.03,
                    'precip_mean_weekly': base_features.get('precip_mean_weekly', 20) * 1.05,
                    # Near-zero anomalies (conditions close to county normal = optimal)
                    'precip_anomaly_mm': 25,  # Small positive (25mm above normal)
                    'precip_anomaly_pct': 5.0,  # 5% above normal (not 15%!)
                    # Minimal dry conditions (but not zero - some dry is normal)
                    'weeks_dry': max(1.0, base_features.get('weeks_dry', 3) * 0.7),  # Fewer dry weeks
                    'weeks_very_dry': 0,  # No very dry weeks
                    'weeks_wet': max(0, base_features.get('weeks_wet', 2) - 0.5),  # Moderate, not excessive
                    # Minimal water stress
                    'water_stress_reproductive': 0.0,
                    # No heat stress (critical for optimal conditions)
                    'weeks_heat_stress': 0,
                    'weeks_extreme_heat': 0,
                    'heat_moisture_stress': 0.0,
                    # Optimal temperatures (close to normal)
                    'temp_mean_season': base_features.get('temp_mean_season', 20),  # At normal
                    'temp_mean_reproductive': base_features.get('temp_mean_reproductive', 22),  # At normal
                    'temp_anomaly': 0.0,  # Near-zero anomaly (optimal)
                    'gdd_anomaly': 0.0,  # Near-zero anomaly (optimal)
                    # Normal temperature variability (not too high)
                    'temp_std_season': base_features.get('temp_std_season', 3) * 0.9,  # Slightly lower variability
                    # Normal humidity (not excessive)
                    'weeks_high_humidity': max(0, base_features.get('weeks_high_humidity', 2) * 0.8),
                }
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Extreme Heat")
            st.write("Very high temperatures and heat stress")
            
            if st.button("Run Heat Scenario", key="heat"):
                base_features = base['result']['features_used']
                scenarios['Extreme Heat'] = {
                    # Extreme heat stress
                    'weeks_heat_stress': base_features.get('weeks_heat_stress', 2) * 3.5,
                    'weeks_extreme_heat': base_features.get('weeks_extreme_heat', 0.5) * 4.0,
                    'heat_moisture_stress': base_features.get('heat_moisture_stress', 1.0) * 2.0,
                    # High temperatures
                    'temp_mean_season': base_features.get('temp_mean_season', 20) + 3.0,
                    'temp_max_season': base_features.get('temp_max_season', 28) + 4.0,
                    'temp_min_season': base_features.get('temp_min_season', 15) + 2.0,
                    'temp_mean_reproductive': base_features.get('temp_mean_reproductive', 22) + 3.0,
                    'temp_max_reproductive': base_features.get('temp_max_reproductive', 30) + 3.5,
                    'temp_anomaly': 3.0,  # 3°C above normal
                    'temp_std_season': base_features.get('temp_std_season', 3) * 1.2,  # More variability
                    'temp_range_avg': base_features.get('temp_range_avg', 10) + 2.0,
                    # Increased GDD
                    'gdd_total': base_features.get('gdd_total', 2800) * 1.15,
                    'gdd_vegetative': base_features.get('gdd_vegetative', 1000) * 1.15,
                    'gdd_reproductive': base_features.get('gdd_reproductive', 800) * 1.15,
                    'gdd_grainfill': base_features.get('gdd_grainfill', 600) * 1.15,
                    'gdd_anomaly': base_features.get('gdd_anomaly', 0) + 200,
                    # High humidity during heat (stressful combination)
                    'weeks_high_humidity': base_features.get('weeks_high_humidity', 2) * 1.5,
                }
        
        with col4:
            st.subheader("Excessive Rain")
            st.write("Above-normal precipitation (can reduce yield)")
            
            if st.button("Run Rain Scenario", key="rain"):
                base_features = base['result']['features_used']
                scenarios['Excessive Rain'] = {
                    # Very high precipitation
                    'precip_total': base_features.get('precip_total', 500) * 1.5,
                    'precip_vegetative': base_features.get('precip_vegetative', 150) * 1.5,
                    'precip_reproductive': base_features.get('precip_reproductive', 150) * 1.6,
                    'precip_grainfill': base_features.get('precip_grainfill', 150) * 1.4,
                    'precip_mean_weekly': base_features.get('precip_mean_weekly', 20) * 1.5,
                    'precip_max_weekly': base_features.get('precip_max_weekly', 40) * 1.8,  # Heavy downpours
                    'precip_std': base_features.get('precip_std', 15) * 1.3,  # High variability
                    # Large positive anomalies
                    'precip_anomaly_mm': 200,
                    'precip_anomaly_pct': 40.0,
                    # Wet conditions
                    'weeks_wet': 8.0,
                    'weeks_dry': 0,  # No dry weeks
                    'weeks_very_dry': 0,
                    # Water logging stress (negative for yield)
                    'water_stress_reproductive': 0.0,  # No water deficit, but potential for waterlogging
                    # Cooler temperatures (often associated with excessive rain)
                    'temp_mean_season': base_features.get('temp_mean_season', 20) - 1.0,
                    'temp_max_season': base_features.get('temp_max_season', 28) - 1.5,
                    'temp_mean_reproductive': base_features.get('temp_mean_reproductive', 22) - 1.0,
                    'temp_anomaly': -1.0,  # 1°C below normal
                    # Reduced heat stress
                    'weeks_heat_stress': max(0, base_features.get('weeks_heat_stress', 2) * 0.5),
                    'weeks_extreme_heat': 0,
                    'heat_moisture_stress': max(0, base_features.get('heat_moisture_stress', 1.0) * 0.5),
                    # High humidity
                    'weeks_high_humidity': base_features.get('weeks_high_humidity', 2) * 2.0,
                    'rh_mean': min(100, base_features.get('rh_mean', 70) + 5),
                    'rh_reproductive': min(100, base_features.get('rh_reproductive', 75) + 5),
                }
    
    else:  # Custom scenario
        st.subheader("Custom Weather Parameters")
        st.caption("Adjust weather parameters to test specific conditions. Only modified parameters will override defaults.")
        
        base_features = base['result']['features_used']
        
        # Organize parameters into tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Precipitation", "Temperature", "Stress Indicators", "Anomalies"])
        
        custom_overrides = {}
        
        with tab1:
            st.write("**Precipitation Parameters**")
            
            custom_precip_total = st.number_input(
                "Total Precipitation (mm)",
                value=float(base_features.get('precip_total', 500)),
                step=10.0,
                help="Total growing season rainfall"
            )
            if custom_precip_total != base_features.get('precip_total', 500):
                custom_overrides['precip_total'] = custom_precip_total
            
            custom_precip_repro = st.number_input(
                "Reproductive Stage Precipitation (mm)",
                value=float(base_features.get('precip_reproductive', 150)),
                step=5.0,
                help="Rainfall during critical reproductive period (most important for yield)"
            )
            if custom_precip_repro != base_features.get('precip_reproductive', 150):
                custom_overrides['precip_reproductive'] = custom_precip_repro
            
            custom_weeks_dry = st.number_input(
                "Dry Weeks (precip < 10mm/week)",
                value=float(base_features.get('weeks_dry', 3)),
                step=0.5,
                help="Number of weeks with low rainfall"
            )
            if custom_weeks_dry != base_features.get('weeks_dry', 3):
                custom_overrides['weeks_dry'] = custom_weeks_dry
            
            custom_weeks_wet = st.number_input(
                "Wet Weeks (precip > 40mm/week)",
                value=float(base_features.get('weeks_wet', 2)),
                step=0.5,
                help="Number of weeks with heavy rainfall"
            )
            if custom_weeks_wet != base_features.get('weeks_wet', 2):
                custom_overrides['weeks_wet'] = custom_weeks_wet
        
        with tab2:
            st.write("**Temperature Parameters**")
            
            custom_gdd_total = st.number_input(
                "Total Growing Degree Days (GDD)",
                value=float(base_features.get('gdd_total', 2800)),
                step=50.0,
                help="Accumulated heat units for the season"
            )
            if custom_gdd_total != base_features.get('gdd_total', 2800):
                custom_overrides['gdd_total'] = custom_gdd_total
            
            custom_temp_mean = st.number_input(
                "Mean Season Temperature (°C)",
                value=float(base_features.get('temp_mean_season', 20)),
                step=0.5,
                help="Average temperature during growing season"
            )
            if custom_temp_mean != base_features.get('temp_mean_season', 20):
                custom_overrides['temp_mean_season'] = custom_temp_mean
            
            custom_temp_repro = st.number_input(
                "Reproductive Stage Mean Temp (°C)",
                value=float(base_features.get('temp_mean_reproductive', 22)),
                step=0.5,
                help="Average temperature during critical reproductive period"
            )
            if custom_temp_repro != base_features.get('temp_mean_reproductive', 22):
                custom_overrides['temp_mean_reproductive'] = custom_temp_repro
        
        with tab3:
            st.write("**Stress Indicators**")
            
            custom_heat_stress = st.number_input(
                "Heat Stress Weeks (Tmax > 32°C)",
                value=float(base_features.get('weeks_heat_stress', 2)),
                step=0.5,
                help="Weeks with maximum temperature above 32°C"
            )
            if custom_heat_stress != base_features.get('weeks_heat_stress', 2):
                custom_overrides['weeks_heat_stress'] = custom_heat_stress
            
            custom_extreme_heat = st.number_input(
                "Extreme Heat Weeks (Tmax > 35°C)",
                value=float(base_features.get('weeks_extreme_heat', 0.5)),
                step=0.5,
                help="Weeks with maximum temperature above 35°C"
            )
            if custom_extreme_heat != base_features.get('weeks_extreme_heat', 0.5):
                custom_overrides['weeks_extreme_heat'] = custom_extreme_heat
            
            custom_water_stress = st.number_input(
                "Water Stress (Reproductive)",
                value=float(base_features.get('water_stress_reproductive', 1.0)),
                step=0.5,
                help="Water stress indicator during reproductive stage"
            )
            if custom_water_stress != base_features.get('water_stress_reproductive', 1.0):
                custom_overrides['water_stress_reproductive'] = custom_water_stress
        
        with tab4:
            st.write("**Climate Anomalies**")
            st.caption("Deviations from county normal (30-year average). Near-zero = conditions close to normal.")
            
            custom_precip_anomaly_mm = st.number_input(
                "Precipitation Anomaly (mm)",
                value=float(base_features.get('precip_anomaly_mm', 0)),
                step=10.0,
                help="Difference from county normal precipitation (mm)"
            )
            if custom_precip_anomaly_mm != base_features.get('precip_anomaly_mm', 0):
                custom_overrides['precip_anomaly_mm'] = custom_precip_anomaly_mm
            
            custom_precip_anomaly_pct = st.number_input(
                "Precipitation Anomaly (%)",
                value=float(base_features.get('precip_anomaly_pct', 0)),
                step=1.0,
                help="Percentage difference from county normal precipitation"
            )
            if custom_precip_anomaly_pct != base_features.get('precip_anomaly_pct', 0):
                custom_overrides['precip_anomaly_pct'] = custom_precip_anomaly_pct
            
            custom_temp_anomaly = st.number_input(
                "Temperature Anomaly (°C)",
                value=float(base_features.get('temp_anomaly', 0)),
                step=0.5,
                help="Difference from county normal temperature"
            )
            if custom_temp_anomaly != base_features.get('temp_anomaly', 0):
                custom_overrides['temp_anomaly'] = custom_temp_anomaly
            
            custom_gdd_anomaly = st.number_input(
                "GDD Anomaly",
                value=float(base_features.get('gdd_anomaly', 0)),
                step=25.0,
                help="Difference from county normal GDD"
            )
            if custom_gdd_anomaly != base_features.get('gdd_anomaly', 0):
                custom_overrides['gdd_anomaly'] = custom_gdd_anomaly
        
        if st.button("Run Custom Scenario", type="primary"):
            if custom_overrides:
                scenarios['Custom'] = custom_overrides
            else:
                st.info("No parameters changed. Modify at least one parameter to run a custom scenario.")
    
    # Run scenarios and display results
    if scenarios:
        st.markdown("---")
        st.header("Scenario Results")
        
        scenario_results = {}
        
        for scenario_name, weather_overrides in scenarios.items():
            try:
                result = predict_yield(
                    base['state'],
                    base['county'],
                    base['year'],
                    model_name=selected_model,
                    weather_overrides=weather_overrides,
                    df=df
                )
                scenario_results[scenario_name] = result
            except Exception as e:
                st.error(f"Error running {scenario_name} scenario: {str(e)}")
        
        if scenario_results:
            # Comparison table
            comparison_data = []
            
            for name, result in scenario_results.items():
                diff = result['predicted_yield'] - base['result']['predicted_yield']
                diff_pct = (diff / base['result']['predicted_yield']) * 100
                
                comparison_data.append({
                    'Scenario': name,
                    'Predicted Yield (BU/ACRE)': f"{result['predicted_yield']:.1f}",
                    'Difference': f"{diff:+.1f}",
                    'Change %': f"{diff_pct:+.1f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Comparison chart
            fig = go.Figure()
            
            scenarios_list = list(scenario_results.keys())
            yields = [base['result']['predicted_yield']] + [scenario_results[s]['predicted_yield'] for s in scenarios_list]
            labels = ['Base'] + scenarios_list
            
            colors = ['#1f77b4'] + ['#2ca02c' if y > base['result']['predicted_yield'] else '#d62728' for y in yields[1:]]
            
            fig.add_trace(go.Bar(
                x=labels,
                y=yields,
                marker_color=colors,
                text=[f"{y:.1f}" for y in yields],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Scenario Comparison',
                xaxis_title='Scenario',
                yaxis_title='Predicted Yield (BU/ACRE)',
                plot_bgcolor='white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

