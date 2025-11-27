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
                scenarios['Drought'] = {
                    'precip_total': base['result']['features_used'].get('precip_total', 500) * 0.6,
                    'weeks_heat_stress': base['result']['features_used'].get('weeks_heat_stress', 2) * 2.5,
                    'precip_anomaly_mm': -100
                }
        
        with col2:
            st.subheader("Optimal Weather")
            st.write("Ideal growing conditions")
            
            if st.button("Run Optimal Scenario", key="optimal"):
                scenarios['Optimal'] = {
                    'precip_total': base['result']['features_used'].get('precip_total', 500) * 1.2,
                    'weeks_heat_stress': 0,
                    'precip_anomaly_mm': 50
                }
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Extreme Heat")
            st.write("Very high temperatures and heat stress")
            
            if st.button("Run Heat Scenario", key="heat"):
                scenarios['Extreme Heat'] = {
                    'weeks_heat_stress': base['result']['features_used'].get('weeks_heat_stress', 2) * 3,
                    'gdd_total': base['result']['features_used'].get('gdd_total', 2800) * 1.1,
                    'temp_anomaly': 2.0
                }
        
        with col4:
            st.subheader("Excessive Rain")
            st.write("Above-normal precipitation")
            
            if st.button("Run Rain Scenario", key="rain"):
                scenarios['Excessive Rain'] = {
                    'precip_total': base['result']['features_used'].get('precip_total', 500) * 1.5,
                    'precip_anomaly_mm': 150,
                    'weeks_wet': 8
                }
    
    else:  # Custom scenario
        st.subheader("Custom Weather Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            custom_precip = st.number_input(
                "Total Precipitation (mm)",
                value=float(base['result']['features_used'].get('precip_total', 500)),
                step=10.0
            )
            custom_heat_stress = st.number_input(
                "Heat Stress Weeks",
                value=float(base['result']['features_used'].get('weeks_heat_stress', 2)),
                step=0.5
            )
        
        with col2:
            custom_gdd = st.number_input(
                "Total GDD",
                value=float(base['result']['features_used'].get('gdd_total', 2800)),
                step=50.0
            )
            custom_precip_anomaly = st.number_input(
                "Precipitation Anomaly (mm)",
                value=float(base['result']['features_used'].get('precip_anomaly_mm', 0)),
                step=10.0
            )
        
        if st.button("Run Custom Scenario"):
            scenarios['Custom'] = {
                'precip_total': custom_precip,
                'weeks_heat_stress': custom_heat_stress,
                'gdd_total': custom_gdd,
                'precip_anomaly_mm': custom_precip_anomaly
            }
    
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

