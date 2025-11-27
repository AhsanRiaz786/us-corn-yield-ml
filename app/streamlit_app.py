import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import joblib
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
st.set_page_config(
    page_title="US Corn Yield Prediction",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'modeling_dataset_final.csv'
MODELS_DIR = PROJECT_ROOT / 'models'
MODEL_PATH = MODELS_DIR / 'xgboost_model.pkl'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'
COLS_PATH = MODELS_DIR / 'feature_columns.pkl'

# Load Data
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Data file not found at {DATA_PATH}")
        return None
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}")
        return None, None, None
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(COLS_PATH)
    return model, scaler, features

# Main App
def main():
    st.title("🌽 US Corn Yield Prediction System")
    st.markdown("""
    **Predicting county-level corn yields using machine learning.**
    This dashboard integrates historical agricultural data, weather patterns, and soil properties to forecast crop productivity.
    """)
    
    # Load resources
    df = load_data()
    model, scaler, feature_cols = load_model()
    
    if df is None:
        st.stop()
        
    # Sidebar Filters
    st.sidebar.header("Filters")
    
    # Year Selection
    years = sorted(df['Year'].unique(), reverse=True)
    selected_year = st.sidebar.selectbox("Select Year", years, index=0)
    
    # State Selection
    states = sorted(df['State'].unique())
    selected_state = st.sidebar.multiselect("Select State(s)", states, default=states[:3] if len(states) > 3 else states)
    
    # Filter Data
    filtered_df = df[df['Year'] == selected_year]
    if selected_state:
        filtered_df = filtered_df[filtered_df['State'].isin(selected_state)]
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    avg_yield = filtered_df['Yield_BU_ACRE'].mean()
    total_prod = filtered_df['Production_BU'].sum() / 1e6 if 'Production_BU' in filtered_df.columns else 0
    total_area = filtered_df['Area_Harvested_ACRES'].sum() / 1e6 if 'Area_Harvested_ACRES' in filtered_df.columns else 0
    
    col1.metric("Avg Yield", f"{avg_yield:.1f} Bu/Acre", delta=f"{avg_yield - df['Yield_BU_ACRE'].mean():.1f} vs All-Time")
    col2.metric("Total Production", f"{total_prod:.1f} M Bu")
    col3.metric("Harvested Area", f"{total_area:.1f} M Acres")
    col4.metric("Counties", len(filtered_df))
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🗺️ Geographic Analysis", "📊 Model Performance", "🔮 What-If Simulation"])
    
    with tab1:
        st.subheader(f"Corn Yield by County ({selected_year})")
        
        # Map
        # We need lat/lon for pydeck. Assuming they are in the dataset or we can merge them.
        # The dataset documentation mentions 'County centroids' were used.
        # Let's check if lat/lon columns exist.
        map_cols = [c for c in df.columns if 'lat' in c.lower() or 'lon' in c.lower()]
        
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            # Pydeck Map
            layer = pdk.Layer(
                "ScatterplotLayer",
                filtered_df,
                get_position=["Longitude", "Latitude"],
                get_color="[255, (1 - Yield_BU_ACRE / 250) * 255, 0, 160]",
                get_radius=20000,
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                radius_scale=1,
                radius_min_pixels=3,
                radius_max_pixels=20,
            )
            
            view_state = pdk.ViewState(
                latitude=filtered_df['Latitude'].mean(),
                longitude=filtered_df['Longitude'].mean(),
                zoom=5,
                pitch=0,
            )
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{County}, {State}\nYield: {Yield_BU_ACRE} Bu/Acre"},
                map_style="mapbox://styles/mapbox/light-v9"
            )
            st.pydeck_chart(r)
        else:
            st.warning("Latitude/Longitude columns not found for map visualization.")
            
        # Distribution Chart
        fig_dist = px.histogram(filtered_df, x="Yield_BU_ACRE", nbins=30, title="Yield Distribution", color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab2:
        st.subheader("Model Performance Analysis")
        
        # Load comparison table if exists
        comp_path = PROJECT_ROOT / 'results' / 'tables' / 'model_comparison.csv'
        if comp_path.exists():
            comp_df = pd.read_csv(comp_path)
            st.dataframe(comp_df.style.highlight_max(axis=0, subset=['Test_R2']), use_container_width=True)
            
            # Bar chart of R2
            fig_perf = px.bar(comp_df, x='Model', y='Test_R2', title="Model Accuracy (R²)", color='Test_R2', color_continuous_scale='Viridis')
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("Run model training to see performance metrics.")
            
        # Feature Importance
        imp_path = PROJECT_ROOT / 'results' / 'tables' / 'feature_importance.csv'
        if imp_path.exists():
            imp_df = pd.read_csv(imp_path).head(15)
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Top 15 Feature Importance", color='Importance')
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

    with tab3:
        st.subheader("Interactive Prediction Simulator")
        
        if model:
            col_sim1, col_sim2 = st.columns([1, 2])
            
            with col_sim1:
                st.markdown("### Adjust Conditions")
                
                # Select a baseline county
                sample_county = st.selectbox("Select Baseline County", filtered_df['County'].unique())
                baseline_data = filtered_df[filtered_df['County'] == sample_county].iloc[0]
                
                # Input sliders for key features
                # We'll pick a few important ones based on documentation
                precip_change = st.slider("Precipitation Change (%)", -50, 50, 0)
                temp_change = st.slider("Temperature Change (°C)", -5.0, 5.0, 0.0)
                
                # Create modified feature vector
                input_data = pd.DataFrame([baseline_data[feature_cols].values], columns=feature_cols)
                
                # Apply modifications (simplified logic)
                # Assuming columns like 'Precip_Total', 'Tmax_Mean' exist
                precip_cols = [c for c in feature_cols if 'precip' in c.lower()]
                temp_cols = [c for c in feature_cols if 'temp' in c.lower() or 'tmax' in c.lower()]
                
                for c in precip_cols:
                    input_data[c] = input_data[c] * (1 + precip_change/100)
                
                for c in temp_cols:
                    input_data[c] = input_data[c] + temp_change
                
                # Scale
                input_scaled = scaler.transform(input_data)
                
                # Predict
                pred_yield = model.predict(input_scaled)[0]
                baseline_yield = baseline_data['Yield_BU_ACRE'] # Or predicted baseline
                
            with col_sim2:
                st.markdown("### Prediction Result")
                
                delta = pred_yield - baseline_yield
                st.metric("Predicted Yield", f"{pred_yield:.1f} Bu/Acre", delta=f"{delta:.1f} Bu/Acre")
                
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = pred_yield,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Yield Forecast"},
                    delta = {'reference': baseline_yield},
                    gauge = {
                        'axis': {'range': [0, 300]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 100], 'color': "lightgray"},
                            {'range': [100, 200], 'color': "gray"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': baseline_yield}}))
                
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.info(f"""
                **Scenario:**
                - County: {sample_county}
                - Precipitation: {precip_change:+}%
                - Temperature: {temp_change:+.1f}°C
                """)

if __name__ == "__main__":
    main()
