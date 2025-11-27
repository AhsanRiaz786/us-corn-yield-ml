"""
Main Streamlit application entry point for US Corn Yield Prediction Dashboard.

This is the root page that serves as the landing page and navigation hub.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from app.config import PAGE_CONFIG

# Configure page
st.set_page_config(**PAGE_CONFIG)

# Custom CSS
def load_css():
    with open("app/assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


def main():
    """Main application function."""
    
    # Hero Section
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">US Corn Yield Prediction</div>
            <div class="hero-subtitle">Advanced Machine Learning for Agricultural Intelligence</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data summary
    try:
        from app.utils.data_loader import load_data, get_dataset_summary
        from app.utils.visualizations import plot_choropleth_map
        
        with st.spinner("Loading data..."):
            df = load_data()
            summary = get_dataset_summary(df)
        
        # Map Visualization (Hero Feature)
        st.subheader("National Yield Overview")
        latest_year = summary['year_range'][1]
        map_fig = plot_choropleth_map(df, latest_year)
        st.plotly_chart(map_fig, use_container_width=True)
        
        st.markdown("---")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Records</div>
                    <div class="metric-value">{summary['total_records']:,}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Counties</div>
                    <div class="metric-value">{summary['num_counties']:,}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">States</div>
                    <div class="metric-value">{summary['num_states']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Year Range</div>
                    <div class="metric-value">{summary['year_range'][0]}-{summary['year_range'][1]}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick navigation
        st.header("Navigation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Core Features")
            st.markdown("""
            - **[Yield Prediction](/Yield_Prediction)** - Make yield predictions for any county
            - **[Data Exploration](/Data_Exploration)** - Explore the dataset with interactive visualizations
            - **[Scenarios](/Scenarios)** - Analyze what-if scenarios
            """)
        
        with col2:
            st.subheader("Analysis")
            st.markdown("""
            - **[Model Insights](/Model_Insights)** - Deep dive into model performance
            - **[About](/About)** - Methodology and documentation
            """)
        
        st.markdown("---")
        
        # Model performance summary
        st.header("Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Model", "XGBoost Regressor")
        
        with col2:
            st.metric("R² Score", "0.863")
        
        with col3:
            st.metric("MAE", "11.22 BU/ACRE")
        
        st.markdown("""
        The XGBoost model achieves strong performance with an R² of 0.863 and mean absolute error of 11.22 BU/ACRE.
        This represents a significant improvement over baseline methods.
        """)
        
        st.markdown("---")
        
        # Data sources
        st.header("Data Sources")
        
        st.markdown("""
        This project integrates data from multiple authoritative sources:
        
        - **USDA NASS** - Historical corn yield, area, and production data at county level
        - **NASA POWER** - Daily weather data including temperature, precipitation, and solar radiation
        - **USDA NRCS** - Soil properties including water capacity, pH, clay content, and organic matter
        
        Data spans from 1981 to 2023 across major corn-producing states in the United States.
        """)
        
    except Exception as e:
        st.error(f"Error loading application: {str(e)}")
        st.info("Please ensure all data files and models are available in the correct directories.")


if __name__ == "__main__":
    main()

