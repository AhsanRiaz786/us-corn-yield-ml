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
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    
    st.markdown('<div class="main-header">US Corn Yield Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Machine Learning-Based Yield Forecasting</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data summary
    try:
        from app.utils.data_loader import load_data, get_dataset_summary
        
        with st.spinner("Loading data..."):
            df = load_data()
            summary = get_dataset_summary(df)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{summary['total_records']:,}")
        
        with col2:
            st.metric("Counties", f"{summary['num_counties']:,}")
        
        with col3:
            st.metric("States", summary['num_states'])
        
        with col4:
            st.metric("Year Range", f"{summary['year_range'][0]}-{summary['year_range'][1]}")
        
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

