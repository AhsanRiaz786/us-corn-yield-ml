"""
Model Insights Page - Deep dive into model performance and analysis.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.config import MODEL_METADATA, FEATURE_CATEGORIES
from app.utils.model_loader import get_available_models
from pathlib import Path
import json


st.set_page_config(page_title="Model Insights", layout="wide")
st.title("Model Performance & Insights")

# Model comparison
st.header("Model Comparison")

comparison_data = []
comparison_data = []
available_models = get_available_models()

for model_name, metadata in MODEL_METADATA.items():
    if model_name in available_models:
        comparison_data.append({
        'Model': metadata['name'],
        'R² Score': metadata['r2'],
        'MAE (BU/ACRE)': metadata['mae'],
        'RMSE (BU/ACRE)': metadata['rmse']
    })

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Performance visualization
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('R² Score', 'MAE (BU/ACRE)', 'RMSE (BU/ACRE)'),
    horizontal_spacing=0.1
)

models = comparison_df['Model'].values
r2_scores = comparison_df['R² Score'].values
mae_scores = comparison_df['MAE (BU/ACRE)'].values
rmse_scores = comparison_df['RMSE (BU/ACRE)'].values

fig.add_trace(
    go.Bar(x=models, y=r2_scores, name='R²', marker_color='#1f77b4'),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=models, y=mae_scores, name='MAE', marker_color='#2ca02c'),
    row=1, col=2
)

fig.add_trace(
    go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color='#ff7f0e'),
    row=1, col=3
)

fig.update_layout(
    title='Model Performance Comparison',
    showlegend=False,
    height=400
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Feature importance
st.header("Feature Importance")

# Try to load feature importance data
feature_importance_path = Path("results/tables/feature_importance.csv")

if feature_importance_path.exists():
    try:
        feature_importance_df = pd.read_csv(feature_importance_path)
        
        # Top features
        top_n = st.slider("Number of Top Features", 10, 30, 15)
        
        top_features = feature_importance_df.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_features['Importance'].values,
            y=top_features['Feature'].values,
            orientation='h',
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Most Important Features',
            xaxis_title='Importance (Gain)',
            yaxis_title='Feature',
            height=max(400, top_n * 25),
            yaxis=dict(autorange='reversed'),
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance by category
        st.subheader("Feature Importance by Category")
        
        category_importance = {}
        for category, features in FEATURE_CATEGORIES.items():
            category_df = feature_importance_df[feature_importance_df['Feature'].isin(features)]
            if len(category_df) > 0:
                category_importance[category] = category_df['Importance'].sum()
        
        if category_importance:
            cat_df = pd.DataFrame(list(category_importance.items()), columns=['Category', 'Total Importance'])
            cat_df = cat_df.sort_values('Total Importance', ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cat_df['Total Importance'].values,
                y=cat_df['Category'].values,
                orientation='h',
                marker_color='#2ca02c'
            ))
            
            fig.update_layout(
                title='Cumulative Feature Importance by Category',
                xaxis_title='Total Importance',
                yaxis_title='Category',
                height=400,
                yaxis=dict(autorange='reversed'),
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Could not load feature importance data: {str(e)}")
else:
    st.info("Feature importance data not available. Run model training to generate this analysis.")

st.markdown("---")

# Key findings
st.header("Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Best Model")
    st.write("**XGBoost Regressor** achieves the best performance:")
    st.write(f"- R² Score: {MODEL_METADATA['xgboost']['r2']:.3f}")
    st.write(f"- Mean Absolute Error: {MODEL_METADATA['xgboost']['mae']:.2f} BU/ACRE")
    st.write(f"- Root Mean Squared Error: {MODEL_METADATA['xgboost']['rmse']:.2f} BU/ACRE")

with col2:
    st.subheader("Performance Highlights")
    st.write("- Historical yield features (lags) are the strongest predictors")
    st.write("- Weather features, especially heat stress, significantly impact predictions")
    st.write("- Soil properties provide valuable context but less predictive power")
    st.write("- Model performs best on high-yield regions and stable weather years")

st.markdown("---")

# Model details
st.header("Model Details")

selected_model_for_details = st.selectbox(
    "Select Model for Details",
    available_models,
    format_func=lambda x: MODEL_METADATA[x]['name']
)

model_info = MODEL_METADATA[selected_model_for_details]

st.write(f"**{model_info['name']}**")
st.write(model_info['description'])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R² Score", f"{model_info['r2']:.3f}")
with col2:
    st.metric("MAE", f"{model_info['mae']:.2f} BU/ACRE")
with col3:
    st.metric("RMSE", f"{model_info['rmse']:.2f} BU/ACRE")

