"""
FishSense — About & Methodology
"""

import streamlit as st
import json
import os

st.set_page_config(
    page_title="About — FishSense",
    page_icon="🔬",
    layout="wide"
)

# Custom dark theme CSS (match main app)
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .card h3 { color: #c4b5fd; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0 1rem 0;'>
    <h1 style='margin: 0; font-size: 2.5rem; background: linear-gradient(135deg, #a78bfa 0%, #c4b5fd 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
        🔬 About FishSense
    </h1>
    <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem; color: #cbd5e1;'>
        Methodology, Model Details & Data Sources
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Load model metadata
metadata = None
metadata_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model_metadata.json')
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

#============================================================================
# MODEL PERFORMANCE
#============================================================================
st.markdown("<h2 style='color: #ffffff;'>📈 Model Performance</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3 style='color: #e0e0e0;'>🎯 Accuracy Metrics</h3>", unsafe_allow_html=True)
    if metadata:
        st.metric("Cross-Validated Accuracy", f"{metadata.get('cv_accuracy', metadata['test_accuracy'])*100:.1f}%")
        st.metric("Spatial CV Accuracy", f"{metadata.get('spatial_cv_accuracy', metadata['test_accuracy'])*100:.1f}%")
        st.metric("Number of Classes", len(metadata['classes']))

with col2:
    st.markdown("<h3 style='color: #e0e0e0;'>🌲 Model Details</h3>", unsafe_allow_html=True)
    if metadata:
        st.metric("Algorithm", "Random Forest")
        st.metric("Number of Trees", "100")
        st.metric("Training Samples", metadata['n_train_samples'])

with col3:
    st.markdown("<h3 style='color: #e0e0e0;'>🔬 Top Features</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color: #cbd5e1;'>
    1. <strong>Current Speed</strong> (33.9%)<br>
    2. <strong>Temperature</strong> (21.4%)<br>
    3. <strong>Temp Deviation</strong> (20.6%)
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

#============================================================================
# METHODOLOGY
#============================================================================
st.markdown("<h2 style='color: #ffffff;'>📋 Methodology</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='card'>
        <h3>🛰️ Data Sources</h3>
        <p style='color: #cbd5e1; line-height: 1.8;'>
            <strong style='color: #a78bfa;'>Primary:</strong> Google Earth Engine (MODIS SST, HYCOM Currents)<br>
            <strong style='color: #a78bfa;'>Fallback:</strong> Open-Meteo Marine API (Copernicus forecast data)<br>
            <strong style='color: #a78bfa;'>Coverage:</strong> Sri Lankan territorial waters (77°E–84°E, 4°N–12°N)<br>
            <strong style='color: #a78bfa;'>Parameters:</strong> Sea Surface Temperature, Ocean Current Speed & Direction
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
        <h3>🧪 Validation Approach</h3>
        <p style='color: #cbd5e1; line-height: 1.8;'>
            <strong style='color: #a78bfa;'>Cross-Validation:</strong> 5-fold stratified CV<br>
            <strong style='color: #a78bfa;'>Spatial CV:</strong> Testing on unseen geographic regions<br>
            <strong style='color: #a78bfa;'>Noise Simulation:</strong> 15% random label noise to prevent overfitting<br>
            <strong style='color: #a78bfa;'>Rationale:</strong> Synthetic training data requires noise to reflect real ecological variance
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='card'>
        <h3>🤖 Machine Learning Pipeline</h3>
        <p style='color: #cbd5e1; line-height: 1.8;'>
            <strong style='color: #a78bfa;'>Classification:</strong> Random Forest (100 estimators)<br>
            <strong style='color: #a78bfa;'>Clustering:</strong> K-Means (3 clusters) for zone identification<br>
            <strong style='color: #a78bfa;'>Scaling:</strong> StandardScaler for feature normalization<br>
            <strong style='color: #a78bfa;'>Features:</strong> SST, Current Speed, Current U/V, Temp Deviation, Lat/Lon
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
        <h3>🐟 Zone Classifications</h3>
        <p style='color: #cbd5e1; line-height: 1.8;'>
            <span style='color: #3b82f6;'>●</span> <strong>Nutrient-Rich Coastal</strong> — Strong currents, reef & coastal fish<br>
            <span style='color: #f59e0b;'>●</span> <strong>Seasonal Fishing</strong> — Best during monsoon transitions<br>
            <span style='color: #14b8a6;'>●</span> <strong>Deep Water Pelagic</strong> — Calm waters, tuna & pelagic species
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

#============================================================================
# TECHNOLOGY STACK
#============================================================================
st.markdown("<h2 style='color: #ffffff;'>⚙️ Technology Stack</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class='card' style='text-align: center;'>
        <div style='font-size: 2rem;'>🐍</div>
        <strong style='color: #c4b5fd;'>Python</strong>
        <p style='color: #94a3b8; font-size: 0.8rem;'>Core Language</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class='card' style='text-align: center;'>
        <div style='font-size: 2rem;'>🎯</div>
        <strong style='color: #c4b5fd;'>Scikit-Learn</strong>
        <p style='color: #94a3b8; font-size: 0.8rem;'>ML Models</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class='card' style='text-align: center;'>
        <div style='font-size: 2rem;'>🛰️</div>
        <strong style='color: #c4b5fd;'>Earth Engine</strong>
        <p style='color: #94a3b8; font-size: 0.8rem;'>Satellite Data</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class='card' style='text-align: center;'>
        <div style='font-size: 2rem;'>📊</div>
        <strong style='color: #c4b5fd;'>Streamlit</strong>
        <p style='color: #94a3b8; font-size: 0.8rem;'>Web Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

#============================================================================
# PROJECT INFO
#============================================================================
st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 20px;'>
    <p><strong style='color: #a78bfa;'>FishSense v1.0</strong></p>
    <p>Final Year Project — University of Westminster / IIT Sri Lanka</p>
    <p>Student: Nisandu Senanayake (w1871483)</p>
    <p>Supervisor: Kanishka Hewageegana</p>
    <p style='margin-top: 10px;'><a href='https://github.com/Nisandu20212028/FishSense' style='color: #a78bfa;'>📂 GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
