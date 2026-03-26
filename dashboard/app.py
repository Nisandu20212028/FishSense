"""
FishSense: Interactive Dashboard with ML Model Integration
AI-powered fishing zone prediction system
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static, st_folium
import joblib
import json
import os
import requests
from datetime import datetime, timedelta

# Try to import Google Earth Engine
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

#============================================================================
# PAGE CONFIGURATION
#============================================================================
st.set_page_config(
    page_title="FishSense - AI Fishing Predictions",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar visible
)

#============================================================================
# CUSTOM CSS FOR GLASSMORPHIC PREMIUM THEME
#============================================================================
st.markdown("""
<style>
    /* Import Poppins Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Main App Background - Purple Gradient */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #e0e0e0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    
    /* Navigation Header - Glassmorphic */
    .nav-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0;
        padding: 1rem 2rem;
        margin: -1rem -2rem 2rem -2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Dashboard Header */
    .dashboard-header {
  text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        margin-bottom: 1.5rem;
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 60px rgba(102, 126, 234, 0.5);
        letter-spacing: -0.5px;
    }
    
    /* Top Metrics Container - Glassmorphic Cards */
    .top-metrics-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .top-metric-card {
        flex: 1;
        min-width: 220px;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 20px;
        padding: 1.75rem 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4),
                    0 0 0 1px rgba(255, 255, 255, 0.05) inset;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .top-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .top-metric-card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4),
                    0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .top-metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
        filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.5));
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #a78bfa;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0.5rem 0;
        line-height: 1;
        display: block;
    }
    
    .metric-subtitle {
        font-size: 0.75rem;
        color: #c4b5fd !important;
        opacity: 0.9;
        margin-top: 0.5rem;
        display: block;
    }
    
    /* Glassmorphic Section Cards */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, transparent 100%);
        padding-left: 0.75rem;
        margin-left: -0.75rem;
        margin-right: -0.75rem;
        padding-right: 0.75rem;
    }
    
    /* Map Container - Glassmorphic */
    .map-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    
    .map-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    /* Input Fields - Modern Glass Style */
    .stNumberInput input, .stTextInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput input:focus, .stTextInput input:focus {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Sliders - Purple Gradient */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        height: 8px !important;
        border-radius: 10px !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        width: 24px !important;
        height: 24px !important;
        border-radius: 50% !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        border: 3px solid #667eea !important;
    }
    
    /* Buttons - Gradient with Glow */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4),
                    0 0 0 1px rgba(255, 255, 255, 0.1) inset !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: none !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 16px 48px rgba(102, 126, 234, 0.6),
                    0 0 0 1px rgba(255, 255, 255, 0.2) inset !important;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(0.98) !important;
    }
    
    /* Model Info Table */
    .model-info-table {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 0;
        overflow: hidden;
    }
    
    .model-info-row {
        display: flex;
        justify-content: space-between;
        padding: 1rem 1.25rem;
        border-bottom: 1px solid rgba(102, 126, 234, 0.1);
        transition: background 0.2s ease;
    }
    
    .model-info-row:last-child {
        border-bottom: none;
    }
    
    .model-info-row:hover {
        background: rgba(255, 255, 255, 0.05);
    }
    
    .model-info-label {
        color: #a78bfa;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .model-info-value {
        color: #ffffff;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    /* Prediction Cards with Color-Coded Glow */
    .prediction-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        border: 2px solid;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: all 0.4s ease;
    }
    
    .prediction-card.prediction-high {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border-color: #10b981;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
    }
    
    .prediction-card.prediction-high:hover {
        box-shadow: 0 12px 48px rgba(16, 185, 129, 0.5);
        transform: translateY(-4px);
    }
    
    .prediction-card.prediction-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.05) 100%);
        border-color: #f59e0b;
        box-shadow: 0 8px 32px rgba(245, 158, 11, 0.3);
    }
    
    .prediction-card.prediction-medium:hover {
        box-shadow: 0 12px 48px rgba(245, 158, 11, 0.5);
        transform: translateY(-4px);
    }
    
    .prediction-card.prediction-low {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border-color: #ef4444;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
    }
    
    .prediction-card.prediction-low:hover {
        box-shadow: 0 12px 48px rgba(239, 68, 68, 0.5);
        transform: translateY(-4px);
    }
    
    .prediction-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        animation: float 3s ease-in-out infinite;
    }
    
    .prediction-title {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.75rem;
    }
    
    .prediction-description {
        font-size: 1rem;
        color: #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.3);
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #ffffff;
        border: 1px solid rgba(102, 126, 234, 0.4);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* Animations */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
    }
    
    @keyframes pulse-glow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.03);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Labels and Text */
    label, .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    .stMarkdown strong {
        color: #ffffff !important;
    }
    
    /* Caption text */
    .stCaptionContainer {
        color: #a78bfa !important;
    }
    
    /* Number Input - Force dark text for visibility */
    [data-testid="stNumberInput"] input {
        color: #1a1a2e !important;
        background-color: #ffffff !important;
    }
    
    /* Sidebar Number Input - Force dark text */
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        color: #1a1a2e !important;
        background-color: #ffffff !important;
    }
    
    /* Metric Values - Make visible */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem !important;
    }
    
    /* Metric Labels */
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }
    
    /* Sidebar Metrics */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #a78bfa !important;
    }
    
    /* Slider Values - Make visible */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]::after {
        background-color: #ffffff !important;
        border: 2px solid #8b5cf6 !important;
    }
    
    /* Slider value display - FORCE white text */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Force ALL text inside sliders to be white */
    .stSlider * {
        color: #ffffff !important;
    }
    
    /* Slider min/max numbers at track ends */
    .stSlider small {
        color: #ffffff !important;
    }
    
    /* Any div text in slider */
    .stSlider div {
        color: #ffffff !important;
    }
    
    /* Slider thumb label */
    [data-baseweb="slider"] div[role="slider"] {
        background: #ffffff !important;
    }
    
    /* Input Labels - Make visible */
    label {
        color: #ffffff !important;
    }
    
    /* Slider Labels */
    .stSlider label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Number Input Labels */
    [data-testid="stNumberInput"] label {
        color: #ffffff !important;
    }
    
    /* Help text / tooltips */
    .stTooltipIcon {
        color: #cbd5e1 !important;
    }
    
    /* Expander header */
    [data-testid="stExpander"] summary {
        color: #cbd5e1 !important;
        background: rgba(30, 41, 59, 0.7) !important;
    }
    /* Expander container - dark theme */
    [data-testid="stExpander"],
    [data-testid="stExpander"] > div,
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] details > div {
        background: rgba(30, 41, 59, 0.5) !important;
        border-color: rgba(139, 92, 246, 0.2) !important;
        border-radius: 8px !important;
        color: #cbd5e1 !important;
    }
    /* Expander content area */
    [data-testid="stExpanderDetails"],
    [data-testid="stExpanderDetails"] > div,
    [data-testid="stExpanderDetails"] > div > div {
        background: transparent !important;
        background-color: transparent !important;
        color: #cbd5e1 !important;
    }
    
    /* Hide stray keyboard hint text that leaks from expanders */
    .element-container p:empty,
    div[data-testid="stExpanderDetails"] + div:empty {
        display: none !important;
    }
    
    /* Slider current value display box */
    .stSlider [data-testid="stThumbValue"] {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 2px solid #8b5cf6 !important;
        font-weight: 600 !important;
        padding: 0.3rem 0.6rem !important;
        border-radius: 8px !important;
    }
    
    /* Info icon tooltip */
    .stTooltipIcon svg {
        fill: #cbd5e1 !important;
    }
    
    button[kind="icon"] {
        color: #cbd5e1 !important;
    }
    
    
    
    
    
    /* Tip Box */
    .tip-box {
        background: rgba(139, 92, 246, 0.1);
        border-left: 4px solid #8b5cf6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Info boxes - Make text white */
    .stAlert {
        color: #ffffff !important;
    }
    
    [data-testid="stAlert"] {
        color: #ffffff !important;
    }
    
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] strong {
        color: #ffffff !important;
    }
    
    
    /* Prediction styling */
    .prediction-high, .prediction-medium, .prediction-low {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        border: 2px solid;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: all 0.4s ease;
    }
    
    .prediction-subtitle {
        font-size: 0.9rem;
        color: #c4b5fd;
        margin-top: 0.5rem;
    }
    
    /* Hide widget key labels that appear as text */
    [data-testid="stSidebar"] .stDownloadButton::before,
    [data-testid="stSidebar"] [class*="key"]::before {
        content: none !important;
    }
    
    /* Hide any standalone text containing "key" at start of sidebar */
    [data-testid="stSidebar"] \u003e div \u003e div:first-child \u003e div:first-child {
        display: none;
    }
    
    /* Streamlit Sidebar - Glassmorphic Style */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.95) 100%) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.3) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    
    /* Sidebar content */
    .css-1d391kg, [data-testid="stSidebarNav"] {
        background: transparent !important;
    }
    
    /* Sidebar headings */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .element-container {
        color: #ffffff !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar links */
    [data-testid="stSidebar"] a {
        color: #a78bfa !important;
        transition: color 0.2s ease;
    }
    
    [data-testid="stSidebar"] a:hover {
        color: #c4b5fd !important;
    }
    
    /* Sidebar collapse button */
    [data-testid="collapsedControl"] {
        background: rgba(102, 126, 234, 0.2) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        color: #ffffff !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background: rgba(102, 126, 234, 0.4) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Sidebar bullet points and lists */
    [data-testid="stSidebar"] ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    [data-testid="stSidebar"] li::before {
        content: "▸ ";
        color: #667eea;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    /* Hide keyboard shortcut labels and widget keys - AGGRESSIVE */
    [data-testid="stSidebar"] label[for*="key"],
    [data-testid="stSidebar"] label:contains("keyboard"),
    [data-testid="stSidebar"] div:contains("keyboard_double") {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        overflow: hidden !important;
    }
    
    /* Hide expander summary labels with key prefix */
    [data-testid="stSidebar"] summary label,
    [data-testid="stSidebar"] [data-testid="stExpander"] label {
        display: none !important;
    }
    
    /* Hide any direct text nodes starting with key or keyboard */
    [data-testid="stSidebar"] [class*="Label"] {
        display: none !important;
    }
    
    /* Very specific: hide first div in sidebar (often contains keyboard_double) */
    [data-testid="stSidebar"] > div > div:first-child > div:first-child:not([data-testid]) {
        display: none !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* CRITICAL: Hide keyboard_double text at top of sidebar */
    [data-testid="stSidebar"] > div:first-child > div:first-child > div:first-child {
        font-size: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        line-height: 0 !important;
    }
    
    /* Hide any stray label elements in sidebar */
    [data-testid="stSidebar"] label[class*="css-"]:empty,
    [data-testid="stSidebar"] label[class*="css-"]:not(:has(span)):not(:has(div)) {
    }
</style>

<script>
    // Remove keyboard shortcut hints that appear in sidebar
    const observer = new MutationObserver(function(mutations) {
        // Remove keyboard_double and keyboard_arrow texts
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            // Find and hide all elements containing keyboard text
            const allElements = sidebar.querySelectorAll('*');
            allElements.forEach(el => {
                if (el.textContent.includes('keyboard_') || 
                    el.textContent.trim().startsWith('keyboard')) {
                    // Check if this is just a text node with keyboard hint
                    const text = el.textContent.trim();
                    if (text === 'keyboard_double' || 
                        text === 'keyboard_arrow_down' || 
                        text.startsWith('keyboard_')) {
                        el.style.display = 'none';
                        el.style.visibility = 'hidden';
                        el.style.fontSize = '0';
                        el.style.height = '0';
                        el.style.width = '0';
                        el.style.overflow = 'hidden';
                    }
                }
            });
            
            // Also target specific keyboard hint spans
            const keyboardSpans = sidebar.querySelectorAll('span');
            keyboardSpans.forEach(span => {
                const text = span.textContent.trim();
                if (text.startsWith('keyboard') || text.includes('key_')) {
                    span.style.display = 'none';
                }
            });
        }
    });
    
    // Start observing
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        characterData: true
    });
    
    // Run immediately on load
    setTimeout(() => {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            const allElements = sidebar.querySelectorAll('*');
            allElements.forEach(el => {
                const text = el.textContent.trim();
                if (text === 'keyboard_double' || 
                    text === 'keyboard_arrow_down' || 
                    text.startsWith('keyboard_')) {
                    el.style.display = 'none';
                }
            });
        }
    }, 100);
</script>

""", unsafe_allow_html=True)

# Inject JavaScript to hide keyboard hints
components.html("""
<script>
(function() {
    function hideKeyboardHints() {
        try {
            const parentDoc = window.parent.document;
            
            // Target ALL elements in the document, not just sidebar
            const allElements = parentDoc.querySelectorAll('*');
            allElements.forEach(el => {
                // Only check leaf nodes (no children elements, just text)
                if (el.children.length === 0) {
                    const text = (el.textContent || '').trim();
                    if (text.startsWith('keyboard') || text.startsWith('key_') ||
                        text === 'keyboard_arrow_down' || text === 'keyboard_double') {
                        el.style.display = 'none';
                        el.style.fontSize = '0';
                        el.style.height = '0';
                        el.style.overflow = 'hidden';
                    }
                }
            });
            
            // Also specifically target expander summary labels with 'key' prefix
            const summaries = parentDoc.querySelectorAll('[data-testid="stExpander"] summary');
            summaries.forEach(summary => {
                summary.childNodes.forEach(node => {
                    if (node.nodeType === 3) { // Text node
                        const t = node.textContent.trim();
                        if (t.startsWith('key') || t.startsWith('keyboard')) {
                            node.textContent = '';
                        }
                    }
                });
            });
        } catch(e) {}
    }
    
    setInterval(hideKeyboardHints, 50);
})();
</script>
""", height=0)

#============================================================================

#============================================================================
# GOOGLE EARTH ENGINE INITIALIZATION
#============================================================================
gee_init_error = None  # Stores GEE init error for UI display

def initialize_gee():
    """Initialize Google Earth Engine (supports local auth & Streamlit Cloud)"""
    global gee_init_error
    if not GEE_AVAILABLE:
        gee_init_error = "earthengine-api not installed"
        return False
    try:
        # Check if Streamlit secrets has GEE service account
        has_secrets = False
        try:
            if hasattr(st, 'secrets') and 'gee_service_account' in st.secrets:
                has_secrets = True
        except Exception as se:
            gee_init_error = f"Secrets check error: {str(se)[:200]}"
        
        if has_secrets:
            import tempfile
            
            # Convert Streamlit's AttrDict to a regular dict
            secret_dict = {}
            for key in st.secrets['gee_service_account']:
                secret_dict[key] = str(st.secrets['gee_service_account'][key])
            
            # Write service account JSON to temp file (most reliable method)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(secret_dict, f)
                key_file_path = f.name
            
            service_account = secret_dict['client_email']
            credentials = ee.ServiceAccountCredentials(
                service_account, key_file=key_file_path
            )
            ee.Initialize(credentials=credentials, project='fishsense-480120')
            
            # Clean up temp file
            try:
                os.remove(key_file_path)
            except Exception:
                pass
        else:
            if not gee_init_error:
                gee_init_error = "No service account in secrets (local auth attempted)"
            ee.Initialize(project='fishsense-480120')
        
        # Test if it works
        ee.Number(1).getInfo()
        gee_init_error = None  # Clear error on success
        return True
    except Exception as e:
        gee_init_error = str(e)[:500]
        return False

gee_initialized = initialize_gee() if GEE_AVAILABLE else False

#============================================================================
# OPEN-METEO API INTEGRATION (FALLBACK FOR FRESH CURRENTS)
#============================================================================
def fetch_open_meteo_currents(lon, lat):
    """
    Fetch near-real-time ocean currents from Open-Meteo Marine API.
    Uses European Copernicus Marine Service backend (updated hourly).
    """
    try:
        url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&current=ocean_current_velocity,ocean_current_direction"
        
        # Fast 10-second timeout, API usually responds in <1s
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'current' in data:
                # Speed is in km/h, convert to m/s
                speed_kmh = data['current']['ocean_current_velocity']
                direction_deg = data['current']['ocean_current_direction']
                
                if speed_kmh is None or direction_deg is None:
                    return None
                    
                speed_ms = float(speed_kmh) / 3.6
                
                # Convert polar to Cartesian (u, v)
                # Oceanographic standard: direction TO (clockwise from North)
                import numpy as np
                angle_rad = np.radians(90 - direction_deg)
                u_m_s = speed_ms * np.cos(angle_rad)
                v_m_s = speed_ms * np.sin(angle_rad)
                
                # Format date
                time_str = data['current']['time']
                om_date = datetime.strptime(time_str[:10], '%Y-%m-%d')
                
                return {
                    'current_speed': float(speed_ms),
                    'u': float(u_m_s),
                    'v': float(v_m_s),
                    'date': om_date
                }
        return None
    except Exception as e:
        print(f"Open-Meteo Fetch Error: {str(e)}")
        return None

def fetch_open_meteo_sst(lon, lat):
    """
    Fetch near-real-time Sea Surface Temperature from Open-Meteo Weather API.
    Uses the standard forecast model over ocean coordinates.
    Returns temperature in Celsius.
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&timezone=auto"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'hourly' in data:
                temps = data['hourly']['temperature_2m']
                dates = data['hourly']['time']
                
                # Find the current hour's temperature
                now_str = datetime.now().strftime('%Y-%m-%dT%H:00')
                
                try:
                    idx = dates.index(now_str)
                    current_temp = temps[idx]
                    if current_temp is not None:
                        return {
                            'sst': float(current_temp),
                            'date': datetime.now()
                        }
                except ValueError:
                    pass
                
                # Fallback: grab the most recent valid temperature
                for i in range(len(temps) - 1, -1, -1):
                    if temps[i] is not None:
                        om_date = datetime.strptime(dates[i][:10], '%Y-%m-%d')
                        return {
                            'sst': float(temps[i]),
                            'date': om_date
                        }
        return None
    except Exception as e:
        print(f"Open-Meteo SST Fetch Error: {str(e)}")
        return None

#============================================================================
# REAL-TIME DATA FETCHING FUNCTION
#============================================================================
def fetch_realtime_data(lat, lon):
    """Fetch real-time SST and ocean current data from Google Earth Engine"""
    if not gee_initialized:
        return None, "Google Earth Engine not initialized"
    
    try:
        # Create point geometry
        point = ee.Geometry.Point([lon, lat])
        
        # Try current dates first, then fall back to historical data if unavailable
        # This handles the case where satellite datasets haven't caught up to current dates
        # Satellite datasets (HYCOM, MODIS) typically lag 1-3 months behind real-time
        # Try progressively older date ranges until we find available data
        now = datetime.now()
        date_ranges_to_try = [
            # Try most recent data first
            (now - timedelta(days=90), now, "current"),
            (now - timedelta(days=180), now - timedelta(days=60), "recent"),
            (now - timedelta(days=365), now - timedelta(days=150), "last year"),
        ]
        # Add quarterly fallbacks going back up to 2 years
        for months_back in range(3, 24, 3):
            end = now - timedelta(days=months_back * 30)
            start = end - timedelta(days=90)
            label = f"{start.strftime('%Y %b')}-{end.strftime('%b')}"
            date_ranges_to_try.append((start, end, label))
        
        data_source_period = None
        sst_image = None
        sst_band = None
        current_image = None
        
        # Try each date range until we find data
        for start_date, end_date, period_name in date_ranges_to_try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Check if HYCOM has data for this period
            try:
                test_collection = ee.ImageCollection('HYCOM/sea_water_velocity') \
                    .filterDate(start_str, end_str) \
                    .filterBounds(point)
                
                test_count = test_collection.size().getInfo()
                
                if test_count > 0:
                    # Found data! Use this date range
                    data_source_period = period_name
                    break
            except Exception:
                continue
        
        if not data_source_period:
            return None, "No satellite data available for this location. Try a different location or use Manual Input mode."
        
        # Format dates for GEE
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        
        # Try MODIS Aqua SST first (better daily coverage)
        sst_image = None
        sst_band = None
        try:
            sst_collection = ee.ImageCollection('NASA/OCEANDATA/MODIS-Aqua/L3SMI') \
                .filterDate(start_str, end_str) \
                .filterBounds(point) \
                .select('sst')
            
            sst_count = sst_collection.size().getInfo()
            if sst_count > 0:
                sst_image = sst_collection.sort('system:time_start', False).first()
                sst_band = 'sst'
            else:
                # Fallback to NOAA CDR
                sst_collection = ee.ImageCollection('NOAA/CDR/SST_WHOI/V2') \
                    .filterDate(start_str, end_str) \
                    .filterBounds(point)
                
                sst_count = sst_collection.size().getInfo()
                if sst_count > 0:
                    sst_image = sst_collection.sort('system:time_start', False).first()
                    sst_band = 'sea_surface_temperature'
                # If no SST data, sst_image remains None - we'll handle this later
        except Exception:
            # If MODIS fails, try NOAA CDR
            try:
                sst_collection = ee.ImageCollection('NOAA/CDR/SST_WHOI/V2') \
                    .filterDate(start_str, end_str) \
                    .filterBounds(point)
                
                sst_count = sst_collection.size().getInfo()
                if sst_count > 0:
                    sst_image = sst_collection.sort('system:time_start', False).first()
                    sst_band = 'sea_surface_temperature'
            except Exception:
                pass  # SST will remain None
        
        # Fetch Ocean Currents from HYCOM
        current_collection = ee.ImageCollection('HYCOM/sea_water_velocity') \
            .filterDate(start_str, end_str) \
            .filterBounds(point)
        
        # Check if collection has data
        current_count = current_collection.size().getInfo()
        if current_count == 0:
            return None, "No ocean current data available for this location in the last 90 days"
        
        current_image = current_collection.sort('system:time_start', False).first()
        
        # Sample SST (only if we have an SST image)
        sst_sample = None
        if sst_image and sst_band:
            sst_sample = sst_image.select(sst_band).sample(
                region=point,
                scale=4000,  # MODIS resolution
                numPixels=1
            ).getInfo()
        
        # Sample Currents
        current_sample = current_image.select(['velocity_u_0', 'velocity_v_0']).sample(
            region=point,
            scale=10000,
            numPixels=1
        ).getInfo()
        
        # Extract values
        sst_celsius = None
        if sst_sample and sst_sample['features'] and len(sst_sample['features']) > 0:
            sst_value = sst_sample['features'][0]['properties'].get(sst_band)
            # MODIS SST is in Celsius, NOAA is in Kelvin
            if sst_band == 'sst':  # MODIS
                sst_celsius = sst_value if sst_value else None
            else:  # NOAA
                sst_celsius = sst_value - 273.15 if sst_value else None
        
        if current_sample and current_sample['features']:
            u_cm_s = current_sample['features'][0]['properties'].get('velocity_u_0')
            v_cm_s = current_sample['features'][0]['properties'].get('velocity_v_0')
            
            if u_cm_s is not None and v_cm_s is not None:
                u_m_s = u_cm_s / 100.0
                v_m_s = v_cm_s / 100.0
                current_speed = np.sqrt(u_m_s**2 + v_m_s**2)
            else:
                u_m_s, v_m_s, current_speed = None, None, None
        else:
            u_m_s, v_m_s, current_speed = None, None, None
        # Get data timestamp from GEE
        gee_currents_date = None
        if current_sample and current_sample['features']:
            timestamp_ms = current_image.get('system:time_start').getInfo()
            data_date = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d')
            gee_currents_date = datetime.fromtimestamp(timestamp_ms / 1000)
        else:
            data_date = "Unknown"
        
        currents_source = "GEE (HYCOM)"
        
        # --- OPEN-METEO FALLBACK LOGIC ---
        # If GEE has no currents, or if GEE data is older than 30 days, try Open-Meteo
        if gee_currents_date is None or (datetime.now() - gee_currents_date).days > 30:
            om_data = fetch_open_meteo_currents(lon, lat)
            
            if om_data is not None:
                # If Open-Meteo succeeded, check if it is newer than GEE (or GEE missing)
                if gee_currents_date is None or om_data['date'] > gee_currents_date:
                    u_m_s = om_data['u']
                    v_m_s = om_data['v']
                    current_speed = om_data['current_speed']
                    data_date = om_data['date'].strftime('%Y-%m-%d')
                    currents_source = "Open-Meteo Marine (Copernicus)"
        # -----------------------------
        
        # --- OPEN-METEO SST FALLBACK LOGIC ---
        # If GEE has no SST data, try Open-Meteo before falling back to seasonal avg
        sst_source = "GEE (MODIS)" if sst_celsius is not None else None
        if sst_celsius is None:
            om_sst_data = fetch_open_meteo_sst(lon, lat)
            if om_sst_data is not None:
                sst_celsius = om_sst_data['sst']
                sst_source = "Open-Meteo (Forecast)"
        # ------------------------------------
        
        # Handle partial data scenarios (important for research functionality)
        if sst_celsius is None and current_speed is not None:
            # We have currents but no SST from GEE or Open-Meteo - last resort seasonal avg
            month = datetime.now().month
            # Seasonal SST estimates based on Sri Lankan coastal patterns
            if month in [12, 1, 2]:  # Winter
                sst_celsius = 27.5
            elif month in [3, 4, 5]:  # Spring  
                sst_celsius = 29.0
            elif month in [6, 7, 8]:  # Summer
                sst_celsius = 28.5
            else:  # Fall
                sst_celsius = 28.0
            sst_source = "Seasonal Estimate"
            
            return {
                'sst': sst_celsius,
                'current_speed': current_speed,
                'current_u': u_m_s,
                'current_v': v_m_s,
                'data_date': data_date,
                'success': True,
                'note': f'SST from {sst_source}, Currents from {currents_source} ({data_date})'
            }, None
        
        elif sst_celsius is not None and current_speed is None:
            # We have SST but no currents - use moderate current estimate
            current_speed = 2.0
            u_m_s = 1.0
            v_m_s = 1.5
            
            return {
                'sst': sst_celsius,
                'current_speed': current_speed,
                'current_u': u_m_s,
                'current_v': v_m_s,
                'data_date': data_date,
                'success': True,
                'note': f'SST from {sst_source}, Currents estimated'
            }, None
        
        elif sst_celsius is None and current_speed is None:
            return None, "No satellite data available for this location. Try a different location or use Manual Input mode."
        
        # Both values available from satellite
        return {
            'sst': sst_celsius,
            'current_speed': current_speed,
            'current_u': u_m_s,
            'current_v': v_m_s,
            'data_date': data_date,
            'success': True,
            'note': f'SST from {sst_source}, Currents from {currents_source} ({data_date})'
        }, None
        
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

#============================================================================
# LOAD MODEL AND DATA
#============================================================================
@st.cache_resource
def load_model():
    """Load the trained Random Forest model and scaler"""
    try:
        model = joblib.load('models/fishsense_rf_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, scaler, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Make sure you have trained the model first by running: python scripts/train_model.py")
        return None, None, None

@st.cache_resource
def load_kmeans_model():
    """Load the K-Means clustering model for zone visualization"""
    try:
        kmeans_model = joblib.load('models/fishsense_kmeans_model.pkl')
        with open('models/kmeans_metadata.json', 'r') as f:
            kmeans_meta = json.load(f)
        return kmeans_model, kmeans_meta
    except Exception:
        return None, None

@st.cache_data
def generate_kmeans_zones(_rf_model, _scaler, _kmeans_model):
    """
    Hybrid approach: K-Means clusters ocean zones, RF predicts fishing potential.
    Generates a grid of points across Sri Lankan waters and classifies each cell.
    """
    if _rf_model is None or _scaler is None or _kmeans_model is None:
        return []
    
    import numpy as np
    
    # Grid across Sri Lankan coastal waters
    lat_range = np.arange(5.0, 10.5, 0.5)
    lon_range = np.arange(78.0, 83.0, 0.5)
    
    zones = []
    for lat_val in lat_range:
        for lon_val in lon_range:
            # Skip land area (rough Sri Lanka bounding box)
            if 79.5 <= lon_val <= 81.9 and 5.9 <= lat_val <= 9.9:
                continue
            
            # Realistic spatially-varying ocean conditions
            # SST: warmer near equator and coast, cooler further out & north
            dist_from_coast = min(abs(lon_val - 79.5), abs(lon_val - 82.0))
            base_sst = 29.5 - (lat_val - 6.0) * 0.4 - dist_from_coast * 0.8
            
            # Currents: stronger on west coast and in straits, weaker in deep east
            if lon_val < 79.5:  # West coast
                current_speed = 2.0 + (9.0 - lat_val) * 0.15
            elif lon_val > 82.0:  # Far east - calmer
                current_speed = 0.4 + abs(lat_val - 7.5) * 0.1
            else:  # North/south of island
                current_speed = 1.2 + abs(lon_val - 80.5) * 0.3
            
            # Decompose into components with directional variation
            import math
            angle = math.radians(45 + lat_val * 10 + lon_val * 5)
            current_u = current_speed * math.cos(angle)
            current_v = current_speed * math.sin(angle)
            
            temp_dev = base_sst - 28.0
            lon_norm = (lon_val - 79.5) / (82.0 - 79.5)
            lat_norm = (lat_val - 5.9) / (9.9 - 5.9)
            
            features = np.array([[base_sst, current_speed, current_u, current_v, 
                                  temp_dev, lon_norm, lat_norm]])
            
            # K-Means assigns cluster
            cluster = _kmeans_model.predict(_scaler.transform(features))[0]
            
            # RF predicts fishing potential
            prediction = _rf_model.predict(_scaler.transform(features))[0]
            
            zones.append({
                'lat': float(lat_val),
                'lon': float(lon_val),
                'cluster': int(cluster),
                'prediction': prediction,
                'sst': float(base_sst),
                'current': float(current_speed)
            })
    
    return zones

# Load models
model, scaler, metadata = load_model()
kmeans_model, kmeans_meta = load_kmeans_model()

#============================================================================
# HEADER
#============================================================================
# Navigation Bar
st.markdown("""
<div style='background: rgba(30, 41, 59, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(139, 92, 246, 0.2); border-radius: 12px; padding: 0.75rem 1.5rem; margin-bottom: 2rem;'>
    <div style='display: flex; align-items: center; gap: 0.75rem;'>
        <span style='font-size: 1.75rem;'>🐟</span>
        <div>
            <h2 style='margin: 0; font-size: 1.25rem; color: #ffffff; font-weight: 600;'>FishSense</h2>
            <p style='margin: 0; font-size: 0.7rem; color: #a78bfa; letter-spacing: 1px; text-transform: uppercase;'>Intelligent Ocean Prediction System</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Title
st.markdown("""
<div style='text-align: center; padding: 2rem 0 1rem 0;'>
    <h1 style='margin: 0; font-size: 3rem; background: linear-gradient(135deg, #a78bfa 0%, #c4b5fd 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>
        🐟 FishSense
    </h1>
    <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem; color: #cbd5e1; font-weight: 400;'>
        AI-Powered Ocean Health & Fishing Zone Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# Welcome tip
st.markdown("""
<div class="tip-box">
    <strong>👋 Welcome!</strong> Select your location and let our AI predict the best fishing zones based on ocean conditions.
    Perfect for fishermen, students, and researchers!
</div>
""", unsafe_allow_html=True)

st.markdown("---")


#============================================================================
# MODE SELECTION
#============================================================================
st.markdown("### 🎛️ How would you like to predict?")
col1, col2 = st.columns(2)
with col1:
    quick_mode = st.button("🎯 Quick Prediction", type="primary", use_container_width=True)
with col2:
    live_mode = st.button("🛰️ Live Ocean Data", use_container_width=True)

# Store mode in session state
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "Live Ocean Data"

if quick_mode:
    st.session_state.input_mode = "Quick Prediction"
elif live_mode:
    st.session_state.input_mode = "Live Ocean Data"

input_mode = st.session_state.input_mode

# Mode description
if input_mode == "Quick Prediction":
    st.info("💡 **Quick Prediction**: Perfect when you know the ocean conditions or want to test different scenarios.")
else:
    st.info("💡 **Live Ocean Data**: Automatically fetches real satellite data for your location. May use historical data if current data unavailable.")

st.markdown("---")

has_prediction = st.session_state.get('last_prediction') is not None

# Initialize default values (will be overridden by mode-specific inputs)
sst = 28.5
current_speed = 2.0
current_u = 1.0
current_v = 1.0

if input_mode == "Quick Prediction":
    # Manual input mode
    st.markdown("<h3 style='color: #ffffff;'>🌊 Ocean Conditions</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        sst = st.slider(
            "🌡️ Water Temperature (°C)",
            min_value=0.0,
            max_value=40.0,
            value=28.5,
            step=0.1,
            help="Optimal fishing temp is usually 20-30°C"
        )
    
    with col2:
        current_speed = st.slider(
            "🌊 Ocean Current Strength (m/s)",
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.1,
            help="Moderate currents (2-4 m/s) are usually best for fishing"
        )
    
    # Advanced parameters
    with st.expander("🔬 Advanced Details (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            current_u = st.slider(
                "Current East-West (m/s)",
                min_value=-5.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Positive = Eastward flow"
            )
        
        with col2:
            current_v = st.slider(
                "Current North-South (m/s)",
                min_value=-5.0,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="Positive = Northward flow"
            )
    st.markdown("---")


#============================================================================
# LOCATION INPUT (collapsible)
#============================================================================
with st.expander("📍 Choose Your Fishing Location", expanded=not has_prediction):

    # --- POPULAR FISHING SPOTS PRESETS ---
    fishing_spots = {
        "📍 Select a location...": None,
        "🐟 Negombo (West Coast)": {"lon": 79.0, "lat": 7.2},
        "🐟 Chilaw (West Coast)": {"lon": 79.1, "lat": 7.6},
        "🐟 Kalpitiya (Northwest)": {"lon": 79.2, "lat": 8.3},
        "🐟 Galle (South Coast)": {"lon": 79.9, "lat": 5.7},
        "🐟 Matara (South Coast)": {"lon": 80.4, "lat": 5.5},
        "🐟 Trincomalee (East Coast)": {"lon": 82.0, "lat": 8.6},
        "🐟 Batticaloa (East Coast)": {"lon": 82.2, "lat": 7.7},
        "🐟 Jaffna (North)": {"lon": 79.7, "lat": 10.0},
        "🐟 Mannar (Northwest)": {"lon": 79.2, "lat": 9.2},
        "🐟 Hambantota (South)": {"lon": 81.3, "lat": 5.7},
    }

    # Initialize session state for location
    if 'selected_lat' not in st.session_state:
        st.session_state.selected_lat = 6.5
    if 'selected_lon' not in st.session_state:
        st.session_state.selected_lon = 79.2

    selected_spot = st.selectbox(
        "🏝️ Quick Select: Popular Fishing Spots",
        options=list(fishing_spots.keys()),
        help="Choose a well-known fishing location to auto-fill coordinates"
    )

    # Update session state if a preset was selected
    if fishing_spots[selected_spot] is not None:
        st.session_state.selected_lat = fishing_spots[selected_spot]["lat"]
        st.session_state.selected_lon = fishing_spots[selected_spot]["lon"]

    # --- CLICKABLE MAP ---
    st.markdown("<p style='color: #a78bfa; font-size: 0.9rem; margin-bottom: 0.3rem;'>🗺️ Or click on the map to select your fishing spot:</p>", unsafe_allow_html=True)

    # Create a small clickable map
    click_map = folium.Map(
        location=[7.5, 80.0],
        zoom_start=7,
        tiles='OpenStreetMap',
        width='100%',
        height=250
    )

    # Add study area boundary
    folium.Rectangle(
        bounds=[[4.5, 77.0], [11.0, 83.5]],
        color='#8b5cf6',
        fill=False,
        weight=2,
        dash_array='5',
        popup='Study Area: Sri Lankan Waters'
    ).add_to(click_map)

    # Add Sri Lanka outline for reference
    folium.Rectangle(
        bounds=[[5.9, 79.5], [9.9, 81.9]],
        color='#ef4444',
        fill=True,
        fillColor='#ef4444',
        fillOpacity=0.1,
        weight=1,
        popup='Land Area (avoid)'
    ).add_to(click_map)

    # Show current selection on the click map
    folium.CircleMarker(
        location=[st.session_state.selected_lat, st.session_state.selected_lon],
        radius=8,
        color='#22c55e',
        fill=True,
        fillColor='#22c55e',
        fillOpacity=0.8,
        popup=f"Selected: [{st.session_state.selected_lat:.2f}, {st.session_state.selected_lon:.2f}]"
    ).add_to(click_map)

    # Render interactive map (captures clicks)
    map_data = st_folium(click_map, height=250, width=None, key="location_picker")

    # Update coordinates from map click
    if map_data and map_data.get('last_clicked'):
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lon = map_data['last_clicked']['lng']
        # Clamp to study area
        clicked_lat = max(4.5, min(11.0, clicked_lat))
        clicked_lon = max(77.0, min(83.5, clicked_lon))
        st.session_state.selected_lat = round(clicked_lat, 2)
        st.session_state.selected_lon = round(clicked_lon, 2)

    st.markdown("---")

    # --- MANUAL COORDINATE INPUTS ---
    st.markdown("<p style='color: #94a3b8; font-size: 0.85rem;'>✏️ Or enter coordinates manually:</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        lon = st.number_input(
            "Longitude",
            min_value=77.0,
            max_value=83.5,
            value=st.session_state.selected_lon,
            step=0.1,
            format="%.2f",
            help="Sri Lankan waters: 77.0°E to 83.5°E"
        )
    with col2:
        lat = st.number_input(
            "Latitude",
            min_value=4.5,
            max_value=11.0,
            value=st.session_state.selected_lat,
            step=0.1,
            format="%.2f",
            help="Sri Lankan waters: 4.5°N to 11.0°N"
        )

    # Sync manual input back to session state
    st.session_state.selected_lat = lat
    st.session_state.selected_lon = lon

    # Validate if coordinates are on land (Sri Lankan land mass)
    # Sri Lanka roughly: 79.5°E to 81.9°E longitude, 5.9°N to 9.9°N latitude
    if 79.5 <= lon <= 81.9 and 5.9 <= lat <= 9.9:
        st.warning("⚠️ **Warning**: These coordinates appear to be on LAND (Sri Lanka mainland). This model predicts ocean fishing zones. Please select coordinates in the ocean:\n- **West coast**: Longitude < 79.5 (e.g., 79.2)\n- **East coast**: Longitude > 81.9 (e.g., 82.1)")
        st.info("💡 **Suggested ocean locations**:\n- West: Lon=79.2, Lat=6.5\n- East: Lon=82.1, Lat=7.5\n- Northwest: Lon=79.3, Lat=8.5")


# Initialize session state for fetched data
if 'fetched_data' not in st.session_state:
    st.session_state.fetched_data = None

#============================================================================
# ACTION BUTTONS AND PREDICTION LOGIC
#============================================================================

if input_mode == "Quick Prediction":
    # Just the predict button using the manulay inputted values
    predict_button = st.button("🎯 Find Fishing Zones", type="primary", key="manual_predict", use_container_width=True)

else:
    # Live data mode
    fetch_button = st.button("🛰️ Fetch Live Ocean Data", type="primary", key="fetch_data", use_container_width=True)
    
    if not gee_initialized:
        err_detail = f" Error: {gee_init_error}" if gee_init_error else ""
        st.warning(f"Google Earth Engine not initialized — using Open-Meteo for live ocean data.{err_detail}")
    
    if fetch_button:
        if gee_initialized:
            # Try GEE first
            with st.spinner("🛰️ Fetching satellite data..."):
                ocean_data, error = fetch_realtime_data(lat, lon)
                if error:
                    st.warning(f"GEE fetch failed: {error}. Trying Open-Meteo...")
                    ocean_data = None
                else:
                    st.session_state.fetched_data = ocean_data
                    st.success(f"✅ Data fetched! {ocean_data['note']}")
        else:
            ocean_data = None
        
        # Fallback to Open-Meteo if GEE fails or isn't available
        if not st.session_state.get('fetched_data'):
            with st.spinner("🌊 Fetching from Open-Meteo..."):
                try:
                    meteo_currents = fetch_open_meteo_currents(lon, lat)
                    meteo_sst_data = fetch_open_meteo_sst(lon, lat)
                    
                    if meteo_currents or meteo_sst_data:
                        fallback_sst = meteo_sst_data['sst'] if meteo_sst_data else 28.5
                        fallback_speed = meteo_currents['current_speed'] if meteo_currents else 1.5
                        fallback_u = meteo_currents['u'] if meteo_currents else 0.9
                        fallback_v = meteo_currents['v'] if meteo_currents else 0.6
                        
                        st.session_state.fetched_data = {
                            'sst': fallback_sst,
                            'current_speed': fallback_speed,
                            'current_u': fallback_u,
                            'current_v': fallback_v,
                            'data_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'success': True,
                            'note': 'SST from Open-Meteo, Currents from Open-Meteo (live)'
                        }
                        st.success("✅ Live ocean data fetched from Open-Meteo!")
                    else:
                        st.error("Could not fetch data from Open-Meteo. Please try again or use Quick Prediction mode.")
                except Exception as e:
                    st.error(f"Open-Meteo fetch failed: {str(e)}. Try Quick Prediction mode.")
    
    # Display fetched data
    if st.session_state.fetched_data:
        data = st.session_state.fetched_data
        # Display fetched data in main area
        st.markdown("### 📊 Fetched Data")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🌡️ Sea Temperature", f"{data['sst']:.2f}°C" if data['sst'] else "N/A")
        with col2:
            st.metric("🌊 Current Speed", f"{data['current_speed']:.2f} m/s" if data['current_speed'] else "N/A")
        
        st.caption(f"📅 Data from: {data['data_date']}")
        
        # Show data source note if available
        if 'note' in data:
            st.info(f"ℹ️ {data['note']}")
        
        # Use fetched data for prediction
        sst = data['sst'] if data['sst'] else 28.5
        current_speed = data['current_speed'] if data['current_speed'] else 2.0
        current_u = data['current_u'] if data['current_u'] else 1.0
        current_v = data['current_v'] if data['current_v'] else 1.0
        
        st.markdown("---")
        predict_button = st.button("🎯 Find Fishing Zones", type="primary", key="realtime_predict", use_container_width=True)
    else:
        st.info("👆 Click 'Fetch Live Ocean Data' to get real-time data for the selected location")
        predict_button = False


# Calculate derived features
mean_sst = 29.0  # Average from training data
temp_deviation = sst - mean_sst

# Spatial features (normalized)
lon_normalized = (lon - 79.5) / (82.0 - 79.5)
lat_normalized = (lat - 5.9) / (9.9 - 5.9)

#============================================================================
# MAIN CONTENT - 2 COLUMNS
#============================================================================

# Auto-scroll anchor
st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1.5])


#============================================================================
# COLUMN 1: MAP
#============================================================================
with col1:
    st.markdown("### 🗺️ Sri Lankan Coastal Waters")
    
    # Create map centered on Sri Lanka
    m = folium.Map(
        location=[lat, lon],  # Use selected location
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Add study area rectangle
    folium.Rectangle(
        bounds=[[5.9, 79.5], [9.9, 81.9]],
        color='blue',
        fill=False,
        weight=2,
        popup='Study Area: Sri Lankan Waters'
    ).add_to(m)
    
    # If prediction made, show it on map
    if predict_button and model is not None:
        # Make prediction
        features = np.array([[
            sst,
            current_speed,
            current_u,
            current_v,
            temp_deviation,
            lon_normalized,
            lat_normalized
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        st.session_state.last_prediction = prediction
        
        # Add to prediction history
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        st.session_state.prediction_history.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'lat': f"{lat:.2f}°N",
            'lon': f"{lon:.2f}°E",
            'sst': f"{sst:.1f}°C",
            'current': f"{current_speed:.1f} m/s",
            'result': prediction
        })
        # Keep only last 10
        st.session_state.prediction_history = st.session_state.prediction_history[-10:]
        
        # Auto-scroll to results
        import streamlit.components.v1 as components
        components.html("""
            <script>
                const target = window.parent.document.getElementById('results-section');
                if (target) { target.scrollIntoView({behavior: 'smooth', block: 'start'}); }
            </script>
        """, height=0)
        
        # Color mapping
        color_map = {
            'High': 'green',
            'Medium': 'orange',
            'Low': 'red'
        }
        
        # Add marker for prediction point
        folium.CircleMarker(
            location=[lat, lon],
            radius=20,
            popup=f"""
            <b>🎯 AI Prediction: {prediction}</b><br>
            SST: {sst}°C<br>
            Current: {current_speed} m/s<br>
            Location: [{lat:.2f}, {lon:.2f}]
            """,
            color=color_map[prediction],
            fill=True,
            fillColor=color_map[prediction],
            fillOpacity=0.7,
            weight=3
        ).add_to(m)
    
    # K-Means Zone Overlay (Hybrid: K-Means clustering + RF prediction)
    # Cluster colors represent K-Means ocean condition groups
    cluster_color_map = {
        0: '#3b82f6',    # blue - Cluster A
        1: '#f59e0b',    # amber - Cluster B
        2: '#14b8a6',    # teal - Cluster C
    }
    cluster_labels = {
        0: 'Nutrient-Rich Coastal Zone',
        1: 'Seasonal Fishing Zone',
        2: 'Deep Water Pelagic Zone',
    }
    cluster_descriptions = {
        0: 'Strong currents bring nutrients close to shore — attracts reef and coastal fish species',
        1: 'Moderate conditions that change seasonally — good during monsoon transitions',
        2: 'Calmer deep waters — suited for tuna, swordfish and other pelagic species',
    }
    
    # Generate K-Means zones
    kmeans_zones = generate_kmeans_zones(model, scaler, kmeans_model)
    
    if kmeans_zones:
        cell_size = 0.5  # Grid cell size in degrees
        for zone in kmeans_zones:
            zone_color = cluster_color_map.get(zone['cluster'], '#6b7280')
            zone_label = cluster_labels.get(zone['cluster'], f"Zone {zone['cluster']}")
            zone_desc = cluster_descriptions.get(zone['cluster'], '')
            folium.Rectangle(
                bounds=[
                    [zone['lat'] - cell_size/2, zone['lon'] - cell_size/2],
                    [zone['lat'] + cell_size/2, zone['lon'] + cell_size/2]
                ],
                color=zone_color,
                fill=True,
                fillColor=zone_color,
                fillOpacity=0.15,
                weight=1,
                opacity=0.4,
                popup=f"""
                <b>{zone_label}</b><br>
                <i>{zone_desc}</i><br><br>
                RF Prediction: <b>{zone['prediction']}</b><br>
                Est. SST: {zone['sst']:.1f}°C<br>
                Est. Current: {zone['current']:.1f} m/s
                """
            ).add_to(m)
    
    # Display map
    folium_static(m, width=700, height=500)
    
    # Legend
    st.markdown("""
    **Legend:**
    - 🔵 **Blue** = Nutrient-Rich Coastal — Strong currents, reef & coastal fish
    - 🟡 **Amber** = Seasonal Fishing — Best during monsoon transitions
    - 🟢 **Teal** = Deep Water Pelagic — Tuna, swordfish territory
    - **Large bright marker** = Your AI prediction (Random Forest)
    - *Ocean zones identified by K-Means clustering*
    """)

#============================================================================
# COLUMN 2: PREDICTION RESULTS
#============================================================================
with col2:
    st.markdown("### 🎯 Prediction Results")
    
    if predict_button and model is not None:
        # Prepare features
        features = np.array([[
            sst,
            current_speed,
            current_u,
            current_v,
            temp_deviation,
            lon_normalized,
            lat_normalized
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get class names
        classes = model.classes_
        
        # Prediction messages and icons
        prediction_info = {
            'High': {
                'icon': '🎣',
                'title': 'Excellent Fishing!',
                'description': 'Perfect conditions for a great catch. This is an ideal fishing spot!',
                'class': 'prediction-high'
            },
            'Medium': {
                'icon': '🐟',
                'title': 'Good Fishing',
                'description': 'Decent conditions. You should find some fish here.',
                'class': 'prediction-medium'
            },
            'Low': {
                'icon': '⚠️',
                'title': 'Poor Fishing',
                'description': 'Not ideal conditions. Consider trying a different location.',
                'class': 'prediction-low'
            }
        }
        
        info = prediction_info[prediction]
        
        # Display prediction with modern card
        st.markdown(f"""
        <div class='{info["class"]}'>
            <div class='prediction-icon'>{info['icon']}</div>
            <div class='prediction-title'>{info['title']}</div>
            <div class='prediction-subtitle'>Fishing Potential: {prediction}</div>
            <div class='prediction-description'>{info['description']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Ocean Conditions as colorful metric cards
        st.markdown("### 🌊 Ocean Conditions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temp_status = "✓ Optimal" if 28.0 <= sst <= 29.0 else "Normal"
            st.markdown(f"""
            <div class='metric-card metric-card-blue'>
                <div class='metric-icon'>🌡️</div>
                <div class='metric-value' style='color: white;'>{sst:.1f}°C</div>
                <div class='metric-label' style='color: white;'>Temperature</div>
                <div style='font-size: 0.65rem; margin-top: 0.3rem; opacity: 0.9; color: white;'>{temp_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            current_status = "✓ Optimal" if 1.0 <= current_speed <= 3.0 else "Weak" if current_speed < 1.0 else "Strong"
            st.markdown(f"""
            <div class='metric-card metric-card-cyan'>
                <div class='metric-icon'>🌊</div>
                <div class='metric-value' style='color: white;'>{current_speed:.1f} m/s</div>
                <div class='metric-label' style='color: white;'>Current</div>
                <div style='font-size: 0.65rem; margin-top: 0.3rem; opacity: 0.9; color: white;'>{current_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calculate current direction
            import math
            direction_deg = math.degrees(math.atan2(current_v, current_u)) % 360
            directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            direction_idx = int((direction_deg + 22.5) / 45) % 8
            direction = directions[direction_idx]
            
            st.markdown(f"""
            <div class='metric-card metric-card-green'>
                <div class='metric-icon'>🧭</div>
                <div class='metric-value' style='color: white;'>{direction}</div>
                <div class='metric-label' style='color: white;'>Direction</div>
                <div style='font-size: 0.65rem; margin-top: 0.3rem; opacity: 0.9; color: white;'>→ {direction}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence levels in expandable section
        with st.expander("📊 Confidence Levels & Details", expanded=False):
            st.markdown("#### AI Prediction Confidence:")
            for class_name, prob in zip(classes, probabilities):
                st.markdown(f"**{class_name}**")
                st.progress(float(prob))
                st.caption(f"{prob*100:.1f}%")
            
            st.markdown("---")
            st.caption(f"🕒 Predicted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.caption(f"📍 Location: [{lat:.2f}°N, {lon:.2f}°E]")
        
        # --- DOWNLOADABLE REPORT ---
        st.markdown("---")
        st.markdown("### 📥 Offline Report")
        st.caption("Download this report to use at sea without internet")
        
        # Generate static SVG map (works completely offline - no tile server needed)
        report_color = {'High': '#22c55e', 'Medium': '#f59e0b', 'Low': '#ef4444'}
        marker_color = report_color.get(prediction, '#8b5cf6')
        
        # SVG coordinate mapping: lon 77-84 -> x 0-500, lat 4-12 -> y 500-0 (flipped)
        svg_w, svg_h = 500, 450
        def to_svg(lon_v, lat_v):
            x = (lon_v - 77.0) / (84.0 - 77.0) * svg_w
            y = svg_h - (lat_v - 4.0) / (12.0 - 4.0) * svg_h
            return x, y
        
        # Build K-Means zone rectangles
        zone_rects = ""
        report_cluster_colors = {0: '#3b82f6', 1: '#f59e0b', 2: '#14b8a6'}
        report_zones = generate_kmeans_zones(model, scaler, kmeans_model)
        if report_zones:
            cell_size = 0.5
            for zone in report_zones:
                zc = report_cluster_colors.get(zone['cluster'], '#6b7280')
                x1, y1 = to_svg(zone['lon'] - cell_size/2, zone['lat'] + cell_size/2)
                x2, y2 = to_svg(zone['lon'] + cell_size/2, zone['lat'] - cell_size/2)
                zone_rects += f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2-x1:.1f}" height="{y2-y1:.1f}" fill="{zc}" fill-opacity="0.2" stroke="{zc}" stroke-width="0.5" stroke-opacity="0.4"/>'
        
        # Sri Lanka land boundary (simplified polygon)
        sri_lanka_points = [
            (79.7, 9.8), (80.0, 9.5), (80.2, 9.3), (80.0, 8.8), (79.8, 8.2),
            (79.9, 7.5), (79.8, 7.0), (80.1, 6.5), (80.2, 6.1), (80.5, 5.9),
            (80.8, 6.0), (81.2, 6.1), (81.6, 6.5), (81.8, 7.0), (81.9, 7.5),
            (81.8, 8.0), (81.5, 8.5), (81.0, 9.0), (80.5, 9.5), (80.2, 9.8),
            (80.0, 9.9), (79.7, 9.8)
        ]
        poly_str = " ".join([f"{to_svg(p[0], p[1])[0]:.1f},{to_svg(p[0], p[1])[1]:.1f}" for p in sri_lanka_points])
        
        # Study area
        sa_x1, sa_y1 = to_svg(79.5, 9.9)
        sa_x2, sa_y2 = to_svg(81.9, 5.9)
        
        # Prediction marker
        mx, my = to_svg(lon, lat)
        
        # Grid lines
        grid_lines = ""
        for g_lon in range(77, 85):
            gx, _ = to_svg(g_lon, 4)
            grid_lines += f'<line x1="{gx:.0f}" y1="0" x2="{gx:.0f}" y2="{svg_h}" stroke="#334155" stroke-width="0.5"/>'
            grid_lines += f'<text x="{gx:.0f}" y="{svg_h-5}" fill="#64748b" font-size="10" text-anchor="middle">{g_lon}°E</text>'
        for g_lat in range(4, 13):
            _, gy = to_svg(77, g_lat)
            grid_lines += f'<line x1="0" y1="{gy:.0f}" x2="{svg_w}" y2="{gy:.0f}" stroke="#334155" stroke-width="0.5"/>'
            grid_lines += f'<text x="5" y="{gy-3:.0f}" fill="#64748b" font-size="10">{g_lat}°N</text>'
        
        map_html = f"""
        <svg viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:600px;background:#0c1222;border-radius:8px;">
            {grid_lines}
            {zone_rects}
            <rect x="{sa_x1:.1f}" y="{sa_y1:.1f}" width="{sa_x2-sa_x1:.1f}" height="{sa_y2-sa_y1:.1f}" fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="5,3"/>
            <polygon points="{poly_str}" fill="#1e293b" stroke="#64748b" stroke-width="1"/>
            <circle cx="{mx:.1f}" cy="{my:.1f}" r="12" fill="{marker_color}" fill-opacity="0.3" stroke="{marker_color}" stroke-width="2"/>
            <circle cx="{mx:.1f}" cy="{my:.1f}" r="5" fill="{marker_color}"/>
            <text x="{mx+15:.1f}" y="{my+4:.1f}" fill="#e2e8f0" font-size="11" font-weight="bold">{prediction} ({lat:.1f}°N, {lon:.1f}°E)</text>
        </svg>"""
        
        # Build confidence bars HTML
        confidence_html = ""
        for class_name, prob in zip(classes, probabilities):
            bar_color = report_color.get(class_name, '#8b5cf6')
            confidence_html += f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                    <span>{class_name}</span><span>{prob*100:.1f}%</span>
                </div>
                <div style="background: #334155; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: {bar_color}; width: {prob*100:.1f}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>"""
        
        # Get data source note
        data_note = ""
        if st.session_state.get('fetched_data') and 'note' in st.session_state.fetched_data:
            data_note = st.session_state.fetched_data['note']
        
        # Current direction
        import math
        dir_deg = math.degrees(math.atan2(current_v, current_u)) % 360
        dir_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        dir_label = dir_names[int((dir_deg + 22.5) / 45) % 8]
        
        # Build the full HTML report
        report_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        prediction_emoji = {'High': '🟢', 'Medium': '🟠', 'Low': '🔴'}
        prediction_desc = {
            'High': 'Excellent conditions for fishing! This is an ideal spot.',
            'Medium': 'Decent conditions. You should find some fish here.',
            'Low': 'Not ideal conditions. Consider a different location.'
        }
        
        html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FishSense Report - {report_date}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
        .header {{ text-align: center; padding: 20px; background: linear-gradient(135deg, #1e293b, #334155); border-radius: 12px; margin-bottom: 20px; border: 1px solid rgba(139, 92, 246, 0.3); }}
        .header h1 {{ font-size: 1.8rem; color: #a78bfa; margin-bottom: 5px; }}
        .header p {{ color: #94a3b8; font-size: 0.9rem; }}
        .card {{ background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(139, 92, 246, 0.2); border-radius: 10px; padding: 20px; margin-bottom: 15px; }}
        .card h2 {{ font-size: 1.2rem; color: #c4b5fd; margin-bottom: 12px; }}
        .prediction-box {{ text-align: center; padding: 25px; border-radius: 10px; margin-bottom: 15px; }}
        .prediction-high {{ background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.1)); border: 2px solid #22c55e; }}
        .prediction-medium {{ background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(251, 191, 36, 0.1)); border: 2px solid #fbbf24; }}
        .prediction-low {{ background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1)); border: 2px solid #ef4444; }}
        .prediction-box h2 {{ font-size: 2rem; margin: 10px 0 5px; color: #ffffff; }}
        .prediction-box p {{ color: #cbd5e1; }}
        .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 15px; }}
        .metric {{ background: rgba(51, 65, 85, 0.6); border-radius: 8px; padding: 15px; text-align: center; }}
        .metric .value {{ font-size: 1.5rem; font-weight: 700; color: #ffffff; }}
        .metric .label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
        .map-container {{ border-radius: 10px; overflow: hidden; margin-bottom: 15px; border: 1px solid rgba(139, 92, 246, 0.2); }}
        .footer {{ text-align: center; padding: 15px; color: #64748b; font-size: 0.8rem; border-top: 1px solid #1e293b; margin-top: 20px; }}
        .data-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(100, 116, 139, 0.2); }}
        .data-row:last-child {{ border-bottom: none; }}
        .note {{ background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6; padding: 10px 15px; border-radius: 0 8px 8px 0; margin-top: 10px; font-size: 0.85rem; color: #93c5fd; }}
        @media print {{
            body {{ background: white; color: #1a1a2e; padding: 10px; }}
            .card {{ border: 1px solid #ddd; }}
            .header {{ background: #f0f0f0; }}
            .header h1 {{ color: #6d28d9; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🐟 FishSense Fishing Report</h1>
        <p>Generated on {report_timestamp}</p>
        <p style="margin-top: 5px; font-size: 0.8rem;">📍 Location: {lat:.2f}°N, {lon:.2f}°E</p>
    </div>

    <div class="prediction-box prediction-{prediction.lower()}">
        <div style="font-size: 2.5rem;">{prediction_emoji.get(prediction, '🎯')}</div>
        <h2>Fishing Potential: {prediction}</h2>
        <p>{prediction_desc.get(prediction, '')}</p>
    </div>

    <div class="card">
        <h2>🌊 Ocean Conditions</h2>
        <div class="metrics">
            <div class="metric">
                <div class="label">🌡️ Temperature</div>
                <div class="value">{sst:.1f}°C</div>
            </div>
            <div class="metric">
                <div class="label">🌊 Current Speed</div>
                <div class="value">{current_speed:.1f} m/s</div>
            </div>
            <div class="metric">
                <div class="label">🧭 Direction</div>
                <div class="value">{dir_label}</div>
            </div>
        </div>
        <div class="data-row"><span>Current U (East-West)</span><span>{current_u:.3f} m/s</span></div>
        <div class="data-row"><span>Current V (North-South)</span><span>{current_v:.3f} m/s</span></div>
        <div class="data-row"><span>Temperature Deviation</span><span>{temp_deviation:+.1f}°C</span></div>
        {"<div class='note'>ℹ️ " + data_note + "</div>" if data_note else ""}
    </div>

    <div class="card">
        <h2>📊 AI Confidence</h2>
        {confidence_html}
    </div>

    <div class="card">
        <h2>🗺️ Location Map with K-Means Zones</h2>
        <div class="map-container">
            {map_html}
        </div>
        <div style="margin-top: 10px;">
            <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;"><b>Zone Legend (K-Means Clusters):</b></div>
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                <span style="display: flex; align-items: center; gap: 5px;"><span style="width: 14px; height: 14px; background: #3b82f6; border-radius: 3px; display: inline-block;"></span> Nutrient-Rich Coastal</span>
                <span style="display: flex; align-items: center; gap: 5px;"><span style="width: 14px; height: 14px; background: #f59e0b; border-radius: 3px; display: inline-block;"></span> Seasonal Fishing</span>
                <span style="display: flex; align-items: center; gap: 5px;"><span style="width: 14px; height: 14px; background: #14b8a6; border-radius: 3px; display: inline-block;"></span> Deep Water Pelagic</span>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>📋 Details</h2>
        <div class="data-row"><span>Latitude</span><span>{lat:.4f}°N</span></div>
        <div class="data-row"><span>Longitude</span><span>{lon:.4f}°E</span></div>
        <div class="data-row"><span>Prediction Model</span><span>Random Forest (FishSense)</span></div>
        <div class="data-row"><span>Report Generated</span><span>{report_timestamp}</span></div>
    </div>

    <div class="footer">
        <p>🐟 FishSense - AI-Powered Ocean Health & Fishing Zone Intelligence</p>
        <p style="margin-top: 5px;">This report can be viewed offline. Save it to your device before heading out.</p>
    </div>
</body>
</html>"""
        
        # Download button
        st.download_button(
            label="📥 Download Fishing Report",
            data=html_report,
            file_name=f"FishSense_Report_{report_date}_{lat:.1f}N_{lon:.1f}E.html",
            mime="text/html",
            use_container_width=True
        )
        
    else:
        # Show placeholder when no prediction yet
        st.markdown("""
        <div class='info-card' style='text-align: center; padding: 3rem 2rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🎯</div>
            <h3>Ready to Predict!</h3>
            <p>Select a prediction mode above, set your ocean conditions and location, then click <strong>'Find Fishing Zones'</strong> to get your prediction.</p>
        </div>
        """, unsafe_allow_html=True)



#============================================================================
# PREDICTION HISTORY
#============================================================================
if st.session_state.get('prediction_history') and len(st.session_state.prediction_history) > 0:
    st.markdown("---")
    st.markdown("<h2 style='color: #ffffff;'>📋 Prediction History</h2>", unsafe_allow_html=True)
    st.caption("Your recent predictions this session — compare different spots to find the best one!")
    
    # Build dataframe from history
    import pandas as pd
    result_emoji = {'High': '🟢 High', 'Medium': '🟠 Medium', 'Low': '🔴 Low'}
    history_data = []
    for entry in reversed(st.session_state.prediction_history):
        history_data.append({
            '🕐 Time': entry['time'],
            '📍 Location': f"{entry['lat']}, {entry['lon']}",
            '🌡️ SST': entry['sst'],
            '🌊 Current': entry['current'],
            '🎯 Result': result_emoji.get(entry['result'], entry['result'])
        })
    
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history, use_container_width=True, hide_index=True)


#============================================================================
# BOTTOM SECTION: MODEL PERFORMANCE
#============================================================================
st.markdown("---")
st.markdown("<h2 style='color: #ffffff;'>📈 Model Performance & Features</h2>", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("<h3 style='color: #e0e0e0;'>🎯 Accuracy Metrics</h3>", unsafe_allow_html=True)
    if metadata:
        st.metric("Cross-Validated Accuracy", f"{metadata.get('cv_accuracy', metadata['test_accuracy'])*100:.1f}%")
        st.metric("Spatial CV Accuracy", f"{metadata.get('spatial_cv_accuracy', metadata['test_accuracy'])*100:.1f}%")
        st.metric("Number of Classes", len(metadata['classes']))

with col5:
    st.markdown("<h3 style='color: #e0e0e0;'>🌲 Model Details</h3>", unsafe_allow_html=True)
    if metadata:
        st.metric("Algorithm", "Random Forest")
        st.metric("Number of Trees", "100")
        st.metric("Training Samples", metadata['n_train_samples'])

with col6:
    st.markdown("<h3 style='color: #e0e0e0;'>🔬 Top Features</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color: #cbd5e1;'>
    1. <strong>Current Speed</strong> (33.9%)<br>
    2. <strong>Temperature</strong> (21.4%)<br>
    3. <strong>Temp Deviation</strong> (20.6%)
    </div>
    """, unsafe_allow_html=True)

