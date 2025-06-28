import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt
from src.infer import FirePredictor
from src.utils import load_config

# Custom CSS for dark fire theme and glassmorphism
st.markdown("""
<style>
/* Dark Fire Theme with Glassmorphism */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

/* Main background with animated fire gradient */
.main {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0a0a 25%, #2a0a0a 50%, #1a0a0a 75%, #0a0a0a 100%);
    background-size: 400% 400%;
    animation: fireGradient 8s ease-in-out infinite;
}

@keyframes fireGradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating ember particles */
.ember {
    position: fixed;
    width: 4px;
    height: 4px;
    background: radial-gradient(circle, #ff6b35, #ff4500);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
    z-index: -1;
}

@keyframes float {
    0%, 100% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { transform: translateY(-100px) rotate(360deg); opacity: 0; }
}

/* Glassmorphism containers */
.glass-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.glass-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Glowing borders */
.glow-border {
    border: 2px solid transparent;
    background: linear-gradient(45deg, #ff6b35, #ff4500, #ff6b35);
    background-size: 200% 200%;
    animation: glowPulse 2s ease-in-out infinite;
    border-radius: 20px;
    padding: 2px;
}

@keyframes glowPulse {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Custom title styling */
.fire-title {
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    font-size: 3rem;
    background: linear-gradient(45deg, #ff6b35, #ff4500, #ff8c42);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: titleGlow 3s ease-in-out infinite;
    text-align: center;
    margin: 20px 0;
    text-shadow: 0 0 30px rgba(255, 107, 53, 0.5);
}

@keyframes titleGlow {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background: rgba(0, 0, 0, 0.8);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 107, 53, 0.3);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(45deg, #ff6b35, #ff4500);
    border: none;
    border-radius: 15px;
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 107, 53, 0.5);
    background: linear-gradient(45deg, #ff4500, #ff6b35);
}

/* Selectbox styling */
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 107, 53, 0.3);
    border-radius: 10px;
    color: white;
}

/* Date input styling */
.stDateInput > div > div {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 107, 53, 0.3);
    border-radius: 10px;
    color: white;
}

/* Text styling */
.stMarkdown {
    color: #f0f0f0;
}

/* JSON display styling */
.json-container {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 107, 53, 0.2);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}

/* Info box styling */
.stAlert {
    background: rgba(255, 107, 53, 0.1);
    border: 1px solid rgba(255, 107, 53, 0.3);
    border-radius: 15px;
    color: #ffd700;
}

/* Error box styling */
.stAlert[data-baseweb="notification"] {
    background: rgba(255, 69, 0, 0.1);
    border: 1px solid rgba(255, 69, 0, 0.3);
    color: #ff6b6b;
}

/* Success box styling */
.stAlert[data-baseweb="toast"] {
    background: rgba(0, 255, 127, 0.1);
    border: 1px solid rgba(0, 255, 127, 0.3);
    color: #00ff7f;
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom ember particles */
.ember-1 { left: 10%; animation-delay: 0s; }
.ember-2 { left: 20%; animation-delay: 1s; }
.ember-3 { left: 30%; animation-delay: 2s; }
.ember-4 { left: 40%; animation-delay: 3s; }
.ember-5 { left: 50%; animation-delay: 4s; }
.ember-6 { left: 60%; animation-delay: 5s; }
.ember-7 { left: 70%; animation-delay: 6s; }
.ember-8 { left: 80%; animation-delay: 7s; }
.ember-9 { left: 90%; animation-delay: 8s; }

</style>

<!-- Floating ember particles -->
<div class="ember ember-1"></div>
<div class="ember ember-2"></div>
<div class="ember ember-3"></div>
<div class="ember ember-4"></div>
<div class="ember ember-5"></div>
<div class="ember ember-6"></div>
<div class="ember ember-7"></div>
<div class="ember ember-8"></div>
<div class="ember ember-9"></div>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="🔥 Forest Fire AI Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title with fire aesthetics
st.markdown('<h1 class="fire-title">🔥 FOREST FIRE AI PREDICTOR 🔥</h1>', unsafe_allow_html=True)

# Load configuration and initialize predictor
config = load_config()
predictor = FirePredictor()

# Glassmorphism sidebar
with st.sidebar:
    st.markdown('<div class="glass-container glow-border">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Prediction Controls")
    
    region = st.selectbox(
        "📍 Select Region", 
        list(config['regions'].keys()),
        help="Choose the forest region for fire prediction"
    )
    
    date = st.date_input(
        "📅 Select Date", 
        datetime.date.today(),
        help="Select the date for fire prediction"
    )
    
    time_window = st.selectbox(
        "⏰ Forecast Window", 
        [6, 12, 24], 
        index=2,
        help="Choose prediction time window in hours"
    )
    
    predict_button = st.button(
        "🚀 Predict Fire Spread",
        help="Run AI prediction for fire spread analysis"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
if predict_button:
    with st.spinner("🔥 Running AI prediction..."):
        try:
            result = predictor.predict(region=region, date=str(date), time_window=f"{time_window}h")
            pred = np.array(result['prediction'])
            
            # Create glassmorphism container for prediction
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.markdown("### 🔥 Fire Spread Prediction")
            
            # Enhanced heatmap with fire theme
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            # Use fire-themed colormap
            im = ax.imshow(pred, cmap='hot', interpolation='gaussian', alpha=0.8)
            ax.set_title(f'🔥 Fire Spread Prediction - {region.title()}', 
                        color='#ff6b35', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Longitude', color='#f0f0f0', fontsize=12)
            ax.set_ylabel('Latitude', color='#f0f0f0', fontsize=12)
            
            # Customize colorbar
            cbar = plt.colorbar(im, ax=ax, label='Fire Probability')
            cbar.set_label('Fire Probability', color='#f0f0f0', fontsize=12)
            cbar.ax.tick_params(colors='#f0f0f0')
            
            # Style the plot
            ax.tick_params(colors='#f0f0f0')
            ax.spines['bottom'].set_color('#ff6b35')
            ax.spines['top'].set_color('#ff6b35')
            ax.spines['left'].set_color('#ff6b35')
            ax.spines['right'].set_color('#ff6b35')
            
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Metadata in glassmorphism container
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.markdown("### 📊 Live Input Conditions")
            st.markdown('<div class="json-container">', unsafe_allow_html=True)
            st.json(result['metadata'])
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download section
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.markdown("### 💾 Download Results")
            st.download_button(
                "📥 Download Prediction (JSON)", 
                data=str(result), 
                file_name=f"fire_prediction_{region}_{date}.json",
                help="Download the prediction results as JSON file"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Region information
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.markdown("### 🗺️ Region Information")
            region_info = config['regions'][region]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**📍 Center Coordinates:** `{region_info['center']}`")
            with col2:
                st.markdown(f"**🗺️ Bounds:** `{region_info['bounds']}`")
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("ℹ️ This is a demo with synthetic data. In production, this would use real satellite and weather data.")
else:
    # Welcome section with glassmorphism
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 🎯 Welcome to Forest Fire AI Predictor")
    st.info("🔥 Select parameters and click 'Predict Fire Spread' to begin AI-powered fire prediction analysis.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Project information with enhanced styling
    st.markdown('<div class="glass-container glow-border">', unsafe_allow_html=True)
    st.markdown("""
    ## 🚀 About This Project
    
    This **Forest Fire Spread Prediction** system leverages cutting-edge AI technology:
    
    ### 🤖 **AI/ML Models**
    - **Transformer + Cellular Automata** hybrid architecture
    - **Real-time pattern recognition** and spatial analysis
    - **Advanced deep learning** algorithms
    
    ### 📡 **Data Sources**
    - **Satellite imagery** (Sentinel-2, LISS-4)
    - **Weather data** integration
    - **Terrain features** analysis
    - **Historical fire patterns**
    
    ### 🗺️ **Coverage Areas**
    - **Uttarakhand** - Himalayan forests
    - **Himachal Pradesh** - Mountain regions
    - **Karnataka** - Western Ghats
    
    ### ⏰ **Prediction Windows**
    - **6-hour** short-term forecasts
    - **12-hour** medium-term analysis
    - **24-hour** extended predictions
    
    ---
    
    **Built for Bharatiya Antariksh Hackathon 2025** 🚀
    
    *Empowering forest conservation through AI-driven fire prediction*
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Features showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("### 🔥 **Real-time Analysis**")
        st.markdown("Instant fire spread predictions using live satellite and weather data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("### 🎯 **High Accuracy**")
        st.markdown("Advanced AI models trained on extensive historical fire data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("### 📊 **Visual Insights**")
        st.markdown("Interactive heatmaps and detailed risk assessments")
        st.markdown('</div>', unsafe_allow_html=True) 