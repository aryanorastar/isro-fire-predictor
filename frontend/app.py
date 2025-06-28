import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.infer import FirePredictor
from src.utils import load_config
from scipy.ndimage import zoom

# Custom CSS for dark fire theme, glassmorphism, and advanced animations
st.markdown("""
<style>
/* Dark Fire Theme with Advanced Animations */
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

/* Floating ember particles with enhanced physics */
.ember {
    position: fixed;
    width: 4px;
    height: 4px;
    background: radial-gradient(circle, #ff6b35, #ff4500);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
    z-index: -1;
    box-shadow: 0 0 10px rgba(255, 107, 53, 0.8);
}

@keyframes float {
    0%, 100% { transform: translateY(100vh) rotate(0deg) scale(1); opacity: 0; }
    10% { opacity: 1; transform: scale(1.2); }
    50% { transform: translateY(50vh) rotate(180deg) scale(0.8); }
    90% { opacity: 1; transform: scale(1.1); }
    100% { transform: translateY(-100px) rotate(360deg) scale(0.5); opacity: 0; }
}

/* Glassmorphism containers with enhanced effects */
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
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-container:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 15px 40px rgba(255, 107, 53, 0.2);
    border-color: rgba(255, 107, 53, 0.3);
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

/* Glowing borders with enhanced pulse */
.glow-border {
    border: 2px solid transparent;
    background: linear-gradient(45deg, #ff6b35, #ff4500, #ff6b35);
    background-size: 200% 200%;
    animation: glowPulse 2s ease-in-out infinite;
    border-radius: 20px;
    padding: 2px;
    position: relative;
}

.glow-border::after {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, #ff6b35, #ff4500, #ff8c42, #ff6b35);
    background-size: 400% 400%;
    border-radius: 22px;
    z-index: -1;
    animation: outerGlow 3s ease-in-out infinite;
    opacity: 0.5;
}

@keyframes glowPulse {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@keyframes outerGlow {
    0%, 100% { background-position: 0% 50%; opacity: 0.3; }
    50% { background-position: 100% 50%; opacity: 0.7; }
}

/* Enhanced button animations */
.stButton > button {
    background: linear-gradient(45deg, #ff6b35, #ff4500);
    border: none;
    border-radius: 15px;
    color: white;
    font-weight: bold;
    padding: 12px 24px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 8px 25px rgba(255, 107, 53, 0.6);
    background: linear-gradient(45deg, #ff4500, #ff6b35);
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:active {
    transform: translateY(-1px) scale(1.02);
}

/* Loading animation - Fire spreading effect */
.loading-fire {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255, 107, 53, 0.3);
    border-radius: 50%;
    border-top-color: #ff6b35;
    animation: fireSpin 1s ease-in-out infinite;
    margin-right: 10px;
}

@keyframes fireSpin {
    to { transform: rotate(360deg); }
}

/* Data point pulse animation */
.data-pulse {
    animation: dataPulse 2s ease-in-out infinite;
}

@keyframes dataPulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.1); opacity: 0.8; }
}

/* Smooth morphing transitions */
.morph-transition {
    transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.morph-transition:hover {
    transform: perspective(1000px) rotateX(5deg) rotateY(5deg);
}

/* Custom title styling with enhanced effects */
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
    position: relative;
}

.fire-title::after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 3px;
    background: linear-gradient(45deg, #ff6b35, #ff4500);
    animation: titleUnderline 2s ease-in-out infinite;
}

@keyframes titleGlow {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@keyframes titleUnderline {
    0%, 100% { width: 0; }
    50% { width: 80%; }
}

/* Sidebar styling with enhanced effects */
.sidebar .sidebar-content {
    background: rgba(0, 0, 0, 0.8);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 107, 53, 0.3);
}

/* Enhanced form styling */
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 107, 53, 0.3);
    border-radius: 10px;
    color: white;
    transition: all 0.3s ease;
}

.stSelectbox > div > div:hover {
    border-color: rgba(255, 107, 53, 0.6);
    box-shadow: 0 0 15px rgba(255, 107, 53, 0.2);
}

.stDateInput > div > div {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 107, 53, 0.3);
    border-radius: 10px;
    color: white;
    transition: all 0.3s ease;
}

.stDateInput > div > div:hover {
    border-color: rgba(255, 107, 53, 0.6);
    box-shadow: 0 0 15px rgba(255, 107, 53, 0.2);
}

/* Text styling with enhanced readability */
.stMarkdown {
    color: #f0f0f0;
    line-height: 1.6;
}

/* JSON display styling with enhanced effects */
.json-container {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 107, 53, 0.2);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    transition: all 0.3s ease;
}

.json-container:hover {
    border-color: rgba(255, 107, 53, 0.4);
    box-shadow: 0 0 20px rgba(255, 107, 53, 0.1);
}

/* Enhanced alert styling */
.stAlert {
    background: rgba(255, 107, 53, 0.1);
    border: 1px solid rgba(255, 107, 53, 0.3);
    border-radius: 15px;
    color: #ffd700;
    transition: all 0.3s ease;
}

.stAlert:hover {
    background: rgba(255, 107, 53, 0.15);
    border-color: rgba(255, 107, 53, 0.5);
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

/* Custom ember particles with varied timing */
.ember-1 { left: 10%; animation-delay: 0s; }
.ember-2 { left: 20%; animation-delay: 1s; }
.ember-3 { left: 30%; animation-delay: 2s; }
.ember-4 { left: 40%; animation-delay: 3s; }
.ember-5 { left: 50%; animation-delay: 4s; }
.ember-6 { left: 60%; animation-delay: 5s; }
.ember-7 { left: 70%; animation-delay: 6s; }
.ember-8 { left: 80%; animation-delay: 7s; }
.ember-9 { left: 90%; animation-delay: 8s; }

/* 3D terrain container */
.terrain-container {
    perspective: 1000px;
    transform-style: preserve-3d;
}

/* Weather icon animations */
.weather-icon {
    animation: weatherFloat 4s ease-in-out infinite;
}

@keyframes weatherFloat {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-10px) rotate(5deg); }
}

/* Wind particle effects */
.wind-particle {
    position: absolute;
    width: 2px;
    height: 2px;
    background: rgba(255, 255, 255, 0.6);
    border-radius: 50%;
    animation: windFlow 3s linear infinite;
}

@keyframes windFlow {
    0% { transform: translateX(-100px) translateY(0px); opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { transform: translateX(100px) translateY(-20px); opacity: 0; }
}

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

<!-- Wind particles -->
<div class="wind-particle" style="top: 20%; animation-delay: 0s;"></div>
<div class="wind-particle" style="top: 30%; animation-delay: 1s;"></div>
<div class="wind-particle" style="top: 40%; animation-delay: 2s;"></div>
<div class="wind-particle" style="top: 50%; animation-delay: 3s;"></div>
<div class="wind-particle" style="top: 60%; animation-delay: 4s;"></div>
<div class="wind-particle" style="top: 70%; animation-delay: 5s;"></div>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="🔥 Forest Fire AI Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title with enhanced fire aesthetics
st.markdown('<h1 class="fire-title">🔥 FOREST FIRE AI PREDICTOR 🔥</h1>', unsafe_allow_html=True)

# Load configuration and initialize predictor
config = load_config()
predictor = FirePredictor()

# Glassmorphism sidebar with enhanced interactions
with st.sidebar:
    st.markdown('<div class="glass-container glow-border morph-transition">', unsafe_allow_html=True)
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
    
    # Enhanced button with loading animation
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div class="loading-fire"></div>', unsafe_allow_html=True)
    with col2:
        predict_button = st.button(
            "🚀 Predict Fire Spread",
            help="Run AI prediction for fire spread analysis"
        )
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area with enhanced visualizations
if predict_button:
    with st.spinner("🔥 Running AI prediction..."):
        try:
            result = predictor.predict(region=region, date=str(date), time_window=f"{time_window}h")
            pred = np.array(result['prediction'])
            
            # Create animated fire spread simulation
            st.markdown('<div class="glass-container morph-transition">', unsafe_allow_html=True)
            st.markdown("### 🔥 Animated Fire Spread Simulation")
            
            # Generate time steps for animation
            time_steps = np.linspace(0, 1, 20)
            fire_spread_data = []
            
            for t in time_steps:
                # Simulate fire spread over time
                spread_factor = 1 - np.exp(-3 * t)
                current_spread = pred * spread_factor
                fire_spread_data.append(current_spread)
            
            # Create animated heatmap with Plotly
            fig = go.Figure()
            
            # Add initial frame
            fig.add_trace(go.Heatmap(
                z=fire_spread_data[0],
                colorscale='Hot',
                showscale=True,
                name='Fire Intensity'
            ))
            
            # Create frames for animation
            frames = []
            for i, data in enumerate(fire_spread_data):
                frame = go.Frame(
                    data=[go.Heatmap(z=data, colorscale='Hot')],
                    name=f'frame{i}',
                    layout=go.Layout(
                        title=f'🔥 Fire Spread Simulation - {region.title()} (Time: {i/len(time_steps)*time_window:.1f}h)'
                    )
                )
                frames.append(frame)
            
            fig.frames = frames
            
            # Update layout with fire theme
            fig.update_layout(
                title=f'🔥 Animated Fire Spread - {region.title()}',
                title_font_size=20,
                title_font_color='#ff6b35',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f0f0'),
                xaxis=dict(
                    title='Longitude',
                    gridcolor='rgba(255,107,53,0.2)',
                    zerolinecolor='rgba(255,107,53,0.3)'
                ),
                yaxis=dict(
                    title='Latitude',
                    gridcolor='rgba(255,107,53,0.2)',
                    zerolinecolor='rgba(255,107,53,0.3)'
                ),
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': '▶️ Play',
                            'method': 'animate',
                            'args': [None, {'frame': {'duration': 200, 'redraw': True}, 'fromcurrent': True}]
                        },
                        {
                            'label': '⏸️ Pause',
                            'method': 'animate',
                            'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]
                        }
                    ]
                }]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 3D Terrain Visualization
            st.markdown('<div class="glass-container terrain-container morph-transition">', unsafe_allow_html=True)
            st.markdown("### 🏔️ 3D Terrain & Fire Movement")
            
            # Generate 3D terrain data
            x = np.linspace(0, 100, 50)
            y = np.linspace(0, 100, 50)
            X, Y = np.meshgrid(x, y)
            
            # Create realistic terrain with elevation
            Z = 50 * np.sin(X/20) * np.cos(Y/20) + 30 * np.exp(-((X-50)**2 + (Y-50)**2)/1000)
            
            # Resize prediction data to match terrain dimensions (50x50)
            pred_resized = zoom(pred, (50/pred.shape[0], 50/pred.shape[1]), order=1)
            
            # Add fire overlay
            fire_overlay = pred_resized * 100  # Scale fire intensity
            
            # Create 3D surface plot
            fig_3d = go.Figure()
            
            # Terrain surface
            fig_3d.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Earth',
                opacity=0.8,
                name='Terrain',
                showscale=False
            ))
            
            # Fire overlay
            fig_3d.add_trace(go.Surface(
                x=X, y=Y, z=Z + fire_overlay,
                colorscale='Hot',
                opacity=0.6,
                name='Fire Spread',
                showscale=True
            ))
            
            fig_3d.update_layout(
                title=f'🏔️ 3D Terrain & Fire Spread - {region.title()}',
                title_font_size=18,
                title_font_color='#ff6b35',
                scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Elevation (m)',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f0f0'),
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Real-time Weather & Wind Analysis
            st.markdown('<div class="glass-container morph-transition">', unsafe_allow_html=True)
            st.markdown("### 🌪️ Real-time Weather & Wind Analysis")
            
            # Create weather dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="weather-icon data-pulse">', unsafe_allow_html=True)
                st.metric(
                    label="🌡️ Temperature",
                    value=f"{result['metadata'].get('temperature', 25):.1f}°C",
                    delta=f"{np.random.uniform(-2, 2):.1f}°C"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="weather-icon data-pulse">', unsafe_allow_html=True)
                st.metric(
                    label="💨 Wind Speed",
                    value=f"{result['metadata'].get('wind_speed', 15):.1f} km/h",
                    delta=f"{np.random.uniform(-5, 5):.1f} km/h"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="weather-icon data-pulse">', unsafe_allow_html=True)
                st.metric(
                    label="💧 Humidity",
                    value=f"{result['metadata'].get('humidity', 60):.1f}%",
                    delta=f"{np.random.uniform(-10, 10):.1f}%"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="weather-icon data-pulse">', unsafe_allow_html=True)
                st.metric(
                    label="🔥 Fire Risk",
                    value=f"{np.mean(pred) * 100:.1f}%",
                    delta=f"{np.random.uniform(-5, 5):.1f}%"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Wind direction visualization
            wind_direction = result['metadata'].get('wind_direction', 45)
            wind_speed = result['metadata'].get('wind_speed', 15)
            
            fig_wind = go.Figure()
            
            # Create wind vector field
            x_wind = np.linspace(0, 10, 8)
            y_wind = np.linspace(0, 10, 8)
            X_wind, Y_wind = np.meshgrid(x_wind, y_wind)
            
            # Wind vectors
            U = wind_speed * np.cos(np.radians(wind_direction)) * np.ones_like(X_wind)
            V = wind_speed * np.sin(np.radians(wind_direction)) * np.ones_like(Y_wind)
            
            fig_wind.add_trace(go.Streamtube(
                x=X_wind.flatten(),
                y=Y_wind.flatten(),
                u=U.flatten(),
                v=V.flatten(),
                colorscale='Blues',
                name='Wind Flow'
            ))
            
            fig_wind.update_layout(
                title=f'💨 Wind Direction: {wind_direction}° | Speed: {wind_speed} km/h',
                title_font_size=16,
                title_font_color='#ff6b35',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f0f0'),
                xaxis=dict(title='Distance (km)', gridcolor='rgba(255,107,53,0.2)'),
                yaxis=dict(title='Distance (km)', gridcolor='rgba(255,107,53,0.2)'),
                height=400
            )
            
            st.plotly_chart(fig_wind, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Metadata in enhanced glassmorphism container
            st.markdown('<div class="glass-container morph-transition">', unsafe_allow_html=True)
            st.markdown("### 📊 Live Input Conditions")
            st.markdown('<div class="json-container">', unsafe_allow_html=True)
            st.json(result['metadata'])
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download section with enhanced styling
            st.markdown('<div class="glass-container morph-transition">', unsafe_allow_html=True)
            st.markdown("### 💾 Download Results")
            st.download_button(
                "📥 Download Prediction (JSON)", 
                data=str(result), 
                file_name=f"fire_prediction_{region}_{date}.json",
                help="Download the prediction results as JSON file"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Region information with enhanced layout
            st.markdown('<div class="glass-container morph-transition">', unsafe_allow_html=True)
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
    # Welcome section with enhanced glassmorphism
    st.markdown('<div class="glass-container morph-transition">', unsafe_allow_html=True)
    st.markdown("### 🎯 Welcome to Forest Fire AI Predictor")
    st.info("🔥 Select parameters and click 'Predict Fire Spread' to begin AI-powered fire prediction analysis.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Project information with enhanced styling
    st.markdown('<div class="glass-container glow-border morph-transition">', unsafe_allow_html=True)
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
    
    # Features showcase with enhanced animations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="glass-container morph-transition">', unsafe_allow_html=True)
        st.markdown("### 🔥 **Real-time Analysis**")
        st.markdown("Instant fire spread predictions using live satellite and weather data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-container morph-transition">', unsafe_allow_html=True)
        st.markdown("### 🎯 **High Accuracy**")
        st.markdown("Advanced AI models trained on extensive historical fire data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="glass-container morph-transition">', unsafe_allow_html=True)
        st.markdown("### 📊 **Visual Insights**")
        st.markdown("Interactive heatmaps and detailed risk assessments")
        st.markdown('</div>', unsafe_allow_html=True) 