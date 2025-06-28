import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt
from src.infer import FirePredictor
from src.utils import load_config

st.set_page_config(page_title="Forest Fire Spread Prediction", layout="wide")
st.title("🔥 Forest Fire Spread Prediction Dashboard")

config = load_config()
predictor = FirePredictor()

# Sidebar controls
st.sidebar.header("Prediction Controls")
region = st.sidebar.selectbox("Select Region", list(config['regions'].keys()))
date = st.sidebar.date_input("Select Date", datetime.date.today())
time_window = st.sidebar.selectbox("Forecast Window (hours)", [6, 12, 24], index=2)

if st.sidebar.button("Predict Fire Spread"):
    with st.spinner("Running prediction..."):
        try:
            result = predictor.predict(region=region, date=str(date), time_window=f"{time_window}h")
            pred = np.array(result['prediction'])
            
            # Display prediction as heatmap using matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(pred, cmap='hot', interpolation='nearest')
            ax.set_title(f'Fire Spread Prediction - {region.title()}')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, label='Fire Probability')
            st.pyplot(fig)
            
            # Display metadata
            st.subheader("Live Input Conditions")
            st.json(result['metadata'])
            
            # Download button
            st.download_button(
                "Download Prediction (JSON)", 
                data=str(result), 
                file_name="prediction.json"
            )
            
            # Display region info
            st.subheader("Region Information")
            region_info = config['regions'][region]
            st.write(f"**Center Coordinates:** {region_info['center']}")
            st.write(f"**Bounds:** {region_info['bounds']}")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("This is a demo with synthetic data. In production, this would use real satellite and weather data.")
else:
    st.info("Select parameters and click 'Predict Fire Spread' to begin.")
    
    # Display project info
    st.markdown("""
    ## About This Project
    
    This Forest Fire Spread Prediction system uses:
    - **AI/ML Models**: Transformer + Cellular Automata hybrid
    - **Data Sources**: Satellite imagery, weather data, terrain features
    - **Regions**: Indian forest areas (Uttarakhand, Himachal Pradesh, Karnataka)
    - **Predictions**: 6, 12, and 24-hour fire spread forecasts
    
    Built for **Bharatiya Antariksh Hackathon 2025** 🚀
    """) 