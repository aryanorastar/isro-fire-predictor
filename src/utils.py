"""
Utility functions for Forest Fire Spread Prediction
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from shapely.geometry import box
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union
import requests
from datetime import datetime, timedelta
import json
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    # Get the directory containing this script (src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root
    project_root = os.path.dirname(current_dir)
    # Construct the full path to config.yaml
    full_config_path = os.path.join(project_root, config_path)
    
    with open(full_config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_device() -> torch.device:
    """Setup device (CPU/GPU) for training and inference."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """Calculate Normalized Difference Vegetation Index (NDVI)."""
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
    return np.clip(ndvi, -1, 1)


def calculate_slope(elevation: np.ndarray, pixel_size: float = 30.0) -> np.ndarray:
    """Calculate slope from elevation data using Sobel operators."""
    from scipy.ndimage import sobel
    
    # Sobel operators for gradient calculation
    sobel_x = sobel(elevation, axis=1) / (2 * pixel_size)
    sobel_y = sobel(elevation, axis=0) / (2 * pixel_size)
    
    # Calculate slope in degrees
    slope = np.arctan(np.sqrt(sobel_x**2 + sobel_y**2)) * 180 / np.pi
    return slope


def normalize_data(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize data using specified method."""
    if method == "minmax":
        return (data - data.min()) / (data.max() - data.min() + 1e-8)
    elif method == "zscore":
        return (data - data.mean()) / (data.std() + 1e-8)
    elif method == "robust":
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        return (data - q25) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_raster(raster: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize raster to target size using bilinear interpolation."""
    from scipy.ndimage import zoom
    
    if len(raster.shape) == 2:
        zoom_factors = (target_size[0] / raster.shape[0], target_size[1] / raster.shape[1])
        return zoom(raster, zoom_factors, order=1)
    else:
        zoom_factors = (1, target_size[0] / raster.shape[1], target_size[1] / raster.shape[2])
        return zoom(raster, zoom_factors, order=1)


def calculate_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU)."""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-8)


def calculate_f1_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate F1 score."""
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return 2 * (precision * recall) / (precision + recall + 1e-8)


def get_weather_data(lat: float, lon: float, api_key: str) -> Dict:
    """Fetch weather data from OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"]["deg"],
            "pressure": data["main"]["pressure"]
        }
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return {}


def create_fire_risk_index(ndvi: np.ndarray, slope: np.ndarray, 
                          temperature: float, humidity: float, 
                          wind_speed: float) -> np.ndarray:
    """Create a composite fire risk index."""
    # Normalize inputs
    ndvi_norm = normalize_data(ndvi)
    slope_norm = normalize_data(slope)
    
    # Weighted combination
    risk_index = (
        0.3 * (1 - ndvi_norm) +  # Lower vegetation = higher risk
        0.2 * slope_norm +       # Steeper slopes = higher risk
        0.2 * (temperature / 50) +  # Higher temperature = higher risk
        0.2 * (1 - humidity / 100) +  # Lower humidity = higher risk
        0.1 * (wind_speed / 20)   # Higher wind = higher risk
    )
    
    return np.clip(risk_index, 0, 1)


def save_prediction_as_geotiff(prediction: np.ndarray, 
                              output_path: str,
                              reference_raster: str) -> None:
    """Save prediction as GeoTIFF with proper georeference."""
    with rasterio.open(reference_raster) as src:
        profile = src.profile.copy()
    
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction.astype(rasterio.float32), 1)


def load_sample_data() -> Dict:
    """Load sample data for demonstration purposes."""
    # Create synthetic data for demonstration
    size = (256, 256)
    
    # Synthetic satellite data (4 bands: B, G, R, NIR)
    satellite_data = np.random.rand(4, *size).astype(np.float32)
    
    # Synthetic terrain data
    elevation = np.random.rand(*size) * 1000  # 0-1000m elevation
    slope = calculate_slope(elevation)
    ndvi = calculate_ndvi(satellite_data[3], satellite_data[2])  # NIR, Red
    
    # Synthetic weather data
    weather_data = {
        "temperature": 25.0,
        "humidity": 60.0,
        "wind_speed": 5.0,
        "wind_direction": 180.0,
        "pressure": 1013.25
    }
    
    return {
        "satellite": satellite_data,
        "elevation": elevation,
        "slope": slope,
        "ndvi": ndvi,
        "weather": weather_data
    }


def create_temporal_sequence(data: np.ndarray, time_steps: int) -> np.ndarray:
    """Create temporal sequence for time-series prediction."""
    if len(data.shape) == 3:
        # Single channel data
        sequence = np.zeros((time_steps, *data.shape))
        for t in range(time_steps):
            # Add some temporal variation
            noise = np.random.normal(0, 0.1, data.shape)
            sequence[t] = data + noise * t
    else:
        # Multi-channel data
        sequence = np.zeros((time_steps, *data.shape))
        for t in range(time_steps):
            noise = np.random.normal(0, 0.1, data.shape)
            sequence[t] = data + noise * t
    
    return sequence


def get_region_bounds(region_name: str, config: Dict) -> Tuple[float, float, float, float]:
    """Get bounding box for a specific region."""
    regions = config.get("regions", {})
    if region_name not in regions:
        raise ValueError(f"Region {region_name} not found in config")
    
    return tuple(regions[region_name]["bounds"])


def format_prediction_results(prediction: np.ndarray, 
                            confidence: float,
                            region: str,
                            time_window: str) -> Dict:
    """Format prediction results for API response."""
    return {
        "prediction": prediction.tolist(),
        "confidence": confidence,
        "region": region,
        "time_window": time_window,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "model_version": "1.0.0",
            "input_size": prediction.shape,
            "prediction_type": "fire_spread_probability"
        }
    }


def validate_input_data(data: Dict) -> bool:
    """Validate input data structure and types."""
    required_keys = ["satellite", "elevation", "slope", "ndvi", "weather"]
    
    for key in required_keys:
        if key not in data:
            logger.error(f"Missing required key: {key}")
            return False
    
    # Validate weather data
    weather = data["weather"]
    required_weather_keys = ["temperature", "humidity", "wind_speed"]
    
    for key in required_weather_keys:
        if key not in weather:
            logger.error(f"Missing weather key: {key}")
            return False
    
    return True


def setup_logging(config=None):
    """Setup logging configuration - disabled for Streamlit Cloud."""
    pass

def _is_streamlit_cloud():
    """Detect if running on Streamlit Cloud."""
    return (
        '/mount/src/' in os.getcwd() or 
        os.environ.get('STREAMLIT_CLOUD') is not None or
        os.environ.get('STREAMLIT_SHARING') is not None
    ) 