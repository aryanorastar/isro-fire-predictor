"""
Data preprocessing module for Forest Fire Spread Prediction
Handles satellite data, weather data, and terrain feature processing
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from shapely.geometry import box, Point
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import json

from .utils import (
    load_config, normalize_data, calculate_ndvi, calculate_slope,
    resize_raster, get_weather_data, setup_logging
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Main data preprocessing class for forest fire prediction."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        setup_logging(self.config)
        self.data_dir = self.config["paths"]["data_dir"]
        self.input_size = tuple(self.config["data"]["input_size"])
        
    def load_satellite_data(self, region: str, date: str) -> Dict[str, np.ndarray]:
        """Load satellite data for a specific region and date."""
        logger.info(f"Loading satellite data for {region} on {date}")
        
        # For demonstration, we'll create synthetic satellite data
        # In production, this would load from actual satellite sources
        size = self.input_size
        
        # Simulate Sentinel-2 data (10 bands)
        sentinel2_data = {
            "B2": np.random.rand(*size).astype(np.float32),  # Blue
            "B3": np.random.rand(*size).astype(np.float32),  # Green
            "B4": np.random.rand(*size).astype(np.float32),  # Red
            "B8": np.random.rand(*size).astype(np.float32),  # NIR
            "B11": np.random.rand(*size).astype(np.float32), # SWIR1
            "B12": np.random.rand(*size).astype(np.float32), # SWIR2
        }
        
        # Simulate LISS-4 data (3 bands)
        liss4_data = {
            "B2": np.random.rand(*size).astype(np.float32),  # Green
            "B3": np.random.rand(*size).astype(np.float32),  # Red
            "B4": np.random.rand(*size).astype(np.float32),  # NIR
        }
        
        return {
            "sentinel2": sentinel2_data,
            "liss4": liss4_data
        }
    
    def load_terrain_data(self, region: str) -> Dict[str, np.ndarray]:
        """Load terrain data including elevation, slope, and aspect."""
        logger.info(f"Loading terrain data for {region}")
        
        size = self.input_size
        
        # Simulate SRTM elevation data
        elevation = np.random.rand(*size) * 3000  # 0-3000m elevation
        
        # Calculate derived terrain features
        slope = calculate_slope(elevation)
        aspect = self._calculate_aspect(elevation)
        
        return {
            "elevation": elevation.astype(np.float32),
            "slope": slope.astype(np.float32),
            "aspect": aspect.astype(np.float32)
        }
    
    def _calculate_aspect(self, elevation: np.ndarray) -> np.ndarray:
        """Calculate aspect (direction of slope) from elevation data."""
        from scipy.ndimage import sobel
        
        sobel_x = sobel(elevation, axis=1)
        sobel_y = sobel(elevation, axis=0)
        
        aspect = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        aspect = (aspect + 360) % 360  # Convert to 0-360 degrees
        
        return aspect
    
    def calculate_vegetation_indices(self, satellite_data: Dict) -> Dict[str, np.ndarray]:
        """Calculate various vegetation indices from satellite data."""
        logger.info("Calculating vegetation indices")
        
        indices = {}
        
        # NDVI (Normalized Difference Vegetation Index)
        if "sentinel2" in satellite_data:
            nir = satellite_data["sentinel2"]["B8"]
            red = satellite_data["sentinel2"]["B4"]
            indices["ndvi"] = calculate_ndvi(nir, red)
        
        # EVI (Enhanced Vegetation Index)
        if "sentinel2" in satellite_data:
            nir = satellite_data["sentinel2"]["B8"]
            red = satellite_data["sentinel2"]["B4"]
            blue = satellite_data["sentinel2"]["B2"]
            indices["evi"] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        
        # NBR (Normalized Burn Ratio)
        if "sentinel2" in satellite_data:
            nir = satellite_data["sentinel2"]["B8"]
            swir2 = satellite_data["sentinel2"]["B12"]
            indices["nbr"] = (nir - swir2) / (nir + swir2 + 1e-8)
        
        return indices
    
    def load_weather_data(self, region: str, date: str) -> Dict[str, float]:
        """Load weather data for a specific region and date."""
        logger.info(f"Loading weather data for {region} on {date}")
        
        # Get region center coordinates
        region_config = self.config["regions"].get(region, {})
        if not region_config:
            raise ValueError(f"Region {region} not found in configuration")
        
        center = region_config["center"]
        lat, lon = center[0], center[1]
        
        # Try to get real weather data if API key is available
        api_key = self.config["api"].get("openweathermap_key")
        if api_key and api_key != "${OPENWEATHERMAP_API_KEY}":
            weather_data = get_weather_data(lat, lon, api_key)
            if weather_data:
                return weather_data
        
        # Fallback to synthetic weather data
        logger.warning("Using synthetic weather data")
        return {
            "temperature": np.random.uniform(15, 35),  # 15-35°C
            "humidity": np.random.uniform(30, 90),     # 30-90%
            "wind_speed": np.random.uniform(0, 20),    # 0-20 m/s
            "wind_direction": np.random.uniform(0, 360), # 0-360°
            "pressure": np.random.uniform(1000, 1025), # 1000-1025 hPa
            "precipitation": np.random.uniform(0, 50)  # 0-50 mm
        }
    
    def create_fire_history_data(self, region: str, days_back: int = 30) -> np.ndarray:
        """Create historical fire data for the region."""
        logger.info(f"Creating fire history data for {region} (last {days_back} days)")
        
        size = self.input_size
        fire_history = np.zeros((days_back, *size), dtype=np.float32)
        
        # Simulate some historical fire events
        for day in range(days_back):
            # Random fire events with decreasing probability for older days
            fire_prob = 0.1 * np.exp(-day / 10)  # Exponential decay
            
            if np.random.random() < fire_prob:
                # Create a fire event
                fire_center = (
                    np.random.randint(0, size[0]),
                    np.random.randint(0, size[1])
                )
                
                # Create fire spread pattern
                for i in range(size[0]):
                    for j in range(size[1]):
                        distance = np.sqrt((i - fire_center[0])**2 + (j - fire_center[1])**2)
                        intensity = np.exp(-distance / 20)  # Gaussian spread
                        fire_history[day, i, j] = intensity
        
        return fire_history
    
    def prepare_model_input(self, region: str, date: str) -> Dict[str, np.ndarray]:
        """Prepare complete model input from all data sources."""
        logger.info(f"Preparing model input for {region} on {date}")
        
        # Load all data sources
        satellite_data = self.load_satellite_data(region, date)
        terrain_data = self.load_terrain_data(region)
        weather_data = self.load_weather_data(region, date)
        vegetation_indices = self.calculate_vegetation_indices(satellite_data)
        fire_history = self.create_fire_history_data(region)
        
        # Create multi-channel input tensor
        channels = []
        
        # Satellite channels (6 bands from Sentinel-2)
        for band in ["B2", "B3", "B4", "B8", "B11", "B12"]:
            channels.append(satellite_data["sentinel2"][band])
        
        # Terrain channels
        channels.append(terrain_data["elevation"])
        channels.append(terrain_data["slope"])
        channels.append(terrain_data["aspect"])
        
        # Vegetation indices
        channels.append(vegetation_indices["ndvi"])
        channels.append(vegetation_indices["evi"])
        channels.append(vegetation_indices["nbr"])
        
        # Weather channels (spatialized)
        weather_channels = self._spatialize_weather(weather_data, self.input_size)
        channels.extend(weather_channels)
        
        # Fire history (most recent day)
        channels.append(fire_history[-1])
        
        # Stack all channels
        input_tensor = np.stack(channels, axis=0)
        
        # Normalize each channel
        input_tensor = self._normalize_channels(input_tensor)
        
        return {
            "input_tensor": input_tensor,
            "satellite_data": satellite_data,
            "terrain_data": terrain_data,
            "weather_data": weather_data,
            "vegetation_indices": vegetation_indices,
            "fire_history": fire_history
        }
    
    def _spatialize_weather(self, weather_data: Dict, size: Tuple[int, int]) -> List[np.ndarray]:
        """Convert point weather data to spatial arrays."""
        channels = []
        
        # Temperature (add some spatial variation)
        temp_base = weather_data["temperature"]
        temp_spatial = temp_base + np.random.normal(0, 2, size)  # ±2°C variation
        channels.append(temp_spatial.astype(np.float32))
        
        # Humidity
        hum_base = weather_data["humidity"]
        hum_spatial = hum_base + np.random.normal(0, 5, size)  # ±5% variation
        channels.append(hum_spatial.astype(np.float32))
        
        # Wind speed
        wind_base = weather_data["wind_speed"]
        wind_spatial = wind_base + np.random.normal(0, 1, size)  # ±1 m/s variation
        channels.append(wind_spatial.astype(np.float32))
        
        # Wind direction (convert to u and v components)
        wind_dir = weather_data["wind_direction"] * np.pi / 180
        u_wind = wind_base * np.cos(wind_dir)
        v_wind = wind_base * np.sin(wind_dir)
        
        u_spatial = u_wind + np.random.normal(0, 0.5, size)
        v_spatial = v_wind + np.random.normal(0, 0.5, size)
        
        channels.append(u_spatial.astype(np.float32))
        channels.append(v_spatial.astype(np.float32))
        
        return channels
    
    def _normalize_channels(self, input_tensor: np.ndarray) -> np.ndarray:
        """Normalize each channel of the input tensor."""
        normalized = np.zeros_like(input_tensor)
        
        for i in range(input_tensor.shape[0]):
            channel = input_tensor[i]
            
            # Skip normalization for binary or categorical data
            if i == input_tensor.shape[0] - 1:  # Fire history channel
                normalized[i] = channel
            else:
                normalized[i] = normalize_data(channel, method="minmax")
        
        return normalized
    
    def create_training_dataset(self, regions: List[str], 
                               start_date: str, 
                               end_date: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create training dataset from multiple regions and dates."""
        logger.info(f"Creating training dataset for regions: {regions}")
        
        inputs = []
        targets = []
        
        # Generate dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = [start + timedelta(days=i) for i in range((end - start).days)]
        
        for region in tqdm(regions, desc="Processing regions"):
            for date in tqdm(dates, desc=f"Processing dates for {region}"):
                try:
                    # Prepare input
                    data = self.prepare_model_input(region, date.strftime("%Y-%m-%d"))
                    input_tensor = data["input_tensor"]
                    
                    # Create synthetic target (fire spread prediction)
                    target = self._create_synthetic_target(input_tensor, region, date)
                    
                    inputs.append(input_tensor)
                    targets.append(target)
                    
                except Exception as e:
                    logger.warning(f"Error processing {region} on {date}: {e}")
                    continue
        
        return np.array(inputs), np.array(targets)
    
    def _create_synthetic_target(self, input_tensor: np.ndarray, 
                                region: str, date: datetime) -> np.ndarray:
        """Create synthetic fire spread target for training."""
        size = self.input_size
        
        # Use fire history and risk factors to create realistic targets
        fire_history = input_tensor[-1]  # Last channel is fire history
        ndvi = input_tensor[8]  # NDVI channel
        slope = input_tensor[7]  # Slope channel
        temperature = input_tensor[11]  # Temperature channel
        humidity = input_tensor[12]  # Humidity channel
        wind_speed = input_tensor[13]  # Wind speed channel
        
        # Create fire risk map
        risk_map = (
            0.3 * (1 - ndvi) +      # Lower vegetation = higher risk
            0.2 * slope +           # Steeper slopes = higher risk
            0.2 * (temperature / 50) +  # Higher temperature = higher risk
            0.2 * (1 - humidity / 100) +  # Lower humidity = higher risk
            0.1 * (wind_speed / 20)   # Higher wind = higher risk
        )
        
        # Add some fire spread from existing fires
        target = fire_history.copy()
        
        # Simulate fire spread
        for i in range(size[0]):
            for j in range(size[1]):
                if fire_history[i, j] > 0.1:  # Existing fire
                    # Spread to neighboring pixels
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < size[0] and 0 <= nj < size[1] and 
                                (di != 0 or dj != 0)):
                                spread_prob = risk_map[ni, nj] * 0.3
                                if np.random.random() < spread_prob:
                                    target[ni, nj] = min(1.0, target[ni, nj] + 0.2)
        
        return target.astype(np.float32)
    
    def save_preprocessed_data(self, data: Dict, output_path: str) -> None:
        """Save preprocessed data to disk."""
        logger.info(f"Saving preprocessed data to {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as numpy arrays
        np.savez_compressed(
            output_path,
            input_tensor=data["input_tensor"],
            satellite_data=data["satellite_data"],
            terrain_data=data["terrain_data"],
            weather_data=data["weather_data"],
            vegetation_indices=data["vegetation_indices"],
            fire_history=data["fire_history"]
        )
    
    def load_preprocessed_data(self, file_path: str) -> Dict:
        """Load preprocessed data from disk."""
        logger.info(f"Loading preprocessed data from {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        
        return {
            "input_tensor": data["input_tensor"],
            "satellite_data": data["satellite_data"].item(),
            "terrain_data": data["terrain_data"].item(),
            "weather_data": data["weather_data"].item(),
            "vegetation_indices": data["vegetation_indices"].item(),
            "fire_history": data["fire_history"]
        }


def main():
    """Main function for testing the preprocessor."""
    preprocessor = DataPreprocessor()
    
    # Test with a sample region and date
    region = "uttarakhand"
    date = "2024-01-15"
    
    try:
        data = preprocessor.prepare_model_input(region, date)
        print(f"Input tensor shape: {data['input_tensor'].shape}")
        print(f"Number of channels: {data['input_tensor'].shape[0]}")
        print(f"Weather data: {data['weather_data']}")
        
        # Save sample data
        preprocessor.save_preprocessed_data(data, "data/sample_preprocessed.npz")
        print("Sample data saved successfully!")
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")


if __name__ == "__main__":
    main() 