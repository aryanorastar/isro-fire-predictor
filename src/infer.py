"""
Inference script for Forest Fire Spread Prediction
"""

import os
import torch
import numpy as np
from .utils import load_config, setup_device, format_prediction_results
from .preprocess import DataPreprocessor
from .train import TransformerCAHybrid

class FirePredictor:
    def __init__(self, model_path=None, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.device = setup_device()
        self.preprocessor = DataPreprocessor(config_path)
        self.model_path = model_path or os.path.join(self.config['paths']['models_dir'], 'best_model.pt')
        
        # Initialize model
        self.model = TransformerCAHybrid(
            in_channels=self.config['model']['input_channels'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            ca_iterations=self.config['model']['ca_iterations'],
            img_size=tuple(self.config['data']['input_size'])
        ).to(self.device)
        
        # Try to load model, but don't fail if it doesn't exist
        try:
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"Model loaded from {self.model_path}")
            else:
                print(f"[WARNING] Model file {self.model_path} not found. Using untrained model for demo.")
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}. Using untrained model for demo.")
        
        self.model.eval()

    def predict(self, region, date, time_window="24h"):
        # Preprocess input
        data = self.preprocessor.prepare_model_input(region, date)
        x = torch.tensor(data['input_tensor']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # If model is untrained, create a synthetic prediction based on input data
            if not os.path.exists(self.model_path):
                pred = self._create_demo_prediction(data, region, date)
            else:
                pred = self.model(x).cpu().numpy()[0, 0]
        
        # Calculate confidence based on prediction variance
        confidence = float(np.mean(pred))
        return format_prediction_results(pred, confidence, region, time_window)
    
    def _create_demo_prediction(self, data, region, date):
        """Create a realistic demo prediction based on input data."""
        input_tensor = data['input_tensor']
        
        # Extract key features for demo prediction
        ndvi = input_tensor[8]  # NDVI channel
        slope = input_tensor[7]  # Slope channel
        temperature = input_tensor[11]  # Temperature channel
        humidity = input_tensor[12]  # Humidity channel
        wind_speed = input_tensor[13]  # Wind speed channel
        fire_history = input_tensor[-1]  # Fire history channel
        
        # Create fire risk map
        risk_map = (
            0.3 * (1 - ndvi) +      # Lower vegetation = higher risk
            0.2 * slope +           # Steeper slopes = higher risk
            0.2 * (temperature / 50) +  # Higher temperature = higher risk
            0.2 * (1 - humidity / 100) +  # Lower humidity = higher risk
            0.1 * (wind_speed / 20)   # Higher wind = higher risk
        )
        
        # Add some fire spread from existing fires
        prediction = fire_history.copy()
        
        # Simulate fire spread based on risk factors
        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                if fire_history[i, j] > 0.1:  # Existing fire
                    # Spread to neighboring pixels
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < prediction.shape[0] and 0 <= nj < prediction.shape[1] and 
                                (di != 0 or dj != 0)):
                                spread_prob = risk_map[ni, nj] * 0.3
                                if np.random.random() < spread_prob:
                                    prediction[ni, nj] = min(1.0, prediction[ni, nj] + 0.2)
        
        # Add some noise and smooth the prediction
        prediction = prediction + np.random.normal(0, 0.05, prediction.shape)
        prediction = np.clip(prediction, 0, 1)
        
        return prediction.astype(np.float32)

if __name__ == "__main__":
    predictor = FirePredictor()
    result = predictor.predict(region="uttarakhand", date="2024-01-15", time_window="24h")
    print("Prediction result:", result) 