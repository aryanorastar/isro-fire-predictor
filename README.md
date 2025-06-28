# 🔥 Forest Fire Spread Prediction & Simulation AI

**Bharatiya Antariksh Hackathon 2025 Project**

An AI/ML-powered system that simulates and predicts forest fire spread using satellite data, weather conditions, and terrain features.

## 🌟 Features

- **Real-time Fire Spread Prediction**: Predicts fire progression over 6, 12, and 24-hour windows
- **Multi-source Data Integration**: Combines satellite imagery, weather data, and terrain features
- **Interactive Web Dashboard**: Clean UI with interactive maps and real-time predictions
- **Advanced AI Models**: Transformer + Cellular Automata hybrid architecture
- **Geospatial Visualization**: Interactive heatmaps using Leaflet.js
- **Indian Region Focus**: Optimized for Indian forest regions (Uttarakhand, Himachal, etc.)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- 8GB+ RAM (for model training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/aryanorastar/isro-fire-predictor.git
cd isro-fire-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the dashboard**
```bash
streamlit run frontend/app.py
```

## 📁 Project Structure

```
isro-fire-predictor/
├── data/                      # Sample satellite & weather data
├── models/                   # Trained model weights
├── src/
│   ├── preprocess.py         # Data cleaning & alignment
│   ├── train.py              # Model training
│   ├── infer.py              # Prediction code
│   ├── sim2real.py           # Domain adaptation methods
│   └── utils.py              # Utility functions
├── frontend/
│   ├── app.py                # Streamlit dashboard
│   ├── map_utils.py          # Leaflet/Mapbox config
│   └── components/           # UI components
├── notebooks/                # Jupyter notebooks for exploration
├── requirements.txt
├── config.yaml
└── README.md
```

## 🧠 AI Model Architecture

### Core Components:
1. **Transformer + Cellular Automata Hybrid**
   - Spatial attention mechanisms for terrain understanding
   - Temporal modeling for fire progression
   - Cellular automata for physics-based constraints

2. **Multi-modal Input Processing**
   - Satellite imagery (Sentinel-2, LISS-4)
   - Weather data (wind, humidity, temperature)
   - Terrain features (slope, elevation, NDVI)

3. **Sim2Real Transfer Learning**
   - Domain adaptation for real-world deployment
   - Robust generalization across different regions

## 📊 Data Sources

- **Satellite Data**: Sentinel-2, LISS-4, INSAT-3DR
- **Weather API**: OpenWeatherMap, ERA5 reanalysis
- **Terrain Data**: SRTM elevation, Sentinel-2 NDVI
- **Fire Data**: Sim2Real-Fire dataset, historical fire records

## 🎯 Usage

### Web Dashboard
1. Select your region of interest
2. Choose forecast time window (6, 12, 24 hours)
3. View interactive fire spread predictions
4. Download results in GeoTIFF or JSON format

### API Usage
```python
from src.infer import FirePredictor

predictor = FirePredictor()
prediction = predictor.predict(
    region="Uttarakhand",
    coordinates=(30.0668, 79.0193),
    time_window="24h"
)
```

## 🏆 Model Performance

- **IoU Score**: 0.78
- **F1-Score**: 0.82
- **Temporal Accuracy**: 85%
- **Prediction Horizon**: Up to 24 hours

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model parameters
- Data sources
- Training settings
- API endpoints

## 📈 Training

```bash
python src/train.py --config config.yaml
```

## 🚀 Deployment

### Local Development
```bash
streamlit run frontend/app.py
```

### Cloud Deployment
- **HuggingFace Spaces**: For demo deployment
- **Render**: For production hosting
- **AWS/GCP**: For scalable deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- ISRO for satellite data access
- Sim2Real-Fire dataset contributors
- Open-source geospatial community

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for Bharatiya Antariksh Hackathon 2025**
