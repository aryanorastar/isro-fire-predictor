import folium
import numpy as np

def add_heatmap_to_map(m, prediction, bounds, opacity=0.6):
    """Add a heatmap overlay to a folium map."""
    folium.raster_layers.ImageOverlay(
        image=prediction,
        bounds=bounds,
        opacity=opacity,
        colormap=lambda x: (1, 0, 0, x),  # Red heatmap
        name="Fire Spread Prediction"
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m

def get_region_bounds(region_config):
    """Extract bounding box from region config."""
    return [
        [region_config['bounds'][0], region_config['bounds'][1]],
        [region_config['bounds'][2], region_config['bounds'][3]]
    ] 