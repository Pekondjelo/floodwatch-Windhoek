# Import required libraries
import rasterio
import geopandas as gpd
from rasterio.mask import mask
import numpy as np
from shapely.geometry import mapping
from pathlib import Path
import streamlit as st
from streamlit_folium import st_folium
import folium
from rasterio.warp import transform_bounds
from PIL import Image
import io
import requests
import zipfile
import os
import base64

# === Configuration ===
BASE_DIR = Path("data")
BASE_DIR.mkdir(exist_ok=True)
SHAPEFILE_ZIP_URL = "https://drive.google.com/uc?export=download&id=1wR2JEe0xQ1LWVaO47F74G03VcZXQYVIX"
SHAPEFILE_NAME = "Windhoek_Urban.shp"  # Matches local file; update if ZIP contains different .shp
LOCAL_SHAPEFILE_NAME = "Windhoek_Urban.shp"
DEM_URL = "https://drive.google.com/uc?export=download&id=1lUXCX01maYkKKn-VMclDU2KdNrJk4NFh"
DEM_PATH = BASE_DIR / "whk_raster.tif"
OUTPUT_PATH = "flood_prediction_windhoek.tif"

# Streamlit page configuration
st.set_page_config(page_title="FloodWatch", page_icon="🌊", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {background-color: #f0f5ff;}
        .sidebar .sidebar-content {background-color: #e6f0ff;}
        .stButton>button {background-color: #ff4d4d; color: white;}
        .stSlider .stSliderLabel {color: #004aad;}
        h1 {color: #004aad; font-family: 'Noto Sans', sans-serif;}
        .status-success {color: #28a745; font-weight: bold;}
        .status-error {color: #dc3545; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🌊 FloodWatch")
st.markdown("**Predicting Flood Risks for Windhoek**")
st.markdown("Use the slider to adjust the elevation threshold and visualize flood-prone areas in Windhoek.")

# Sidebar for inputs and metadata
with st.sidebar:
    st.header("FloodWatch Controls")
    st.markdown("Adjust the elevation threshold to predict flood-prone areas.")
    
    threshold_placeholder = st.empty()
    
    st.markdown("### DEM Metadata")
    metadata_placeholder = st.empty()
    
    st.markdown("### Instructions")
    st.write("- Adjust the elevation threshold to see flood risks.")
    st.write("- Red areas indicate flood-prone zones.")
    st.write("- Blue outline shows the Windhoek urban boundary.")
    st.write("- Data files are downloaded from Google Drive if not local.")

# === STEP 1: Download or use local data files ===
try:
    # Check for local shapefile first
    SHAPEFILE_PATH = BASE_DIR / LOCAL_SHAPEFILE_NAME
    if not SHAPEFILE_PATH.exists():
        # Download and extract shapefile as ZIP
        shapefile_zip_path = BASE_DIR / "Windhoek_Urban.zip"
        try:
            st.markdown('<p class="status-success">Downloading shapefile...</p>', unsafe_allow_html=True)
            response = requests.get(SHAPEFILE_ZIP_URL)
            if response.status_code != 200:
                raise Exception("Failed to download shapefile ZIP from Google Drive")
            with open(shapefile_zip_path, "wb") as f:
                f.write(response.content)
            
            # Extract ZIP
            with zipfile.ZipFile(shapefile_zip_path, "r") as zip_ref:
                zip_ref.extractall(BASE_DIR)
            os.remove(shapefile_zip_path)  # Clean up ZIP file
            SHAPEFILE_PATH = BASE_DIR / SHAPEFILE_NAME
        except zipfile.BadZipFile:
            # Fallback: Try downloading as direct .shp
            st.markdown('<p class="status-success">Trying direct shapefile download...</p>', unsafe_allow_html=True)
            SHAPEFILE_PATH = BASE_DIR / SHAPEFILE_NAME
            response = requests.get(SHAPEFILE_ZIP_URL)
            if response.status_code != 200:
                raise Exception("Failed to download shapefile directly")
            with open(SHAPEFILE_PATH, "wb") as f:
                f.write(response.content)
        
        if not SHAPEFILE_PATH.exists():
            raise FileNotFoundError(f"Shapefile {SHAPEFILE_NAME} not found")

    # Download DEM if not present
    if not DEM_PATH.exists():
        st.markdown('<p class="status-success">Downloading DEM...</p>', unsafe_allow_html=True)
        response = requests.get(DEM_URL)
        if response.status_code != 200:
            raise Exception("Failed to download DEM from Google Drive")
        with open(DEM_PATH, "wb") as f:
            f.write(response.content)
    
    st.markdown('<p class="status-success">✅ Data files ready</p>', unsafe_allow_html=True)
except Exception as e:
    st.markdown(f'<p class="status-error">❌ Error preparing data files: {str(e)}</p>', unsafe_allow_html=True)
    st.warning("Ensure Google Drive links are publicly accessible or local files are in the 'data' folder.")
    st.stop()

# === STEP 2: Load shapefile (Area of Interest) ===
try:
    aoi = gpd.read_file(SHAPEFILE_PATH)
    
    if aoi.geometry.is_empty.all() or aoi.geometry.is_valid.all() == False:
        raise ValueError("Invalid or empty geometry in shapefile.")
    st.markdown('<p class="status-success">✅ Shapefile loaded successfully</p>', unsafe_allow_html=True)
except Exception as e:
    st.markdown(f'<p class="status-error">❌ Error loading shapefile: {str(e)}</p>', unsafe_allow_html=True)
    st.stop()

# === STEP 3: Load DEM and clip to shapefile ===
try:
    with rasterio.open(DEM_PATH) as src:
        metadata_placeholder.write("**Raster Metadata**:")
        metadata_placeholder.write(src.meta)
        
        dem_crs = src.crs
        if dem_crs != aoi.crs:
            st.warning(f"⚠️ CRS mismatch: DEM ({dem_crs}) differs from shapefile ({aoi.crs}). Reprojecting shapefile.")
            aoi = aoi.to_crs(dem_crs)

        clipped_dem, out_transform = mask(src, [mapping(geom) for geom in aoi.geometry], crop=True, nodata=src.nodata)
        out_meta = src.meta.copy()

        if not np.any(clipped_dem != src.nodata):
            raise ValueError("Clipped DEM is empty or contains only nodata values.")
        
        valid_mask = clipped_dem != src.nodata
    st.markdown('<p class="status-success">✅ DEM clipped successfully</p>', unsafe_allow_html=True)
except Exception as e:
    st.markdown(f'<p class="status-error">❌ Error processing DEM: {str(e)}</p>', unsafe_allow_html=True)
    st.stop()

# === STEP 4: Compute valid elevation range ===
elevation_data = clipped_dem[0]
valid_elevations = elevation_data[valid_mask[0]]
min_elevation = float(np.min(valid_elevations))
max_elevation = float(np.max(valid_elevations))
st.write(f"**Valid Elevation Range**: Min = {min_elevation:.2f} m, Max = {max_elevation:.2f} m")

# Dynamic elevation threshold slider
ELEVATION_THRESHOLD = threshold_placeholder.slider(
    "Elevation Threshold (meters)",
    min_value=min_elevation,
    max_value=max_elevation,
    value=1600.0,
    step=1.0
)

# === STEP 5: Flood prediction ===
flood_prediction = np.zeros_like(elevation_data, dtype=np.uint8)
flood_prediction[valid_mask[0] & (elevation_data < ELEVATION_THRESHOLD)] = 1
st.markdown('<p class="status-success">✅ Flood prediction generated</p>', unsafe_allow_html=True)
st.write(f"**Flood-Prone Pixels**: {np.sum(flood_prediction)}")

# === STEP 6: Save flood prediction map ===
out_meta.update({
    "driver": "GTiff",
    "height": flood_prediction.shape[0],
    "width": flood_prediction.shape[1],
    "transform": out_transform,
    "count": 1,
    "dtype": 'uint8',
    "nodata": 255
})

try:
    with rasterio.open(OUTPUT_PATH, "w", **out_meta) as dest:
        dest.write(flood_prediction, 1)
    st.markdown(f'<p class="status-success">✅ Flood prediction map saved at: {OUTPUT_PATH}</p>', unsafe_allow_html=True)
    
    with open(OUTPUT_PATH, "rb") as f:
        st.download_button("Download Flood Prediction Raster", f, file_name="flood_prediction_windhoek.tif")
except Exception as e:
    st.markdown(f'<p class="status-error">❌ Error saving flood prediction map: {str(e)}</p>', unsafe_allow_html=True)
    st.stop()

# === STEP 7: Create interactive Folium map ===
try:
    with rasterio.open(OUTPUT_PATH) as flood_raster:
        bounds = flood_raster.bounds
        crs = flood_raster.crs

    bounds_wgs84 = transform_bounds(crs, 'EPSG:4326', *bounds)
    sw = [bounds_wgs84[1], bounds_wgs84[0]]
    ne = [bounds_wgs84[3], bounds_wgs84[2]]

    flood_image = flood_prediction.copy()
    rgba = np.zeros((flood_image.shape[0], flood_image.shape[1], 4), dtype=np.uint8)
    rgba[flood_image == 1] = [255, 0, 0, 128]
    rgba[flood_image == 0] = [0, 0, 0, 0]

    img = Image.fromarray(rgba, mode='RGBA')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)

    # Convert image to base64 for Folium
    img_data = img_buffer.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    img_url = f"data:image/png;base64,{img_base64}"

    map_center = [(sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2]
    m = folium.Map(location=map_center, zoom_start=10, tiles='CartoDB positron')

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[sw, ne],
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1,
        name="Flood-Prone Areas"
    ).add_to(m)

    aoi_wgs84 = aoi.to_crs('EPSG:4326')
    folium.GeoJson(
        aoi_wgs84,
        style_function=lambda x: {'color': 'blue', 'weight': 2, 'fillOpacity': 0},
        name="Windhoek Urban Boundary"
    ).add_to(m)

    folium.LayerControl().add_to(m)

    st.subheader("Interactive Flood Risk Map")
    st_folium(m, width=700, height=500, key="flood_map")
    st.markdown('<p class="status-success">✅ Interactive flood prediction map displayed</p>', unsafe_allow_html=True)
except Exception as e:
    st.markdown(f'<p class="status-error">❌ Error creating interactive map: {str(e)}</p>', unsafe_allow_html=True)
    st.stop()