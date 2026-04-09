# Databricks notebook source
# MAGIC %pip install geopandas shapely requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# TASK: Download and cache World Bank Official Boundaries
#
# Dependencies: None
# Outputs: Cached GeoJSON files in Volume (Admin0, Admin1, Admin2)
#
# This task downloads World Bank Official Boundaries for all admin levels
# and caches them locally. These are used by subsequent tasks:
# - 03_boundaries.py: Province/country boundary extraction
# - 03_boundaries.py: LGU (district) boundary extraction

# COMMAND ----------

import os
import requests
import geopandas as gpd

# COMMAND ----------

# Import shared utilities and configuration
from shared.utils import ensure_dir, file_exists
from extract.config import (
    VOLUME_DIR,
    WB_ADMIN0_URL,
    WB_ADMIN1_URL,
    WB_ADMIN2_URL,
)

# COMMAND ----------

def download_wb_geojson(url: str, cache_path: str) -> gpd.GeoDataFrame:
    """Download World Bank GeoJSON to cache and return as GeoDataFrame."""
    if file_exists(cache_path):
        print(f"Loading cached WB boundaries: {cache_path}")
        return gpd.read_file(cache_path)

    print(f"Downloading World Bank boundaries: {url}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(cache_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Cached to: {cache_path}")
    return gpd.read_file(cache_path)

# COMMAND ----------

# EXECUTE TASK

ensure_dir(VOLUME_DIR)

print("Downloading World Bank Official Boundaries...")
print("=" * 60)

# Download Admin0 (country boundaries)
admin0_path = os.path.join(VOLUME_DIR, "wb_admin0.geojson")
gdf_admin0 = download_wb_geojson(WB_ADMIN0_URL, admin0_path)
print(f"  Admin0: {len(gdf_admin0)} countries")

# Download Admin1 (province/state boundaries)
admin1_path = os.path.join(VOLUME_DIR, "wb_admin1.geojson")
gdf_admin1 = download_wb_geojson(WB_ADMIN1_URL, admin1_path)
print(f"  Admin1: {len(gdf_admin1)} regions")

# Download Admin2 (district boundaries - used for LGU metrics)
admin2_path = os.path.join(VOLUME_DIR, "wb_admin2.geojson")
gdf_admin2 = download_wb_geojson(WB_ADMIN2_URL, admin2_path)
print(f"  Admin2: {len(gdf_admin2)} districts")

print("=" * 60)
print("Task complete. WB boundaries cached to Volume.")
print(f"  {admin0_path}")
print(f"  {admin1_path}")
print(f"  {admin2_path}")
