# Databricks notebook source
# MAGIC %pip install shapely rasterio pycountry

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# TASK: Download WorldPop population raster
#
# Dependencies: None
# Outputs: WorldPop GeoTIFF raster file in Volume
#
# This task downloads the WorldPop constrained population raster for the
# configured country and year. The raster is saved to a Databricks Volume
# for subsequent processing by 02_population.py.

# COMMAND ----------

import os
import urllib.request

# COMMAND ----------

# Import shared utilities and configuration
from shared.utils import ensure_dir, file_exists
from extract.config import (
    COUNTRY,
    POPULATION_YEAR,
    ISO_3,
    VOLUME_DIR,
    FORCE_RECOMPUTE,
    get_raster_path,
)

# COMMAND ----------

def extract_worldpop_raster(
    country_iso3: str,
    population_year: int,
    output_path: str,
    force: bool = False,
) -> str:
    """
    Downloads WorldPop GeoTIFF raster to Volume.
    Returns the file path.
    """
    if not force and file_exists(output_path):
        print(f"Raster already exists, skipping download: {output_path}")
        return output_path

    url = (
        f"https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/"
        f"{population_year}/{country_iso3}/v1/100m/constrained/"
        f"{country_iso3.lower()}_pop_{population_year}_CN_100m_R2024B_v1.tif"
    )
    print(f"Downloading: {url}")

    urllib.request.urlretrieve(url, output_path)
    print(f"WorldPop raster saved: {output_path}")
    return output_path

# COMMAND ----------

# EXECUTE TASK

print(f"Country: {COUNTRY} | ISO-3: {ISO_3}")
print(f"Population year: {POPULATION_YEAR}")

ensure_dir(VOLUME_DIR)
raster_path = get_raster_path()

extract_worldpop_raster(
    country_iso3=ISO_3,
    population_year=POPULATION_YEAR,
    output_path=raster_path,
    force=FORCE_RECOMPUTE,
)

print(f"\nTask complete. Raster saved to: {raster_path}")
