# Databricks notebook source
# Extract pipeline configuration
# This file is imported by all extract tasks

# COMMAND ----------

# MAGIC %run "../shared/core"

# COMMAND ----------

# MAGIC %run "../shared/env"

# COMMAND ----------

# MAGIC %run "../shared/settings"

# COMMAND ----------

# Local imports (skipped in Databricks where %run loads modules)
import os
import geopandas as gpd
if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from shared.core import get_extract_table_names
    from shared.env import file_exists
    from shared.settings import (
        UC_CATALOG,
        UC_SCHEMA,
        COUNTRY,
        ISO_3,
        POPULATION_YEAR,
    )

# COMMAND ----------

# CONFIGURATION

VOLUME_DIR = f"/Volumes/{UC_CATALOG}/sgpbpi163/vgpbpi163"

# Set to True to recompute cached results even if tables exist
FORCE_RECOMPUTE = True

# Include country-level (ADM0) processing
INCLUDE_ADM_LEVEL0 = True

# List of admin level 1 regions to process:
#   - []: all provinces (auto-discovered from WB boundaries)
#   - ["Northern", "Lusaka"]: specific provinces only
ADM_LEVEL1_LIST = []

# Health facilities data source: "osm" or "file"
# - "osm": Query OpenStreetMap Overpass API for hospitals and clinics
# - "file": Use existing curated GeoJSON file (set FACILITIES_INPUT_PATH below)
FACILITIES_SOURCE = "osm"
FACILITIES_INPUT_PATH = f"{VOLUME_DIR}/selected_hosp_input_data.geojson"

# World Bank Official Boundaries GeoJSON URLs (version 5, June 2025)
WB_BOUNDARIES_BASE_URL = "https://datacatalogfiles.worldbank.org/ddh-published-v2/0038272/5/DR0095369/World%20Bank%20Official%20Boundaries%20(GeoJSON)"
WB_ADMIN0_URL = f"{WB_BOUNDARIES_BASE_URL}/World%20Bank%20Official%20Boundaries%20-%20Admin%200.geojson"
WB_ADMIN1_URL = f"{WB_BOUNDARIES_BASE_URL}/World%20Bank%20Official%20Boundaries%20-%20Admin%201.geojson"
WB_ADMIN2_URL = f"{WB_BOUNDARIES_BASE_URL}/World%20Bank%20Official%20Boundaries%20-%20Admin%202.geojson"

# Corrections for known typos in WB boundaries data
# See: https://github.com/worldbank/WB_GAD/issues/25
WB_NAME_CORRECTIONS = {
    "Muchiga": "Muchinga",  # Zambia province typo
}

# COMMAND ----------

# DERIVED CONFIGURATION

COUNTRY_POPULATION_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.population_{ISO_3.lower()}_{POPULATION_YEAR}"
COUNTRY_LGU_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_lgu_{COUNTRY.lower()}"
RASTER_PATH = f"{VOLUME_DIR}/worldpop_{ISO_3.lower()}_{POPULATION_YEAR}.tif"

# COMMAND ----------

# TABLE NAME GENERATOR (partial application of shared.core function)


def get_table_names(country: str, iso3: str, adm_level1: str | None, population_year: int):
    """Generate table names based on configuration."""
    return get_extract_table_names(
        UC_CATALOG, UC_SCHEMA, country, iso3, adm_level1, population_year
    )


def _apply_wb_name_corrections(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply known name corrections to WB boundaries GeoDataFrame."""
    for col in ["NAM_1", "NAM_2"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].replace(WB_NAME_CORRECTIONS)
    return gdf


def load_cached_wb_boundaries(admin_level: int) -> gpd.GeoDataFrame:
    """Load cached World Bank boundaries GeoJSON with name corrections applied."""
    if admin_level == 0:
        cache_path = os.path.join(VOLUME_DIR, "wb_admin0.geojson")
    elif admin_level == 1:
        cache_path = os.path.join(VOLUME_DIR, "wb_admin1.geojson")
    elif admin_level == 2:
        cache_path = os.path.join(VOLUME_DIR, "wb_admin2.geojson")
    else:
        raise ValueError(f"Invalid admin_level: {admin_level}")

    if not file_exists(cache_path):
        raise FileNotFoundError(
            f"WB boundaries not cached: {cache_path}. "
            "Run 01b_download_wb.py first."
        )

    gdf = gpd.read_file(cache_path)
    return _apply_wb_name_corrections(gdf)


def get_all_adm_level1_names(country_iso3: str) -> list[str]:
    """Get all admin level 1 (province/state) names for a country."""
    gdf = load_cached_wb_boundaries(admin_level=1)
    gdf_country = gdf[gdf["ISO_A3"] == country_iso3]
    provinces = sorted(gdf_country["NAM_1"].unique().tolist())
    print(f"Found {len(provinces)} admin level 1 regions (WB): {provinces}")
    return provinces
