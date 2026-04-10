# Databricks notebook source
# Extract pipeline configuration
# This file is imported by all extract tasks

# COMMAND ----------

# Import shared utilities
from shared.core import get_extract_table_names
from shared.settings import (
    UC_CATALOG,
    UC_SCHEMA,
    COUNTRY,
    ISO_3,
    POPULATION_YEAR,
)

# COMMAND ----------

# CONFIGURATION

VOLUME_DIR = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/vgpbpi163"

# Set to True to recompute cached results even if tables exist
FORCE_RECOMPUTE = False

# List of admin level 1 regions to process (set to None to process entire country)
# If empty list [], will auto-discover all provinces from World Bank boundaries
ADM_LEVEL1_LIST = []  # e.g., ["Northern", "North-Western"] or [] for all

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
