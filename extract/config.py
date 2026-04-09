# Databricks notebook source
# Extract pipeline configuration
# This file is imported by all extract tasks

# COMMAND ----------

# CONFIGURATION

COUNTRY = "Zambia"
POPULATION_YEAR = 2025

UC_CATALOG = "prd_mega"
UC_SCHEMA = "sgpbpi163"
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

# DERIVED CONFIGURATION (computed at import time)

import pycountry

def _get_country_codes(country_name: str):
    """Look up ISO codes for a country name."""
    try:
        country = pycountry.countries.lookup(country_name)
        return {
            "name": country.name,
            "alpha_2": country.alpha_2,
            "alpha_3": country.alpha_3,
            "numeric": country.numeric,
        }
    except LookupError:
        return None

_iso_codes = _get_country_codes(COUNTRY)
ISO_2 = _iso_codes["alpha_2"]
ISO_3 = _iso_codes["alpha_3"]

# COMMAND ----------

# TABLE NAME GENERATORS

def get_table_names(country: str, iso3: str, adm_level1: str | None, population_year: int):
    """Generate table names based on configuration."""
    if adm_level1 is not None:
        adm_suffix = f"_{adm_level1.lower().replace('-', '_')}_province"
        return {
            "boundaries": f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_{iso3.lower()}{adm_suffix}",
            "population": f"{UC_CATALOG}.{UC_SCHEMA}.population_{iso3.lower()}_{population_year}{adm_suffix}",
            "facilities": f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{iso3.lower()}_osm{adm_suffix}",
            "lgu": f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_lgu_{country.lower()}{adm_suffix}",
        }
    else:
        return {
            "boundaries": f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_{iso3.lower()}",
            "population": f"{UC_CATALOG}.{UC_SCHEMA}.population_{iso3.lower()}_{population_year}",
            "facilities": f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{iso3.lower()}_osm",
            "lgu": f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_lgu_{country.lower()}",
        }


def get_country_population_table():
    """Get the country-level population table name."""
    return f"{UC_CATALOG}.{UC_SCHEMA}.population_{ISO_3.lower()}_{POPULATION_YEAR}"


def get_country_lgu_table():
    """Get the country-level LGU boundaries table name."""
    return f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_lgu_{COUNTRY.lower()}"


def get_raster_path():
    """Get the WorldPop raster file path."""
    return f"{VOLUME_DIR}/worldpop_{ISO_3.lower()}_{POPULATION_YEAR}.tif"
