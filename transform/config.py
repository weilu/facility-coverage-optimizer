# Databricks notebook source
# Transform pipeline configuration
# This file is imported by all transform tasks

# COMMAND ----------

# MAGIC %run "../shared/core"

# COMMAND ----------

# MAGIC %run "../shared/settings"

# COMMAND ----------

# MAGIC %run "../shared/env"

# COMMAND ----------

# Local imports (skipped in Databricks where %run loads modules)
import os
if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from shared.core import (
        get_k_rings,
        get_transform_table_names as _get_transform_table_names,
        build_transform_combinations as _build_transform_combinations,
        H3_EDGE_LENGTH_M,
    )
    from shared.settings import (
        UC_CATALOG,
        UC_SCHEMA,
        COUNTRY,
        ISO_3 as COUNTRY_ISO3,
        POPULATION_YEAR,
    )
    from shared.env import get_spark
else:
    # Databricks: alias the functions loaded via %run
    _get_transform_table_names = get_transform_table_names
    _build_transform_combinations = build_transform_combinations
    COUNTRY_ISO3 = ISO_3

# COMMAND ----------

# CONFIGURATION

# Include country-level (ADM0) processing
INCLUDE_ADM_LEVEL0 = True

# List of admin level 1 regions to process:
#   - []: all provinces (auto-discovered from UC)
#   - ["Northern", "Lusaka"]: specific provinces only
ADM_LEVEL1_LIST = []

# List of distances to analyze (in meters)
DISTANCES_METERS = [2000, 4000, 5000, 10000]  # e.g., [5000, 10000] for 5km and 10km

TRAVEL_API = ""  # "" for buffer, "osm", or "mapbox"
MAPBOX_ACCESS_TOKEN = ""
MAPBOX_MODE = "driving"

POTENTIAL_TYPE = "grid"  # "grid" or "kmeans"
GRID_SPACING = 0.03
N_CLUSTERS = 100

TARGET_NEW_FACILITIES = 50
H3_RESOLUTION = 8  # Must match extraction resolution

# Set to True to recompute cached results
FORCE_RECOMPUTE = True

# Target access rate for LGU equity analysis
TARGET_ACCESS_RATE_PCT = 90.0

# Base dashboard data table (aggregated metadata for frontend)
BASE_DASHBOARD_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.base_dashboard_data_{COUNTRY_ISO3.lower()}"

# Visualization settings
ENABLE_VISUALIZATION_DEFAULT = True
VIZ_SAMPLE_SIZE = 5_000  # Max points per category for Folium maps


def _get_enable_visualization() -> bool:
    """Get ENABLE_VISUALIZATION from dbutils widget or use default."""
    try:
        val = dbutils.widgets.get("ENABLE_VISUALIZATION")
        return val.lower() in ("true", "1", "yes")
    except:
        return ENABLE_VISUALIZATION_DEFAULT


ENABLE_VISUALIZATION = _get_enable_visualization()

# COMMAND ----------

# HELPER FUNCTIONS (partial applications of shared.core functions)


def get_transform_table_names(
    country: str,
    iso3: str,
    adm_level1: str | None,
    population_year: int,
    distance_meters: int,
):
    """Generate table names for transform step based on configuration."""
    return _get_transform_table_names(
        UC_CATALOG, UC_SCHEMA, country, iso3, adm_level1, population_year, distance_meters
    )


def _get_adm_level1_names_from_uc() -> list[str]:
    """Discover province names from LGU boundary table in UC."""
    spark = get_spark()
    lgu_table = f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_lgu_{COUNTRY.lower()}"
    provinces_df = spark.sql(f"SELECT DISTINCT province FROM {lgu_table} ORDER BY province")
    provinces = [row.province for row in provinces_df.collect()]
    print(f"Discovered {len(provinces)} provinces from UC: {provinces}")
    return provinces


def build_transform_combinations():
    """Build list of (province, distance) combinations to process."""
    adm_list = []

    # ADM0 (country-level)
    if INCLUDE_ADM_LEVEL0:
        adm_list.append(None)

    # ADM1 (provinces)
    if ADM_LEVEL1_LIST == []:
        adm_list.extend(_get_adm_level1_names_from_uc())
    else:
        adm_list.extend(ADM_LEVEL1_LIST)

    return _build_transform_combinations(adm_list, DISTANCES_METERS)
