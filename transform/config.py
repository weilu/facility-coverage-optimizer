# Databricks notebook source
# Transform pipeline configuration
# This file is imported by all transform tasks

# COMMAND ----------

import numpy as np

# CONFIGURATION
COUNTRY = "Zambia"
COUNTRY_ISO3 = "ZMB"
POPULATION_YEAR = 2025

UC_CATALOG = "prd_mega"
UC_SCHEMA = "sgpbpi163"

# List of admin level 1 regions to process (set to None to process entire country)
# If empty list [], will process all provinces (requires extraction to have run first)
ADM_LEVEL1_LIST = ["Northern"]  # e.g., ["Northern", "North-Western"] or [] for all

# List of distances to analyze (in meters)
DISTANCES_METERS = [5000, 10000]  # e.g., [5000, 10000] for 5km and 10km

TRAVEL_API = ""  # "" for buffer, "osm", or "mapbox"
MAPBOX_ACCESS_TOKEN = ""
MAPBOX_MODE = "driving"

POTENTIAL_TYPE = "grid"  # "grid" or "kmeans"
GRID_SPACING = 0.03
N_CLUSTERS = 100

TARGET_NEW_FACILITIES = 50
H3_RESOLUTION = 8  # Must match extraction resolution

# Set to True to recompute cached results
FORCE_RECOMPUTE = False

# Target access rate for LGU equity analysis
TARGET_ACCESS_RATE_PCT = 90.0

# H3 resolution 8 edge length is ~461m
# k_rings = ceil(distance_meters / edge_length)
H3_EDGE_LENGTH_M = {4: 22606, 5: 8544, 6: 3229, 7: 1220, 8: 461, 9: 174, 10: 66}

# COMMAND ----------

# HELPER FUNCTIONS

def get_k_rings(distance_meters: int, h3_resolution: int) -> int:
    """Calculate number of H3 rings for given distance."""
    return int(np.ceil(distance_meters / H3_EDGE_LENGTH_M[h3_resolution]))


def get_transform_table_names(
    country: str,
    iso3: str,
    adm_level1: str | None,
    population_year: int,
    distance_meters: int,
):
    """Generate table names for transform step based on configuration."""
    distance_name = f"{int(distance_meters / 1000)}km"

    if adm_level1 is not None:
        adm_suffix = f"_{adm_level1.lower().replace('-', '_')}_province"
        return {
            "boundaries": f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_{iso3.lower()}{adm_suffix}",
            "facilities": f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{iso3.lower()}_osm{adm_suffix}",
            "population": f"{UC_CATALOG}.{UC_SCHEMA}.population_{iso3.lower()}_{population_year}",
            "population_aoi": f"{UC_CATALOG}.{UC_SCHEMA}.population_aoi_{iso3.lower()}_{population_year}{adm_suffix}_{distance_name}",
            "facilities_h3": f"{UC_CATALOG}.{UC_SCHEMA}.facilities_h3_{iso3.lower()}{adm_suffix}_{distance_name}",
            "facilities_coverage": f"{UC_CATALOG}.{UC_SCHEMA}.facilities_coverage_{iso3.lower()}{adm_suffix}_{distance_name}",
            "potential_locations": f"{UC_CATALOG}.{UC_SCHEMA}.potential_locations_{iso3.lower()}{adm_suffix}_{distance_name}",
            "potential_coverage": f"{UC_CATALOG}.{UC_SCHEMA}.potential_coverage_{iso3.lower()}{adm_suffix}_{distance_name}",
            "lgu": f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_lgu_{country.lower()}",
            "lgu_accessibility": f"{UC_CATALOG}.{UC_SCHEMA}.lgu_accessibility_results_{iso3.lower()}{adm_suffix}_{distance_name}",
        }
    else:
        return {
            "boundaries": f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_{iso3.lower()}",
            "facilities": f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{iso3.lower()}",
            "population": f"{UC_CATALOG}.{UC_SCHEMA}.population_{iso3.lower()}_{population_year}",
            "population_aoi": f"{UC_CATALOG}.{UC_SCHEMA}.population_aoi_{iso3.lower()}_{population_year}_{distance_name}",
            "facilities_h3": f"{UC_CATALOG}.{UC_SCHEMA}.facilities_h3_{iso3.lower()}_{distance_name}",
            "facilities_coverage": f"{UC_CATALOG}.{UC_SCHEMA}.facilities_coverage_{iso3.lower()}_{distance_name}",
            "potential_locations": f"{UC_CATALOG}.{UC_SCHEMA}.potential_locations_{iso3.lower()}_{distance_name}",
            "potential_coverage": f"{UC_CATALOG}.{UC_SCHEMA}.potential_coverage_{iso3.lower()}_{distance_name}",
            "lgu": f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_lgu_{country.lower()}",
            "lgu_accessibility": f"{UC_CATALOG}.{UC_SCHEMA}.lgu_accessibility_results_{iso3.lower()}_{distance_name}",
        }


def build_transform_combinations():
    """Build list of (province, distance) combinations to process."""
    combinations = []
    for adm_level1 in ADM_LEVEL1_LIST if ADM_LEVEL1_LIST else [None]:
        for distance_meters in DISTANCES_METERS:
            combinations.append((adm_level1, distance_meters))
    return combinations
