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
import re
import unicodedata
if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from shared.core import (
        get_k_rings,
        get_transform_table_names as _get_transform_table_names, _sanitize_adm_name,
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

# List of admin level 1 regions to process:
#  - []: all provinces (auto-discovered from UC)
 
# Malawi - ["Central Region","Northern Region","Southern Region"]

# India - ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']

# Cambodia = ['Banteay Meanchey', 'Battambang', 'Kampong Cham', 'Kampong Speu', 'Kampong Thom', 'Kampot', 'Kandal', 'Kep', 'Koh Kong', 'Kratie', 'Mondul Kiri', 'Oddar Meanchey', 'Pailin', 'Phnom Penh', 'Preah Sihanouk', 'Preah Vihear', 'Prey Veng', 'Pursat', 'Ratanak Kiri', 'Siemreap', 'Svay Rieng', 'Takeo', 'Tboung Khmum']

# Chad = ['Barh el Ghazel', 'Batha', 'Borkou', 'Chari-Baguirmi', 'Hadjer-Lamis', 'Kanem', 'Lac', 'Logone Occidental', 'Logone Oriental', 'Mandoul', 'Mayo-Kebbi Est', 'Mayo-Kebbi Ouest', 'Moyen-Chari', 'Ouaddaï', 'Salamat', 'Sila', Tandjilé', "Ville de N'Djamena", 'Wadi Fira']

# Gabon = ['Estuaire', 'Haut-Ogooue', 'Moyen-Ogooue', 'Ngounie', 'Ogooue-Maritime', 'Ogooue-lolo']

# The Gambia = ['Central River North', 'Central River South', 'Kanifing Municipal Council', 'Lower River', 'North Bank',  'Upper River', 'West Coast']

# Afghanistan = ['Agadez', 'Communauté Urbaine de Niamey', 'Diffa', 'Dosso', 'Maradi', 'Tahoua', 'Tillabéri', 'Zinder']

# Cameroon= ['Adamaoua', 'Centre', 'Est', 'Extrême - Nord', 'Littoral', 'Nord','Nord - Ouest', 'Ouest', 'Sud', 'Sud - Ouest']

# Mali = ['District de Bamako', 'Gao', 'Kayes', 'Kidal', 'Koulikoro', 'Mopti', 'Sikasso', 'Ségou', 'Tombouctou']

# Niger = ['Agadez', 'Communauté Urbaine de Niamey', 'Diffa', 'Dosso', 'Maradi', 'Tahoua', 'Tillabéri', 'Zinder']

# Somalia = ['Awdal', 'Banadir', 'Bari', 'Bay', 'Galgaduud', 'Hiraan', 'Juba Hoose', 'Shabelle Dhexe', 'Shabelle Hoose', 'Sool', 'Togdheer', 'Woqooyi Galbeed']

# Sudan = ['Al Jazeera', 'Blue Nile', 'Gadaref', 'Kassala', 'Khartoum', 'Nile', 'Northern', 'Northern Darfur', 'Northern Kordofan', 'Red Sea', 'Southern Darfur', 'Southern Kordofan', 'Western Darfur', 'White Nile']

# Ethiopia = ['Addis Ababa', 'Afar', 'Amhara', 'Dire Dawa', 'Gambela', 'Harari', 'Oromia', 'SNNP', 'Sidama', 'Somali', 'South West Ethiopia', 'Tigray']

# Romania = ['Alba', 'Arad', 'Argeş', 'Bacău', 'Bihor', 'Bistriţa-Năsaud', 'Botoşani', 'Braşov', 'Brăila', 'Bucureşti', 'Buzău', 'Caraş-Severin', 'Cluj', 'Constanţa', 'Covasna', 'Călăraşi', 'Dolj', 'Dâmboviţa', 'Galaţi', 'Giurgiu', 'Gori', 'Harghita', 'Hunedoara',  'Iaşi', 'Ilfov', 'Maramureş', 'Mehedinţi', 'Mureş', 'Neamţ', 'Olt', 'Prahova', 'Satu Mare', 'Sibiu', 'Suceava', 'Sălaj', 'Teleorman', 'Timiş', 'Tulcea', 'Vaslui', 'Vrancea', 'Vâlcea']

# Syria =  ['Al Ḥasakah', 'Aleppo', 'Ar Raqqah', "As Suwaydā'", 'Damascus', 'Dar`ā', 'Dayr az Zawr', 'Hama', 'Idlib', 'Latakia', 'Quneitra', 'Rif Dimashq', 'Ţarţūs', 'Ḥimṣ']

# West Bank and Gaza = ['Al Khalil (Hebron)', 'Al Quds (Jerusalem)', 'Bethlehem', 'Deir al Balah', 'Gaza', 'Jabalya', 'Jenin', 'Khan Yunis', 'Nablus', 'Qalqiliya', 'Ramallah', 'Salfit', 'Tubas', 'Tulkarm']

# Equatorial Guinea = ['Bioko Norte', 'Litoral']

# COMMAND ----------

# CONFIGURATION

# Include country-level (ADM0) processing
INCLUDE_ADM_LEVEL0 = True

# 
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
FORCE_RECOMPUTE = False

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
    lgu_table = f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_lgu_{_sanitize_adm_name(COUNTRY)}"
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
