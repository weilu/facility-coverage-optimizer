# Databricks notebook source
# MAGIC %pip install geopandas shapely pycountry requests

# COMMAND ----------

# MAGIC %pip install -U geopandas shapely

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# TASK: Extract health facilities
#
# Dependencies:
#   - 01b_download_wb.py (WB boundaries cached in Volume)
#   - 03_boundaries.py (Province boundaries extracted)
#
# Outputs: Health facilities UC table per province
#
# This task extracts health facilities for each configured province.
# It supports two modes:
#   - "osm": Query OpenStreetMap Overpass API for hospitals and clinics
#   - "file": Load from pre-curated GeoJSON file

# COMMAND ----------

import requests
import pandas as pd
import geopandas as gpd

# COMMAND ----------

# Import shared utilities and configuration
import os
from shared.utils import (
    get_spark,
    gdf_to_uc_table,
    uc_table_to_gdf,
    pdf_to_uc_table,
    table_exists,
    file_exists,
)
from extract.config import (
    COUNTRY,
    ISO_2,
    ISO_3,
    VOLUME_DIR,
    FORCE_RECOMPUTE,
    ADM_LEVEL1_LIST,
    FACILITIES_SOURCE,
    FACILITIES_INPUT_PATH,
    get_table_names,
    POPULATION_YEAR,
)

spark = get_spark()

# COMMAND ----------

# Boundary helper functions (duplicated from 03_boundaries.py for task independence)

def load_cached_wb_boundaries(admin_level: int) -> gpd.GeoDataFrame:
    """Load cached World Bank boundaries GeoJSON."""
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

    return gpd.read_file(cache_path)


def get_all_adm_level1_names(country_iso3: str) -> list[str]:
    """Get all admin level 1 (province/state) names for a country."""
    gdf = load_cached_wb_boundaries(admin_level=1)
    gdf_country = gdf[gdf["ISO_A3"] == country_iso3]
    provinces = sorted(gdf_country["NAM_1"].unique().tolist())
    print(f"Found {len(provinces)} admin level 1 regions (WB): {provinces}")
    return provinces

# COMMAND ----------

def extract_health_facilities_osm(
    iso_2: str,
    table_name: str,
    selected_boundary: gpd.GeoDataFrame,
    adm_level_name: str = "AOI",
    force: bool = False,
) -> pd.DataFrame:
    """
    Queries OSM Overpass API for hospitals and clinics.
    Saves to UC table and returns DataFrame.
    """
    if not force and table_exists(table_name):
        print(f"OSM facilities already exist, loading: {table_name}")
        return spark.table(table_name).toPandas()

    def query_osm_amenity(amenity: str) -> pd.DataFrame:
        query = f"""
        [out:json];
        area["ISO3166-1"="{iso_2}"];
        (
          node["amenity"="{amenity}"](area);
          way["amenity"="{amenity}"](area);
          rel["amenity"="{amenity}"](area);
        );
        out center;
        """
        response = requests.get(
            "http://overpass-api.de/api/interpreter",
            params={"data": query},
            timeout=120,
        )
        response.raise_for_status()
        elements = response.json()["elements"]
        df = pd.DataFrame(elements)
        if df.empty:
            return pd.DataFrame(columns=["osm_id", "lat", "lon", "name"])
        df["name"] = df["tags"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
        df = df.rename(columns={"id": "osm_id"})
        return df[["osm_id", "lat", "lon", "name"]].dropna(subset=["lat", "lon"])

    print(f"Querying OSM for hospitals in {iso_2}...")
    df_hospitals = query_osm_amenity("hospital")
    print(f"  Hospitals: {len(df_hospitals)}")

    print(f"Querying OSM for clinics in {iso_2}...")
    df_clinics = query_osm_amenity("clinic")
    print(f"  Clinics: {len(df_clinics)}")

    df_health = (
        pd.concat([df_hospitals, df_clinics])
        .drop_duplicates(subset="osm_id")
        .reset_index(drop=True)
    )
    gdf_health = gpd.GeoDataFrame(
        df_health,
        geometry=gpd.points_from_xy(df_health.lon, df_health.lat),
        crs="EPSG:4326"
    )

    # Reproject and spatial join
    gdf_health = gdf_health.to_crs(selected_boundary.crs)
    selected_health = gpd.sjoin(gdf_health, selected_boundary, predicate='within')
    selected_health = selected_health.reset_index().reset_index()
    selected_health['ID'] = selected_health['level_0'].astype(str)+'_current'

    print(f"Number of hospitals and clinics extracted: {len(gdf_health)}")
    print(f"Number of facilities in AOI ({adm_level_name}): {len(selected_health)}")
    selected_health = selected_health.loc[:, ~selected_health.columns.duplicated()]

    # Convert geometry to WKT string before writing to Spark
    selected_health_pdf = selected_health.copy()
    selected_health_pdf["geometry_wkt"] = selected_health_pdf["geometry"].apply(
        lambda geom: geom.wkt if geom is not None else None
    )
    selected_health_pdf = selected_health_pdf.drop(columns=["geometry"])
    print(selected_health_pdf.columns)
    selected_health_pdf = selected_health_pdf.rename(columns={
        "id": "osm_id",
    })

    pdf_to_uc_table(selected_health_pdf, table_name)
    return selected_health_pdf


def extract_existing_facilities(
    input_path: str,
    table_name: str,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """
    Loads existing health facilities GeoJSON and saves to UC table.
    Use this if you have curated facility data instead of OSM.
    """
    if not force and table_exists(table_name):
        print(f"Facilities already exist, loading: {table_name}")
        return uc_table_to_gdf(table_name)

    gdf = gpd.read_file(input_path)
    gdf_to_uc_table(gdf, table_name)
    return gdf

# COMMAND ----------

# EXECUTE TASK: Determine provinces to process

print(f"Country: {COUNTRY} | ISO-2: {ISO_2} | ISO-3: {ISO_3}")
print(f"Facilities source: {FACILITIES_SOURCE}")

if ADM_LEVEL1_LIST == []:
    provinces_to_process = get_all_adm_level1_names(ISO_3)
elif ADM_LEVEL1_LIST is None:
    provinces_to_process = [None]  # Process entire country
else:
    provinces_to_process = ADM_LEVEL1_LIST

print(f"Will process {len(provinces_to_process)} region(s): {provinces_to_process}")

# COMMAND ----------

# EXECUTE TASK: Extract facilities per province

extraction_results = []

for adm_level1 in provinces_to_process:
    print("\n" + "=" * 60)
    print(f"PROCESSING: {adm_level1 if adm_level1 else 'ENTIRE COUNTRY'}")
    print("=" * 60)

    tables = get_table_names(COUNTRY, ISO_3, adm_level1, POPULATION_YEAR)
    boundaries_table = tables["boundaries"]
    facilities_table = tables["facilities"]

    print(f"  Boundaries Table: {boundaries_table}")
    print(f"  Facilities Table: {facilities_table}")

    # Load province boundary (must exist from 03_boundaries.py)
    if not table_exists(boundaries_table):
        raise RuntimeError(
            f"Boundaries table not found: {boundaries_table}. "
            "Run 03_boundaries.py first."
        )
    selected_boundary_gdf = uc_table_to_gdf(boundaries_table)

    # Extract facilities
    if FACILITIES_SOURCE == "osm":
        extract_health_facilities_osm(
            iso_2=ISO_2,
            table_name=facilities_table,
            selected_boundary=selected_boundary_gdf,
            adm_level_name=adm_level1 if adm_level1 else "Country",
            force=FORCE_RECOMPUTE,
        )
    elif FACILITIES_SOURCE == "file":
        if FACILITIES_INPUT_PATH is None:
            raise ValueError("FACILITIES_SOURCE='file' but FACILITIES_INPUT_PATH is not set")
        extract_existing_facilities(
            input_path=FACILITIES_INPUT_PATH,
            table_name=facilities_table,
            force=FORCE_RECOMPUTE,
        )
    else:
        raise ValueError(f"Invalid FACILITIES_SOURCE: {FACILITIES_SOURCE}. Use 'osm' or 'file'.")

    extraction_results.append({
        "adm_level1": adm_level1,
        "facilities_table": facilities_table,
    })

    print(f"  Completed: {adm_level1 if adm_level1 else 'ENTIRE COUNTRY'}")

# COMMAND ----------

# TASK SUMMARY

print("\n" + "=" * 60)
print("FACILITIES EXTRACTION COMPLETE")
print("=" * 60)
print(f"Country: {COUNTRY} ({ISO_3})")
print(f"Source: {FACILITIES_SOURCE}")
print(f"Provinces processed: {len(extraction_results)}")
for result in extraction_results:
    region = result["adm_level1"] if result["adm_level1"] else "Country"
    print(f"  - {region}: {result['facilities_table']}")
print("=" * 60)
