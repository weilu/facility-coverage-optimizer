# Databricks notebook source
# MAGIC %pip install shapely rasterio pycountry gurobipy folium plotly scikit-learn pyproj gadm

# COMMAND ----------

# MAGIC %pip install -U geopandas shapely

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import urllib.request
import requests
import pandas as pd
import geopandas as gpd
import pycountry
from gadm import GADMDownloader

# COMMAND ----------

# CONFIGURATION

COUNTRY = "Zambia"
ADM_LEVEL1 = None
ADM_LEVEL2 = None
POPULATION_YEAR = 2025

UC_CATALOG = "prd_mega"
UC_SCHEMA = "sgpbpi163"
VOLUME_DIR = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/vgpbpi163"

# COMMAND ----------

def get_country_codes(country_name: str):
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


def ensure_dir(path: str):
    """Create directory if it doesn't exist using dbutils."""
    try:
        dbutils.fs.ls(path)
        print(f"Directory exists: {path}")
    except Exception:
        dbutils.fs.mkdirs(path)
        print(f"Directory created: {path}")


def file_exists(path: str) -> bool:
    """Check if file exists using dbutils."""
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


def table_exists(table_name: str) -> bool:
    """Check if UC table exists."""
    try:
        spark.sql(f"DESCRIBE TABLE {table_name}")
        return True
    except Exception:
        return False


def gdf_to_uc_table(gdf: gpd.GeoDataFrame, table_name: str, mode: str = "overwrite"):
    """Save GeoDataFrame to Unity Catalog table with geometry as WKT."""
    pdf = gdf.copy()
    pdf["geometry_wkt"] = pdf.geometry.apply(lambda g: g.wkt if g else None)
    pdf = pdf.drop(columns=["geometry"])

    sdf = spark.createDataFrame(pdf)
    sdf.write.mode(mode).saveAsTable(table_name)
    print(f"Table saved: {table_name} ({len(gdf)} rows)")


def uc_table_to_gdf(table_name: str) -> gpd.GeoDataFrame:
    """Load Unity Catalog table as GeoDataFrame."""
    from shapely.wkt import loads as wkt_loads

    pdf = spark.table(table_name).toPandas()
    pdf["geometry"] = pdf["geometry_wkt"].apply(lambda w: wkt_loads(w) if w else None)
    pdf = pdf.drop(columns=["geometry_wkt"])
    return gpd.GeoDataFrame(pdf, geometry="geometry", crs="EPSG:4326")


def pdf_to_uc_table(pdf: pd.DataFrame, table_name: str, mode: str = "overwrite"):
    """Save pandas DataFrame to Unity Catalog table."""
    sdf = spark.createDataFrame(pdf)
    sdf.write.mode(mode).saveAsTable(table_name)
    print(f"Table saved: {table_name} ({len(pdf)} rows)")

# COMMAND ----------

# EXTRACT: GADM ADMINISTRATIVE BOUNDARIES

def extract_gadm_boundaries(
    country: str,
    adm_level1: str | None,
    adm_level2: str | None,
    table_name: str,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """
    Downloads GADM boundaries and saves to UC table.
    Returns the selected boundary GeoDataFrame.
    """
    if not force and table_exists(table_name):
        print(f"GADM boundaries already exist, loading: {table_name}")
        return uc_table_to_gdf(table_name)

    downloader = GADMDownloader(version="4.0")

    if adm_level1 is not None:
        df_shp = downloader.get_shape_data_by_country_name(country_name=country, ad_level=1)
        selected_gadm = df_shp[df_shp["NAME_1"] == adm_level1]
    elif adm_level2 is not None:
        df_shp = downloader.get_shape_data_by_country_name(country_name=country, ad_level=2)
        selected_gadm = df_shp[df_shp["NAME_2"] == adm_level2]
    else:
        df_shp = downloader.get_shape_data_by_country_name(country_name=country, ad_level=0)
        selected_gadm = df_shp

    gdf_to_uc_table(selected_gadm, table_name)
    return selected_gadm

# COMMAND ----------

# EXTRACT: WORLDPOP POPULATION RASTER

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

# EXTRACT: HEALTH FACILITIES FROM OSM

def extract_health_facilities_osm(iso_2: str, table_name: str, force: bool = False) -> pd.DataFrame:
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
            return pd.DataFrame(columns=["id", "lat", "lon", "name"])
        df["name"] = df["tags"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
        return df[["id", "lat", "lon", "name"]].dropna(subset=["lat", "lon"])

    print(f"Querying OSM for hospitals in {iso_2}...")
    df_hospitals = query_osm_amenity("hospital")
    print(f"  Hospitals: {len(df_hospitals)}")

    print(f"Querying OSM for clinics in {iso_2}...")
    df_clinics = query_osm_amenity("clinic")
    print(f"  Clinics: {len(df_clinics)}")

    df_health = (
        pd.concat([df_hospitals, df_clinics])
        .drop_duplicates(subset="id")
        .reset_index(drop=True)
    )

    pdf_to_uc_table(df_health, table_name)
    return df_health

# COMMAND ----------

# EXTRACT: COPY EXISTING HEALTH FACILITIES FILE (if using pre-curated data)

def extract_existing_facilities(input_path: str, table_name: str, force: bool = False) -> gpd.GeoDataFrame:
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

# RUN EXTRACTION PIPELINE

iso_codes = get_country_codes(COUNTRY)
ISO_2 = iso_codes["alpha_2"]
ISO_3 = iso_codes["alpha_3"]
print(f"Country: {COUNTRY} | ISO-2: {ISO_2} | ISO-3: {ISO_3}")

# COMMAND ----------

# Extract GADM boundaries
gadm_table = f"{UC_CATALOG}.{UC_SCHEMA}.gadm_boundaries_{ISO_3.lower()}"
selected_gadm_gdf = extract_gadm_boundaries(
    country=COUNTRY,
    adm_level1=ADM_LEVEL1,
    adm_level2=ADM_LEVEL2,
    table_name=gadm_table,
)

# COMMAND ----------

# Extract WorldPop raster (stays as file - cannot store GeoTIFF in table)
ensure_dir(VOLUME_DIR)
raster_path = os.path.join(VOLUME_DIR, f"worldpop_{ISO_3.lower()}_{POPULATION_YEAR}.tif")
extract_worldpop_raster(
    country_iso3=ISO_3,
    population_year=POPULATION_YEAR,
    output_path=raster_path,
)

# COMMAND ----------

# Extract health facilities
# Option A: Query OSM (uncomment to use)
# facilities_table = f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{ISO_3.lower()}_osm"
# extract_health_facilities_osm(ISO_2, facilities_table)

# Option B: Use existing curated file
INPUT_FACILITIES_PATH = "/Volumes/prd_mega/sgpbpi163/vgpbpi163/selected_hosp_input_data.geojson"
facilities_table = f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{ISO_3.lower()}"
extract_existing_facilities(INPUT_FACILITIES_PATH, facilities_table)

# COMMAND ----------

# EXTRACTION SUMMARY

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)
print(f"GADM boundaries:    {gadm_table}")
print(f"Population raster:  {raster_path}")
print(f"Health facilities:  {facilities_table}")
print("=" * 60)
