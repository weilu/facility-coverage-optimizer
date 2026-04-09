# Databricks notebook source
# MAGIC %pip install shapely rasterio pycountry folium plotly scikit-learn pyproj

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
import numpy as np
import pycountry
import rasterio
from rasterio.windows import Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

# COMMAND ----------

# CONFIGURATION

COUNTRY = "Zambia"
POPULATION_YEAR = 2025

UC_CATALOG = "prd_mega"
UC_SCHEMA = "sgpbpi163"
VOLUME_DIR = f"/Volumes/{UC_CATALOG}/sgpbpi163/vgpbpi163"

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
    pdf = pd.DataFrame(gdf.drop(columns=["geometry"]))
    pdf["geometry_wkt"] = gdf.geometry.apply(lambda g: g.wkt if g else None)

    # Handle duplicate column names (case-insensitive in Spark)
    cols_lower = [c.lower() for c in pdf.columns]
    seen = set()
    new_cols = []
    for i, col in enumerate(pdf.columns):
        col_lower = cols_lower[i]
        if col_lower in seen:
            # Rename duplicate by appending suffix
            new_col = f"{col}_dup"
            while new_col.lower() in seen:
                new_col = f"{new_col}_"
            new_cols.append(new_col)
            seen.add(new_col.lower())
        else:
            new_cols.append(col)
            seen.add(col_lower)
    pdf.columns = new_cols
    pdf = pdf.reset_index(drop=True)
    sdf = spark.createDataFrame(pdf.to_dict('records')) 
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

# COMMAND ----------

# RUN EXTRACTION PIPELINE

iso_codes = get_country_codes(COUNTRY)
ISO_2 = iso_codes["alpha_2"]
ISO_3 = iso_codes["alpha_3"]
print(f"Country: {COUNTRY} | ISO-2: {ISO_2} | ISO-3: {ISO_3}")

# Determine which provinces to process
if ADM_LEVEL1_LIST == []:
    provinces_to_process = get_all_adm_level1_names(ISO_3)
elif ADM_LEVEL1_LIST is None:
    provinces_to_process = [None]  # Process entire country
else:
    provinces_to_process = ADM_LEVEL1_LIST

print(f"Will process {len(provinces_to_process)} region(s): {provinces_to_process}")

# COMMAND ----------

# EXTRACT: WORLD BANK OFFICIAL BOUNDARIES

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


def extract_boundaries(
    country_iso3: str,
    adm_level1: str | None,
    table_name: str,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """
    Downloads World Bank Official Boundaries and saves to UC table.
    Returns the selected boundary GeoDataFrame.

    World Bank attributes:
    - ISO_A3: ISO 3-letter country code
    - NAM_0: Country name
    - NAM_1: Admin level 1 name (province/state)
    """
    if not force and table_exists(table_name):
        print(f"WB boundaries already exist, loading: {table_name}")
        return uc_table_to_gdf(table_name)

    if adm_level1 is not None:
        cache_path = os.path.join(VOLUME_DIR, "wb_admin1.geojson")
        gdf = download_wb_geojson(WB_ADMIN1_URL, cache_path)
        gdf_country = gdf[gdf["ISO_A3"] == country_iso3]
        selected = gdf_country[gdf_country["NAM_1"] == adm_level1]
        if selected.empty:
            available = sorted(gdf_country["NAM_1"].unique().tolist())
            raise ValueError(f"Admin1 '{adm_level1}' not found. Available: {available}")
    else:
        cache_path = os.path.join(VOLUME_DIR, "wb_admin0.geojson")
        gdf = download_wb_geojson(WB_ADMIN0_URL, cache_path)
        selected = gdf[gdf["ISO_A3"] == country_iso3]
        if selected.empty:
            raise ValueError(f"Country ISO3 '{country_iso3}' not found in WB boundaries")

    print(f"Selected {len(selected)} boundary feature(s)")
    gdf_to_uc_table(selected, table_name)
    return selected


def get_all_adm_level1_names(country_iso3: str) -> list[str]:
    """Get all admin level 1 (province/state) names for a country from World Bank boundaries."""
    cache_path = os.path.join(VOLUME_DIR, "wb_admin1.geojson")
    gdf = download_wb_geojson(WB_ADMIN1_URL, cache_path)
    gdf_country = gdf[gdf["ISO_A3"] == country_iso3]
    provinces = sorted(gdf_country["NAM_1"].unique().tolist())
    print(f"Found {len(provinces)} admin level 1 regions (WB): {provinces}")
    return provinces


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

# EXTRACT: POPULATION RASTER TO UC TABLE (CHUNKED)

def extract_population_chunked(
    raster_path: str,
    table_name: str,
    chunk_size: int = 1024,
    h3_resolution: int = 8,
    force: bool = False,
) -> int:
    """
    Reads WorldPop raster in chunks and saves populated pixels to UC table.
    Uses windowed reading to avoid loading entire raster into memory.
    Adds H3 index for fast spatial filtering.
    Returns total number of populated pixels.
    """
    if not force and table_exists(table_name):
        count = spark.table(table_name).count()
        print(f"Population table already exists: {table_name} ({count:,} rows)")
        return count

    print(f"Processing raster in chunks: {raster_path}")
    print(f"  H3 resolution: {h3_resolution}")

    schema = StructType([
        StructField("xcoord", DoubleType(), False),
        StructField("ycoord", DoubleType(), False),
        StructField("population", DoubleType(), False),
    ])

    total_pixels = 0
    first_chunk = True

    with rasterio.open(raster_path) as src:
        height = src.height
        width = src.width
        transform_affine = src.transform

        n_row_chunks = (height + chunk_size - 1) // chunk_size
        n_col_chunks = (width + chunk_size - 1) // chunk_size
        total_chunks = n_row_chunks * n_col_chunks

        print(f"  Raster size: {width} x {height}")
        print(f"  Chunk size: {chunk_size} x {chunk_size}")
        print(f"  Total chunks: {total_chunks}")

        chunk_num = 0
        for row_off in range(0, height, chunk_size):
            for col_off in range(0, width, chunk_size):
                chunk_num += 1

                win_height = min(chunk_size, height - row_off)
                win_width = min(chunk_size, width - col_off)
                window = Window(col_off, row_off, win_width, win_height)

                data = src.read(1, window=window)

                rows, cols = np.where(data > 0)
                if len(rows) == 0:
                    continue

                values = data[rows, cols].astype(float)

                abs_rows = rows + row_off
                abs_cols = cols + col_off
                x_coords, y_coords = rasterio.transform.xy(
                    transform_affine, abs_rows, abs_cols, offset="center"
                )

                pdf = pd.DataFrame({
                    "xcoord": np.array(x_coords, dtype=float),
                    "ycoord": np.array(y_coords, dtype=float),
                    "population": values,
                })

                sdf = spark.createDataFrame(pdf, schema=schema)

                # Add H3 index (Photon-accelerated)
                sdf = sdf.withColumn(
                    "h3_index",
                    F.expr(f"h3_longlatash3(xcoord, ycoord, {h3_resolution})")
                )

                if first_chunk:
                    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
                    first_chunk = False
                else:
                    sdf.write.mode("append").saveAsTable(table_name)

                total_pixels += len(rows)

                if chunk_num % 50 == 0 or chunk_num == total_chunks:
                    print(f"  Processed chunk {chunk_num}/{total_chunks}, pixels so far: {total_pixels:,}")

    print(f"Population table saved: {table_name} ({total_pixels:,} rows)")
    return total_pixels

# COMMAND ----------

# EXTRACT: HEALTH FACILITIES FROM OSM

def extract_health_facilities_osm(iso_2: str, table_name: str, selected_boundary: gpd.GeoDataFrame, adm_level_name: str = "AOI", force: bool = False) -> pd.DataFrame:
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

    # ── Convert geometry → WKT string before writing to Spark ──────────────
    selected_health_pdf = selected_health.copy()
    selected_health_pdf["geometry_wkt"] = selected_health_pdf["geometry"].apply(
        lambda geom: geom.wkt if geom is not None else None
    )
    selected_health_pdf = selected_health_pdf.drop(columns=["geometry"])
    print(selected_health_pdf.columns)
    selected_health_pdf = selected_health_pdf.rename(columns={
        "id": "osm_id",  # your constructed id (level_0 + '_current')
    })

    pdf_to_uc_table(selected_health_pdf, table_name)
    return selected_health_pdf

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

# COUNTRY-LEVEL EXTRACTIONS (done once regardless of province selection)

# Extract WorldPop raster (download to Volume)
ensure_dir(VOLUME_DIR)
raster_path = os.path.join(VOLUME_DIR, f"worldpop_{ISO_3.lower()}_{POPULATION_YEAR}.tif")
extract_worldpop_raster(
    country_iso3=ISO_3,
    population_year=POPULATION_YEAR,
    output_path=raster_path,
    force=FORCE_RECOMPUTE,
)

# COMMAND ----------

# Convert raster to UC table using chunked processing (country-level)
# Long-running – takes 7 minutes to process
country_population_table = f"{UC_CATALOG}.{UC_SCHEMA}.population_{ISO_3.lower()}_{POPULATION_YEAR}"
extract_population_chunked(
    raster_path=raster_path,
    table_name=country_population_table,
    chunk_size=1024,
    force=FORCE_RECOMPUTE,
)

# COMMAND ----------

# PROVINCE-LEVEL EXTRACTIONS (loop over all configured provinces)

extraction_results = []

for adm_level1 in provinces_to_process:
    print("\n" + "=" * 60)
    print(f"PROCESSING: {adm_level1 if adm_level1 else 'ENTIRE COUNTRY'}")
    print("=" * 60)

    # Generate table names for this province
    tables = get_table_names(COUNTRY, ISO_3, adm_level1, POPULATION_YEAR)
    boundaries_table = tables["boundaries"]
    population_table = tables["population"]
    facilities_table = tables["facilities"]
    lgu_table = tables["lgu"]

    print(f"  Boundaries Table: {boundaries_table}")
    print(f"  Population Table: {population_table}")
    print(f"  Facilities Table: {facilities_table}")
    print(f"  LGU Table: {lgu_table}")

    # Extract administrative boundaries for this province
    selected_boundary_gdf = extract_boundaries(
        country_iso3=ISO_3,
        adm_level1=adm_level1,
        table_name=boundaries_table,
        force=FORCE_RECOMPUTE,
    )

    # Extract health facilities for this province
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
        "boundaries_table": boundaries_table,
        "population_table": population_table,
        "facilities_table": facilities_table,
        "lgu_table": lgu_table,
    })

    print(f"  Completed: {adm_level1 if adm_level1 else 'ENTIRE COUNTRY'}")

# COMMAND ----------

# EXTRACTION SUMMARY (after province loop)

print("\n" + "=" * 60)
print("ALL PROVINCE EXTRACTIONS COMPLETE")
print("=" * 60)
print(f"Population raster (country): {raster_path}")
print(f"Population table (country):  {country_population_table}")
print(f"Regions processed: {len(extraction_results)}")
for result in extraction_results:
    region = result["adm_level1"] if result["adm_level1"] else "Country"
    print(f"  - {region}: {result['boundaries_table']}")
print("=" * 60)

# COMMAND ----------

def extract_boundaries_lgu(
    country_iso3: str,
    table_name: str,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """
    Downloads World Bank Admin2 boundaries for LGU analysis.
    Normalizes to schema: LGU + geometry.
    """
    if not force and table_exists(table_name):
        print(f"WB LGU boundaries already exist, loading: {table_name}")
        return uc_table_to_gdf(table_name)

    cache_path = os.path.join(VOLUME_DIR, "wb_admin2.geojson")
    gdf = download_wb_geojson(WB_ADMIN2_URL, cache_path)
    gdf_country = gdf[gdf["ISO_A3"] == country_iso3]

    if gdf_country.empty:
        raise ValueError(f"No Admin2 boundaries found for ISO3 '{country_iso3}'")

    lgu_gdf = (
        gdf_country[["NAM_2", "geometry"]]
        .copy()
        .rename(columns={"NAM_2": "LGU"})
        .reset_index(drop=True)
    )

    # Ensure WGS-84
    if lgu_gdf.crs is None or lgu_gdf.crs.to_epsg() != 4326:
        lgu_gdf = lgu_gdf.to_crs(epsg=4326)

    print(f"Extracted {len(lgu_gdf)} LGU boundaries (WB) | Uploading to: {table_name}")
    gdf_to_uc_table(lgu_gdf, table_name)
    return lgu_gdf


# ============================================================
# EXTRACT: LGU (DISTRICT) BOUNDARIES (country-level, done once)
# ============================================================

country_lgu_table = f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_lgu_{COUNTRY.lower()}"
lgu_gdf = extract_boundaries_lgu(
    country_iso3=ISO_3,  # e.g. "ZMB" (resolved above)
    table_name=country_lgu_table,
    force=FORCE_RECOMPUTE,
)
print(f"LGU count : {len(lgu_gdf)}")
print(lgu_gdf[["LGU"]].to_string(max_rows=10))

# COMMAND ----------

# FINAL EXTRACTION SUMMARY

print("\n" + "=" * 60)
print("ALL EXTRACTIONS COMPLETE")
print("=" * 60)
print(f"Country: {COUNTRY} ({ISO_3})")
print(f"Population raster:     {raster_path}")
print(f"Population table:      {country_population_table}")
print(f"LGU boundaries:        {country_lgu_table} ({len(lgu_gdf)} LGUs)")
print(f"Provinces processed:   {len(extraction_results)}")
for result in extraction_results:
    region = result["adm_level1"] if result["adm_level1"] else "Country"
    print(f"  - {region}")
print("=" * 60)
