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
import numpy as np
import pycountry
import rasterio
from rasterio.windows import Window
from gadm import GADMDownloader
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
import zipfile

# COMMAND ----------

# CONFIGURATION

COUNTRY = "Zambia"
ADM_LEVEL1 = 'Central'
ADM_LEVEL2 = None
POPULATION_YEAR = 2025

UC_CATALOG = "prd_mega"
UC_SCHEMA = "sgpbpi163"
VOLUME_DIR = f"/Volumes/{UC_CATALOG}/sgpbpi163/vgpbpi163"


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

# COMMAND ----------

# RUN EXTRACTION PIPELINE

iso_codes = get_country_codes(COUNTRY)
ISO_2 = iso_codes["alpha_2"]
ISO_3 = iso_codes["alpha_3"]
print(f"Country: {COUNTRY} | ISO-2: {ISO_2} | ISO-3: {ISO_3}")

# COMMAND ----------


gadm_table = f"{UC_CATALOG}.{UC_SCHEMA}.gadm_boundaries_{ISO_3.lower()}"
population_table = f"{UC_CATALOG}.{UC_SCHEMA}.population_{ISO_3.lower()}_{POPULATION_YEAR}"
facilities_table = f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{ISO_3.lower()}_osm_{ADM_LEVEL1.lower()}"
lgu_table = f"{UC_CATALOG}.{UC_SCHEMA}.gadm_boundaries_lgu_{COUNTRY.lower()}_{ADM_LEVEL1.lower()}"

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

    print("Downloaded the boundaries | Uploading to UC Table")
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

def extract_health_facilities_osm(iso_2: str, table_name: str, selected_gadm: gpd.GeoDataFrame, adm_level_name: str = "AOI", force: bool = False) -> pd.DataFrame:
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
    gdf_health = gdf_health.to_crs(selected_gadm.crs)
    selected_health = gpd.sjoin(gdf_health, selected_gadm, predicate='within')
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

# Extract GADM boundaries
selected_gadm_gdf = extract_gadm_boundaries(
    country=COUNTRY,
    adm_level1=ADM_LEVEL1,
    adm_level2=ADM_LEVEL2,
    table_name=gadm_table,
    force=True,
)

# COMMAND ----------

# Extract WorldPop raster (download to Volume)
ensure_dir(VOLUME_DIR)
raster_path = os.path.join(VOLUME_DIR, f"worldpop_{ISO_3.lower()}_{POPULATION_YEAR}.tif")
extract_worldpop_raster(
    country_iso3=ISO_3,
    population_year=POPULATION_YEAR,
    output_path=raster_path,
)

# COMMAND ----------

# Convert raster to UC table using chunked processing
# Long-running – takes 7 minutes to process
extract_population_chunked(
    raster_path=raster_path,
    table_name=population_table,
    chunk_size=1024,
)

# COMMAND ----------

# Extract health facilities
# Option A: Query OSM (uncomment to use)
extract_health_facilities_osm(iso_2= ISO_2, table_name= facilities_table, selected_gadm= selected_gadm_gdf, adm_level_name= ADM_LEVEL1, force=True)

# Option B: Use existing curated file
# INPUT_FACILITIES_PATH = f"{VOLUME_DIR}/selected_hosp_input_data.geojson"
# facilities_table = f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{ISO_3.lower()}"
# extract_existing_facilities(INPUT_FACILITIES_PATH, facilities_table)

# COMMAND ----------

# EXTRACTION SUMMARY

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)
print(f"GADM boundaries:    {gadm_table}")
print(f"Population raster:  {raster_path}")
print(f"Population table:   {population_table}")
print(f"Health facilities:  {facilities_table}")
print("=" * 60)

# COMMAND ----------

def extract_gadm_boundaries_lgu(
    country_iso3: str,
    table_name: str,
    gadm_version: str = "4.1",
    ad_level: int = 2,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """
    Downloads GADM boundaries at the specified admin level directly from the
    UCDAVIS geodata server (supports GADM 4.1 which has all 116 Zambia districts)
    and saves to UC table with ONLY two columns:
        - LGU           : district name (NAME_2 for ad_level=2)
        - geometry_wkt  : polygon geometry in WKT (EPSG:4326)

    Args:
        country_iso3 : ISO-3 country code  (e.g. "ZMB")
        table_name   : fully-qualified UC table  (catalog.schema.table)
        gadm_version : GADM dataset version, default "4.1"
        ad_level     : admin level to extract  (2 = district/LGU)
        force        : overwrite even if the table already exists

    Returns:
        GeoDataFrame with columns [LGU, geometry]
    """
    if not force and table_exists(table_name):
        print(f"LGU boundaries already exist, loading: {table_name}")
        return uc_table_to_gdf(table_name)

    # ── Build the download URL ────────────────────────────────────────────
    # Example: https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_ZMB_shp.zip
    version_nodot = gadm_version.replace(".", "")
    zip_url = (
        f"https://geodata.ucdavis.edu/gadm/gadm{gadm_version}/shp/"
        f"gadm{version_nodot}_{country_iso3}_shp.zip"
    )
    print(f"Downloading GADM {gadm_version} shapefile: {zip_url}")

    # ── Download zip to a temp file on the Volume ─────────────────────────
    zip_path = os.path.join(VOLUME_DIR, f"gadm{version_nodot}_{country_iso3}_shp.zip")

    if not file_exists(zip_path):
        urllib.request.urlretrieve(zip_url, zip_path)
        print(f"  Downloaded: {zip_path}")
    else:
        print(f"  Zip already cached: {zip_path}")

    # ── Unzip and read the level-specific shapefile ───────────────────────
    # The zip contains files named e.g. gadm41_ZMB_0.shp, gadm41_ZMB_1.shp, gadm41_ZMB_2.shp
    extract_dir = os.path.join(VOLUME_DIR, f"gadm{version_nodot}_{country_iso3}_shp")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
        print(f"  Extracted to: {extract_dir}")

    shp_filename = f"gadm{version_nodot}_{country_iso3}_{ad_level}.shp"
    shp_path = os.path.join(extract_dir, shp_filename)

    lgu_raw_gdf = gpd.read_file(shp_path)
    print(f"  Shapefile loaded: {len(lgu_raw_gdf)} features (GADM {gadm_version}, level {ad_level})")

    # ── Normalise to mandatory schema: LGU + geometry ─────────────────────
    name_col = f"NAME_{ad_level}"          # e.g. NAME_2 for districts
    if name_col not in lgu_raw_gdf.columns:
        raise ValueError(
            f"Expected column '{name_col}' not found. "
            f"Available: {list(lgu_raw_gdf.columns)}"
        )

    lgu_gdf = (
        lgu_raw_gdf[[name_col, "geometry"]]
        .copy()
        .rename(columns={name_col: "LGU"})
        .reset_index(drop=True)
    )

    # Ensure WGS-84
    if lgu_gdf.crs is None or lgu_gdf.crs.to_epsg() != 4326:
        lgu_gdf = lgu_gdf.to_crs(epsg=4326)

    print(
        f"Normalised to {len(lgu_gdf)} LGU boundaries "
        f"| Uploading to UC table: {table_name}"
    )
    # gdf_to_uc_table produces exactly: LGU, geometry_wkt
    gdf_to_uc_table(lgu_gdf, table_name)
    return lgu_gdf


# ============================================================
# ── Execution block — append after health facilities cell ──
# ============================================================

lgu_gdf = extract_gadm_boundaries_lgu(
    country_iso3=ISO_3,          # "ZMB"  (already resolved above)
    table_name=lgu_table,
    gadm_version="4.1",          # <-- GADM 4.1 = 116 districts
    ad_level=2,
)
print(f"LGU count : {len(lgu_gdf)}")
print(lgu_gdf[["LGU"]].to_string(max_rows=10))

# COMMAND ----------

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)
print(f"GADM country boundary: {gadm_table}")
print(f"GADM LGU boundaries:   {lgu_table}  ({len(lgu_gdf)} LGUs)")
print(f"Population raster:     {raster_path}")
print(f"Population table:      {population_table}")
print(f"Health facilities:     {facilities_table}")
print("=" * 60)
