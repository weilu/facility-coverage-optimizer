# Databricks notebook source
# Shared utility functions for extract and transform pipelines

import pandas as pd
import geopandas as gpd
from shapely.wkt import loads as wkt_loads

# COMMAND ----------

def get_spark():
    """Get the active Spark session."""
    from pyspark.sql import SparkSession
    return SparkSession.builder.getOrCreate()


def get_dbutils():
    """Get dbutils in Databricks environment."""
    try:
        from pyspark.dbutils import DBUtils
        return DBUtils(get_spark())
    except ImportError:
        # Fallback for Databricks notebook environment
        import IPython
        return IPython.get_ipython().user_ns.get("dbutils")


# COMMAND ----------

def ensure_dir(path: str):
    """Create directory if it doesn't exist using dbutils."""
    dbutils = get_dbutils()
    try:
        dbutils.fs.ls(path)
        print(f"Directory exists: {path}")
    except Exception:
        dbutils.fs.mkdirs(path)
        print(f"Directory created: {path}")


def file_exists(path: str) -> bool:
    """Check if file exists using dbutils."""
    dbutils = get_dbutils()
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


def table_exists(table_name: str) -> bool:
    """Check if UC table exists."""
    spark = get_spark()
    try:
        spark.sql(f"DESCRIBE TABLE {table_name}")
        return True
    except Exception:
        return False


# COMMAND ----------

def gdf_to_uc_table(gdf: gpd.GeoDataFrame, table_name: str, mode: str = "overwrite"):
    """Save GeoDataFrame to Unity Catalog table with geometry as WKT."""
    spark = get_spark()

    pdf = pd.DataFrame(gdf.drop(columns=["geometry"]))
    pdf["geometry_wkt"] = gdf.geometry.apply(lambda g: g.wkt if g else None)

    # Handle duplicate column names (case-insensitive in Spark)
    cols_lower = [c.lower() for c in pdf.columns]
    seen = set()
    new_cols = []
    for i, col in enumerate(pdf.columns):
        col_lower = cols_lower[i]
        if col_lower in seen:
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
    spark = get_spark()

    pdf = spark.table(table_name).toPandas()
    pdf["geometry"] = pdf["geometry_wkt"].apply(lambda w: wkt_loads(w) if w else None)
    pdf = pdf.drop(columns=["geometry_wkt"])
    return gpd.GeoDataFrame(pdf, geometry="geometry", crs="EPSG:4326")


def pdf_to_uc_table(pdf: pd.DataFrame, table_name: str, mode: str = "overwrite"):
    """Save pandas DataFrame to Unity Catalog table."""
    spark = get_spark()

    sdf = spark.createDataFrame(pdf)
    sdf.write.mode(mode).saveAsTable(table_name)
    print(f"Table saved: {table_name} ({len(pdf)} rows)")


# COMMAND ----------

def get_country_codes(country_name: str):
    """Look up ISO codes for a country name."""
    import pycountry

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
