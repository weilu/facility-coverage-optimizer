# Databricks notebook source
# MAGIC %pip install "numpy<2" geopandas shapely

# COMMAND ----------

# Environment detection and adaptive storage backends
# Enables running pipelines locally (sqlite/csv) or on Databricks (UC tables)
#
# This module provides environment-aware utilities that work both locally
# and in Databricks. Use get_storage_backend() for the OOP approach or
# the convenience wrapper functions for simpler access.

import os
from enum import Enum
from typing import Protocol, runtime_checkable
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.wkt import loads as wkt_loads

# Import from shared.core (local) or assume loaded via %run (Databricks)
if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from shared.core import deduplicate_columns

# COMMAND ----------


class Environment(Enum):
    """Execution environment types."""
    DATABRICKS = "databricks"
    LOCAL = "local"


def detect_environment() -> Environment:
    """
    Detect the current execution environment.

    Returns:
        Environment.DATABRICKS if running in Databricks, else Environment.LOCAL
    """
    # Check for Databricks-specific environment variables
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return Environment.DATABRICKS

    return Environment.LOCAL


# Global environment singleton (lazy initialized)
_current_env: Environment | None = None


def get_environment() -> Environment:
    """Get the current environment (cached)."""
    global _current_env
    if _current_env is None:
        _current_env = detect_environment()
    return _current_env


def is_databricks() -> bool:
    """Check if running in Databricks."""
    return get_environment() == Environment.DATABRICKS


def is_local() -> bool:
    """Check if running locally."""
    return get_environment() == Environment.LOCAL


# -----------------------------------------------------------------------------
# Storage Backend Protocol
# -----------------------------------------------------------------------------

@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backends (UC tables or local files)."""

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        ...

    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        ...

    def ensure_dir(self, path: str) -> None:
        """Create directory if it doesn't exist."""
        ...

    def save_gdf(self, gdf: gpd.GeoDataFrame, table_name: str, mode: str = "overwrite") -> None:
        """Save GeoDataFrame to storage."""
        ...

    def load_gdf(self, table_name: str) -> gpd.GeoDataFrame:
        """Load GeoDataFrame from storage."""
        ...

    def save_pdf(self, pdf: pd.DataFrame, table_name: str, mode: str = "overwrite") -> None:
        """Save pandas DataFrame to storage."""
        ...

    def load_pdf(self, table_name: str) -> pd.DataFrame:
        """Load pandas DataFrame from storage."""
        ...


# -----------------------------------------------------------------------------
# Local Storage Backend (CSV/GeoJSON files)
# -----------------------------------------------------------------------------

class LocalStorageBackend:
    """Local file-based storage backend using CSV and GeoJSON."""

    def __init__(self, base_dir: str = "./data"):
        """
        Initialize local storage backend.

        Args:
            base_dir: Base directory for storing files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _table_to_path(self, table_name: str, ext: str = ".csv") -> Path:
        """Convert table name to file path."""
        # Convert catalog.schema.table to path
        parts = table_name.replace(".", "/")
        return self.base_dir / f"{parts}{ext}"

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists as CSV or GeoJSON file."""
        csv_path = self._table_to_path(table_name, ".csv")
        geojson_path = self._table_to_path(table_name, ".geojson")
        return csv_path.exists() or geojson_path.exists()

    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return Path(path).exists()

    def ensure_dir(self, path: str) -> None:
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Directory ensured: {path}")

    def save_gdf(self, gdf: gpd.GeoDataFrame, table_name: str, mode: str = "overwrite") -> None:
        """Save GeoDataFrame as GeoJSON file."""
        path = self._table_to_path(table_name, ".geojson")
        path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "overwrite" or not path.exists():
            gdf.to_file(path, driver="GeoJSON")
            print(f"GeoDataFrame saved: {path} ({len(gdf)} rows)")
        elif mode == "append":
            existing = gpd.read_file(path)
            combined = pd.concat([existing, gdf], ignore_index=True)
            combined.to_file(path, driver="GeoJSON")
            print(f"GeoDataFrame appended: {path} ({len(combined)} rows total)")

    def load_gdf(self, table_name: str) -> gpd.GeoDataFrame:
        """Load GeoDataFrame from GeoJSON file."""
        path = self._table_to_path(table_name, ".geojson")
        if not path.exists():
            # Try loading from CSV with WKT geometry
            csv_path = self._table_to_path(table_name, ".csv")
            if csv_path.exists():
                pdf = pd.read_csv(csv_path)
                if "geometry_wkt" in pdf.columns:
                    pdf["geometry"] = pdf["geometry_wkt"].apply(
                        lambda w: wkt_loads(w) if w and pd.notna(w) else None
                    )
                    pdf = pdf.drop(columns=["geometry_wkt"])
                    return gpd.GeoDataFrame(pdf, geometry="geometry", crs="EPSG:4326")
            raise FileNotFoundError(f"Table not found: {table_name}")
        return gpd.read_file(path)

    def save_pdf(self, pdf: pd.DataFrame, table_name: str, mode: str = "overwrite") -> None:
        """Save pandas DataFrame as CSV file, creating the file even if the DataFrame is empty."""
        path = self._table_to_path(table_name, ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure a file (with header) is created even when the DataFrame is empty
        if pdf.empty:
            # For append mode, only create the file if it doesn't already exist
            if mode == "append" and path.exists():
                print(f"Empty DataFrame, existing file unchanged: {path}")
                return
            # Write only the header (or an empty file if header already present)
            pdf.to_csv(path, mode="w" if mode == "overwrite" else "a",
                       header=not path.exists(), index=False)
            print(f"Empty DataFrame saved (header only): {path}")
            return

        if mode == "overwrite" or not path.exists():
            pdf.to_csv(path, index=False)
            print(f"DataFrame saved: {path} ({len(pdf)} rows)")
        elif mode == "append":
            pdf.to_csv(path, mode="a", header=not path.exists(), index=False)
            print(f"DataFrame appended: {path}")

    def load_pdf(self, table_name: str) -> pd.DataFrame:
        """Load pandas DataFrame from CSV file."""
        path = self._table_to_path(table_name, ".csv")
        if not path.exists():
            raise FileNotFoundError(f"Table not found: {table_name}")
        return pd.read_csv(path)


# -----------------------------------------------------------------------------
# Databricks Storage Backend (Unity Catalog)
# -----------------------------------------------------------------------------

class DatabricksStorageBackend:
    """Databricks Unity Catalog storage backend."""

    def __init__(self):
        """Initialize Databricks storage backend."""
        self.spark = get_spark()
        self._dbutils = None

    @property
    def dbutils(self):
        """Lazy load dbutils."""
        if self._dbutils is None:
            self._dbutils = dbutils
        return self._dbutils

    def table_exists(self, table_name: str) -> bool:
        """Check if UC table exists."""
        try:
            self.spark.sql(f"DESCRIBE TABLE {table_name}")
            return True
        except Exception:
            return False

    def file_exists(self, path: str) -> bool:
        """Check if file exists using dbutils."""
        try:
            self.dbutils.fs.ls(path)
            return True
        except Exception:
            return False

    def ensure_dir(self, path: str) -> None:
        """Create directory if it doesn't exist using dbutils."""
        try:
            self.dbutils.fs.ls(path)
            print(f"Directory exists: {path}")
        except Exception:
            self.dbutils.fs.mkdirs(path)
            print(f"Directory created: {path}")

    def save_gdf(self, gdf: gpd.GeoDataFrame, table_name: str, mode: str = "overwrite") -> None:
        """Save GeoDataFrame to Unity Catalog table with geometry as WKT."""
        pdf = pd.DataFrame(gdf.drop(columns=["geometry"]))
        pdf["geometry_wkt"] = gdf.geometry.apply(lambda g: g.wkt if g else None)

        # Handle duplicate column names
        pdf.columns = deduplicate_columns(list(pdf.columns))
        pdf = pdf.reset_index(drop=True)

        sdf = self.spark.createDataFrame(pdf.to_dict("records"))
        sdf.write.mode(mode).option("overwriteSchema", "true").saveAsTable(table_name)
        print(f"Table saved: {table_name} ({len(gdf)} rows)")

    def load_gdf(self, table_name: str) -> gpd.GeoDataFrame:
        """Load Unity Catalog table as GeoDataFrame."""
        pdf = self.spark.table(table_name).toPandas()
        pdf["geometry"] = pdf["geometry_wkt"].apply(lambda w: wkt_loads(w) if w else None)
        pdf = pdf.drop(columns=["geometry_wkt"])
        return gpd.GeoDataFrame(pdf, geometry="geometry", crs="EPSG:4326")

    def save_pdf(self, pdf: pd.DataFrame, table_name: str, mode: str = "overwrite") -> None:
        """Save pandas DataFrame to Unity Catalog table."""
        sdf = self.spark.createDataFrame(pdf)
        sdf.write.mode(mode).option("overwriteSchema", "true").saveAsTable(table_name)
        print(f"Table saved: {table_name} ({len(pdf)} rows)")

    def load_pdf(self, table_name: str) -> pd.DataFrame:
        """Load pandas DataFrame from Unity Catalog table."""
        return self.spark.table(table_name).toPandas()


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------

_storage_backend: StorageBackend | None = None


def get_storage_backend(base_dir: str = "./data") -> StorageBackend:
    """
    Get the appropriate storage backend for the current environment.

    Args:
        base_dir: Base directory for local storage (ignored in Databricks)

    Returns:
        StorageBackend instance appropriate for the environment
    """
    global _storage_backend

    if _storage_backend is not None:
        return _storage_backend

    if is_databricks():
        _storage_backend = DatabricksStorageBackend()
    else:
        _storage_backend = LocalStorageBackend(base_dir)

    return _storage_backend


def reset_storage_backend() -> None:
    """Reset the cached storage backend (useful for testing)."""
    global _storage_backend, _current_env
    _storage_backend = None
    _current_env = None


# -----------------------------------------------------------------------------
# Spark Utilities
# -----------------------------------------------------------------------------

def get_spark():
    """Get the active Spark session."""
    from pyspark.sql import SparkSession
    return SparkSession.builder.getOrCreate()


# -----------------------------------------------------------------------------
# Convenience Wrapper Functions
# -----------------------------------------------------------------------------

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    storage = get_storage_backend()
    storage.ensure_dir(path)


def file_exists(path: str) -> bool:
    """Check if file exists."""
    storage = get_storage_backend()
    return storage.file_exists(path)


def table_exists(table_name: str) -> bool:
    """Check if table exists."""
    storage = get_storage_backend()
    return storage.table_exists(table_name)


def gdf_to_uc_table(gdf: gpd.GeoDataFrame, table_name: str, mode: str = "overwrite"):
    """Save GeoDataFrame to storage (UC table or local file)."""
    storage = get_storage_backend()
    storage.save_gdf(gdf, table_name, mode)


def uc_table_to_gdf(table_name: str) -> gpd.GeoDataFrame:
    """Load GeoDataFrame from storage (UC table or local file)."""
    storage = get_storage_backend()
    return storage.load_gdf(table_name)


def pdf_to_uc_table(pdf: pd.DataFrame, table_name: str, mode: str = "overwrite"):
    """Save pandas DataFrame to storage (UC table or local file)."""
    storage = get_storage_backend()
    storage.save_pdf(pdf, table_name, mode)
