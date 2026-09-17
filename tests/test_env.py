"""Tests for environment detection and storage backends in shared/env.py"""

import os
import tempfile
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from shared.env import (
    Environment,
    detect_environment,
    is_local,
    is_databricks,
    LocalStorageBackend,
    DatabricksStorageBackend,
    get_storage_backend,
    reset_storage_backend,
)


class TestEnvironmentDetection:
    """Tests for environment detection."""

    def setup_method(self):
        """Reset environment cache before each test."""
        reset_storage_backend()

    def test_local_environment_detection(self):
        """Test that local environment is detected when not in Databricks."""
        # Remove Databricks env var if present
        old_val = os.environ.pop("DATABRICKS_RUNTIME_VERSION", None)
        try:
            reset_storage_backend()
            env = detect_environment()
            assert env == Environment.LOCAL
        finally:
            if old_val:
                os.environ["DATABRICKS_RUNTIME_VERSION"] = old_val

    def test_is_local(self):
        """is_local is always the inverse of is_databricks (runs locally and on-cluster)."""
        reset_storage_backend()
        assert is_local() is (not is_databricks())

    def test_is_databricks(self):
        """is_databricks reflects the DATABRICKS_RUNTIME_VERSION env (true on a cluster)."""
        reset_storage_backend()
        assert is_databricks() is bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


class TestLocalStorageBackend:
    """Tests for LocalStorageBackend."""

    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = LocalStorageBackend(self.temp_dir)

    def test_table_to_path_conversion(self):
        """Test table name to path conversion."""
        path = self.backend._table_to_path("catalog.schema.table", ".csv")
        assert path.as_posix().endswith("catalog/schema/table.csv")

    def test_table_exists_false(self):
        """Test table_exists returns False for non-existent table."""
        assert self.backend.table_exists("catalog.schema.nonexistent") is False

    def test_file_exists(self):
        """Test file_exists method."""
        # Create a temp file
        temp_file = os.path.join(self.temp_dir, "test.txt")
        with open(temp_file, "w") as f:
            f.write("test")

        assert self.backend.file_exists(temp_file) is True
        assert self.backend.file_exists(os.path.join(self.temp_dir, "nonexistent.txt")) is False

    def test_ensure_dir(self):
        """Test ensure_dir creates directory."""
        new_dir = os.path.join(self.temp_dir, "new_subdir")
        self.backend.ensure_dir(new_dir)
        assert os.path.exists(new_dir)

    def test_save_and_load_pdf(self):
        """Test saving and loading pandas DataFrame."""
        pdf = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "value": [1.1, 2.2, 3.3],
        })

        table_name = "test.schema.dataframe"
        self.backend.save_pdf(pdf, table_name)

        assert self.backend.table_exists(table_name)

        loaded = self.backend.load_pdf(table_name)
        assert len(loaded) == 3
        assert list(loaded.columns) == ["id", "name", "value"]
        assert loaded["name"].tolist() == ["a", "b", "c"]

    def test_save_and_load_gdf(self):
        """Test saving and loading GeoDataFrame."""
        gdf = gpd.GeoDataFrame({
            "id": [1, 2],
            "name": ["point1", "point2"],
            "geometry": [Point(0, 0), Point(1, 1)],
        }, crs="EPSG:4326")

        table_name = "test.schema.geodataframe"
        self.backend.save_gdf(gdf, table_name)

        assert self.backend.table_exists(table_name)

        loaded = self.backend.load_gdf(table_name)
        assert len(loaded) == 2
        assert loaded.crs.to_epsg() == 4326
        assert loaded.iloc[0].geometry.x == 0
        assert loaded.iloc[1].geometry.x == 1

    def test_pdf_append_mode(self):
        """Test appending to existing CSV."""
        pdf1 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        pdf2 = pd.DataFrame({"id": [3, 4], "value": [30, 40]})

        table_name = "test.schema.append_test"
        self.backend.save_pdf(pdf1, table_name, mode="overwrite")
        self.backend.save_pdf(pdf2, table_name, mode="append")

        loaded = self.backend.load_pdf(table_name)
        assert len(loaded) == 4

    def test_load_nonexistent_raises(self):
        """Test that loading non-existent table raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.backend.load_pdf("test.schema.nonexistent")

        with pytest.raises(FileNotFoundError):
            self.backend.load_gdf("test.schema.nonexistent")


class TestStorageBackendFactory:
    """Tests for get_storage_backend factory."""

    def setup_method(self):
        """Reset cached backend before each test."""
        reset_storage_backend()

    def test_get_storage_backend_matches_environment(self):
        """Factory returns the backend for the current environment (local or Databricks)."""
        backend = get_storage_backend()
        expected = DatabricksStorageBackend if is_databricks() else LocalStorageBackend
        assert isinstance(backend, expected)

    def test_get_storage_backend_caches(self):
        """Test that factory caches the backend."""
        backend1 = get_storage_backend()
        backend2 = get_storage_backend()
        assert backend1 is backend2

    def test_reset_clears_cache(self):
        """Test that reset_storage_backend clears cache."""
        backend1 = get_storage_backend()
        reset_storage_backend()
        backend2 = get_storage_backend()
        assert backend1 is not backend2


class TestDdhHelpers:
    """Tests for the DDH volume/URL download helpers."""

    def test_ddh_volume_path_maps_url(self):
        from shared.env import ddh_volume_path
        url = (
            "https://datacatalogfiles.worldbank.org/ddh-published/0038272/DR0095369/"
            "World%20Bank%20Official%20Boundaries%20(GeoJSON)/"
            "World%20Bank%20Official%20Boundaries%20-%20Admin%200.geojson"
        )
        assert ddh_volume_path(url) == (
            "/Volumes/prd_development_data/files/ddh/0038272/DR0095369/"
            "World Bank Official Boundaries (GeoJSON)/"
            "World Bank Official Boundaries - Admin 0.geojson"
        )

    def test_ddh_volume_path_rejects_non_ddh_url(self):
        from shared.env import ddh_volume_path
        with pytest.raises(ValueError):
            ddh_volume_path("https://example.com/foo/bar.geojson")

    def test_ddh_bytes_reads_volume_when_present(self, tmp_path, monkeypatch):
        import shared.env as env
        f = tmp_path / "cached.bin"
        f.write_bytes(b"volume-copy")
        # Resolve the "volume path" to our temp file; ddh_bytes should read it
        # without touching the network.
        monkeypatch.setattr(env, "ddh_volume_path", lambda url: str(f))
        assert env.ddh_bytes("https://datacatalogfiles.worldbank.org/ddh-published/x") == b"volume-copy"
