"""Cluster-aware ``spark`` fixture, and an autouse guard blocking any Unity Catalog
write so the suite can never touch a real (pim/sgpbpi163) destination — the gate
runs on a cluster with real credentials."""

from unittest import mock

import pytest


def pytest_configure(config):
    # Registered here (not only in pyproject) because on-cluster the suite runs via
    # `pytest --pyargs tests` from the installed wheel, where pyproject isn't the rootdir.
    config.addinivalue_line(
        "markers",
        "databricks: requires a real Databricks cluster (H3 SQL); skipped off-cluster",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip `databricks`-marked tests off-cluster (they need real H3 SQL).

    So a bare `pytest` run locally skips them instead of failing; the on-cluster
    gate runs them. CI additionally passes -m "not databricks" to avoid pyspark.
    """
    from shared.env import is_databricks

    if is_databricks():
        return
    skip = pytest.mark.skip(reason="requires a Databricks cluster (H3 SQL)")
    for item in items:
        if "databricks" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def spark():
    """Spark session for integration tests.

    On a Databricks cluster, reuse the existing session and never stop it —
    stopping the shared session would break subsequent pipeline tasks. Locally,
    build a throwaway ``local[2]`` session and stop it at teardown.
    """
    from pyspark.sql import SparkSession

    from shared.env import is_databricks

    if is_databricks():
        yield SparkSession.builder.getOrCreate()
        return

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("pimpam-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(autouse=True, scope="session")
def _block_uc_writes():
    """Fail any test that attempts a Unity Catalog write.

    Tests exercise only the pure transform functions that return DataFrames;
    they must never call the caching wrappers that persist to UC. This patches
    the write entry points so an accidental write raises instead of mutating a
    real table. Local file writes (LocalStorageBackend) are intentionally left
    alone. pyspark is patched only if importable, so PySpark-free CI is fine.
    """
    def _forbidden(name):
        def _raise(*args, **kwargs):
            raise AssertionError(
                f"Test attempted a Unity Catalog write via {name}; "
                "tests must not write to prd_mega (dev pim / prod sgpbpi163)."
            )
        return _raise

    patchers = []

    try:
        from pyspark.sql import DataFrameWriter
        patchers.append(mock.patch.object(
            DataFrameWriter, "saveAsTable", _forbidden("DataFrameWriter.saveAsTable")))
    except ImportError:
        pass  # PySpark-free CI: no Spark writer to guard

    import shared.env as env
    if hasattr(env, "gdf_to_uc_table"):
        patchers.append(mock.patch.object(
            env, "gdf_to_uc_table", _forbidden("shared.env.gdf_to_uc_table")))
    if hasattr(env, "DatabricksStorageBackend"):
        for attr in ("save_gdf", "save_pdf"):
            patchers.append(mock.patch.object(
                env.DatabricksStorageBackend, attr,
                _forbidden(f"DatabricksStorageBackend.{attr}")))

    for p in patchers:
        p.start()
    yield
    for p in patchers:
        p.stop()
