"""Tests for shared/transform_ops.py.

The grid helper is pure (numpy/geopandas) and runs anywhere. The H3 helpers
depend on Databricks-native H3 SQL (h3_longlatash3, h3_kring) that open-source
PySpark lacks, so they are marked ``databricks`` and run only on a cluster
(the pipeline test gate); CI deselects them with -m "not databricks".
"""

import pytest
from shapely.geometry import Polygon

from shared.transform_ops import generate_grid_in_polygon


class TestGenerateGridInPolygon:
    """Tests for generate_grid_in_polygon (pure)."""

    def _square(self):
        # ~0.2° square around (28, -15)
        return Polygon([(27.9, -15.1), (28.1, -15.1), (28.1, -14.9), (27.9, -14.9)])

    def test_returns_only_lon_lat_columns(self):
        result = generate_grid_in_polygon(0.05, self._square())
        assert list(result.columns) == ["longitude", "latitude"]

    def test_points_fall_within_polygon(self):
        poly = self._square()
        result = generate_grid_in_polygon(0.05, poly)
        assert len(result) > 0
        from shapely.geometry import Point
        assert all(poly.intersects(Point(lon, lat))
                   for lon, lat in zip(result["longitude"], result["latitude"]))

    def test_finer_spacing_yields_more_points(self):
        poly = self._square()
        coarse = generate_grid_in_polygon(0.1, poly)
        fine = generate_grid_in_polygon(0.02, poly)
        assert len(fine) > len(coarse)

    def test_empty_when_spacing_larger_than_polygon(self):
        # Grid steps land outside the small polygon after integer-degree flooring.
        result = generate_grid_in_polygon(5.0, self._square())
        assert len(result) == 0


@pytest.mark.databricks
class TestAddFacilityH3Index:
    """Requires Databricks H3 SQL (h3_longlatash3)."""

    def test_adds_non_null_h3_index(self, spark):
        from shared.transform_ops import add_facility_h3_index

        sdf = spark.createDataFrame([("f1", 28.0, -15.0)], ["ID", "lon", "lat"])
        out = add_facility_h3_index(sdf, 8)

        assert "h3_index" in out.columns
        assert out.collect()[0]["h3_index"] is not None


@pytest.mark.databricks
class TestComputeCoverageH3Internal:
    """Requires Databricks H3 SQL (h3_longlatash3, h3_kring)."""

    def test_only_nearby_population_is_covered(self, spark):
        from shared.transform_ops import add_facility_h3_index, compute_coverage_h3_internal

        facilities = add_facility_h3_index(
            spark.createDataFrame([("fac_1", 28.0, -15.0)], ["ID", "lon", "lat"]),
            8,
        )
        population = add_facility_h3_index(
            spark.createDataFrame(
                [("p_near", 28.0, -15.0, 100.0), ("p_far", 40.0, 10.0, 50.0)],
                ["ID", "lon", "lat", "population"],
            ),
            8,
        )

        result_sdf, flat_sdf = compute_coverage_h3_internal(
            facilities, population, h3_resolution=8, k_rings=1
        )

        access = {r["ID"]: r["pop_with_access"] for r in result_sdf.collect()}
        assert access["fac_1"] == 100.0  # only the co-located point within k=1

        pairs = {(r["facility_ID"], r["pop_ID"]) for r in flat_sdf.collect()}
        assert ("fac_1", "p_near") in pairs
        assert ("fac_1", "p_far") not in pairs
