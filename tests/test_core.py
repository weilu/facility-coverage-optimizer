
import pytest
from shared.core import (
    get_k_rings,
    sanitize_col_name,
    get_extract_table_names,
    get_transform_table_names,
    build_transform_combinations,
    solve_mclp_greedy,
    deduplicate_columns,
)


class TestGetKRings:

    def test_5km_resolution_8(self):
        # H3 resolution 8 edge length is 461m
        # 5000m / 461m = 10.85 -> ceil = 11
        result = get_k_rings(5000, 8)
        assert result == 11

    def test_10km_resolution_8(self):
        # 10000m / 461m = 21.69 -> ceil = 22
        result = get_k_rings(10000, 8)
        assert result == 22

    def test_1km_resolution_8(self):
        # 1000m / 461m = 2.17 -> ceil = 3
        result = get_k_rings(1000, 8)
        assert result == 3

    def test_resolution_7(self):
        # H3 resolution 7 edge length is 1220m
        # 5000m / 1220m = 4.1 -> ceil = 5
        result = get_k_rings(5000, 7)
        assert result == 5

    def test_invalid_resolution(self):
        with pytest.raises(ValueError, match="H3 resolution must be 4-10"):
            get_k_rings(5000, 3)

        with pytest.raises(ValueError, match="H3 resolution must be 4-10"):
            get_k_rings(5000, 11)


class TestSanitizeColName:

    def test_simple_name(self):
        assert sanitize_col_name("Lusaka") == "lgu_Lusaka"

    def test_name_with_space(self):
        assert sanitize_col_name("Kapiri Mposhi") == "lgu_Kapiri_Mposhi"

    def test_name_with_parentheses(self):
        assert sanitize_col_name("Choma (East)") == "lgu_Choma_East"

    def test_name_with_hyphen(self):
        assert sanitize_col_name("North-Western") == "lgu_North_Western"

    def test_name_with_leading_trailing_spaces(self):
        assert sanitize_col_name("  Lusaka  ") == "lgu_Lusaka"

    def test_name_with_multiple_special_chars(self):
        assert sanitize_col_name("Test--Name  (2)") == "lgu_Test_Name_2"

    def test_numeric_start(self):
        # Should prefix with lgu_ to avoid starting with digit
        assert sanitize_col_name("123District") == "lgu_123District"


class TestGetExtractTableNames:

    def test_country_level(self):
        result = get_extract_table_names("prd", "sgp", "ZMB", None, 2025)
        assert result["boundaries"] == "prd.sgp.wb_boundaries_zmb"
        assert result["population"] == "prd.sgp.population_zmb_2025"
        assert result["facilities"] == "prd.sgp.health_facilities_zmb_osm"
        assert result["lgu"] == "prd.sgp.wb_boundaries_lgu_zmb"

    def test_province_level(self):
        result = get_extract_table_names("prd", "sgp", "ZMB", "Northern", 2025)
        assert result["boundaries"] == "prd.sgp.wb_boundaries_zmb_northern_province"
        assert result["population"] == "prd.sgp.population_zmb_2025_northern_province"
        assert result["facilities"] == "prd.sgp.health_facilities_zmb_osm_northern_province"
        assert result["lgu"] == "prd.sgp.wb_boundaries_lgu_zmb_northern_province"

    def test_hyphenated_province(self):
        result = get_extract_table_names("prd", "sgp", "ZMB", "North-Western", 2025)
        assert "_north_western_province" in result["boundaries"]


class TestGetTransformTableNames:

    def test_province_with_distance(self):
        result = get_transform_table_names("prd", "sgp", "ZMB", "Northern", 2025, 5000)
        assert result["population_aoi"] == "prd.sgp.population_aoi_zmb_2025_northern_province_5km"
        assert result["facilities_h3"] == "prd.sgp.facilities_h3_zmb_northern_province_5km"
        assert result["potential_coverage"] == "prd.sgp.potential_coverage_zmb_northern_province_5km"

    def test_country_level_with_distance(self):
        result = get_transform_table_names("prd", "sgp", "ZMB", None, 2025, 10000)
        assert result["population_aoi"] == "prd.sgp.population_aoi_zmb_2025_10km"
        assert result["lgu_accessibility"] == "prd.sgp.lgu_accessibility_results_zmb_10km"

    def test_all_keys_present(self):
        result = get_transform_table_names("prd", "sgp", "ZMB", "Northern", 2025, 5000)
        expected_keys = [
            "boundaries", "facilities", "population", "population_aoi",
            "facilities_h3", "facilities_coverage", "potential_locations",
            "potential_coverage", "lgu", "lgu_accessibility"
        ]
        assert all(key in result for key in expected_keys)


class TestBuildTransformCombinations:

    def test_single_province_single_distance(self):
        result = build_transform_combinations(["Northern"], [5000])
        assert result == [("Northern", 5000)]

    def test_multiple_provinces_multiple_distances(self):
        result = build_transform_combinations(["Northern", "Southern"], [5000, 10000])
        assert len(result) == 4
        assert ("Northern", 5000) in result
        assert ("Northern", 10000) in result
        assert ("Southern", 5000) in result
        assert ("Southern", 10000) in result

    def test_empty_list_means_country_level(self):
        result = build_transform_combinations([], [5000])
        assert result == [(None, 5000)]

    def test_none_means_country_level(self):
        result = build_transform_combinations(None, [5000, 10000])
        assert result == [(None, 5000), (None, 10000)]


class TestSolveMclpGreedy:

    def test_simple_case(self):
        # Simple scenario: 3 H3 cells, 2 potential facilities
        w = {"h3_1": 100, "h3_2": 200, "h3_3": 150}
        IJ = {
            "h3_1": ["fac_A"],
            "h3_2": ["fac_A", "fac_B"],
            "h3_3": ["fac_B"],
        }
        J_existing = []
        J_potential = ["fac_A", "fac_B"]

        results = solve_mclp_greedy(w, IJ, J_existing, J_potential, max_new_facilities=2)

        assert len(results) == 3
        # First facility should cover most population
        # fac_A covers h3_1 (100) + h3_2 (200) = 300
        # fac_B covers h3_2 (200) + h3_3 (150) = 350
        assert results[0]["p"] == 0
        assert results[0]["objective"] == 0

        assert results[1]["p"] == 1
        assert "fac_B" in results[1]["selected_facilities"]
        assert results[1]["objective"] == 350

        # Second facility adds remaining coverage
        assert results[2]["p"] == 2
        assert results[2]["objective"] == 450  # All cells covered

    def test_with_existing_facilities(self):
        w = {"h3_1": 100, "h3_2": 200, "h3_3": 150}
        IJ = {
            "h3_1": ["fac_existing", "fac_A"],
            "h3_2": ["fac_A", "fac_B"],
            "h3_3": ["fac_B"],
        }
        J_existing = ["fac_existing"]
        J_potential = ["fac_A", "fac_B"]

        results = solve_mclp_greedy(w, IJ, J_existing, J_potential, max_new_facilities=2)

        # Existing facility covers h3_1 (100), so initial coverage is 100
        # Adding fac_B gives h3_2 (200) + h3_3 (150) = 350 more
        # p=0 baseline reflects existing facility coverage: h3_1 = 100
        assert results[0]["p"] == 0
        assert results[0]["objective"] == 100

        # First greedy step adds fac_B: h3_2 (200) + h3_3 (150) = 350 more → total 450
        assert results[1]["objective"] == 100 + 350  # 450

    def test_no_improvement_possible(self):
        w = {"h3_1": 100}
        IJ = {"h3_1": ["fac_existing"]}
        J_existing = ["fac_existing"]
        J_potential = ["fac_A"]  # fac_A doesn't cover anything

        results = solve_mclp_greedy(w, IJ, J_existing, J_potential, max_new_facilities=5)

        # Only the p=0 baseline is returned; fac_A adds nothing
        assert len(results) == 1
        assert results[0]["p"] == 0
        assert results[0]["objective"] == 100  # existing facility already covers h3_1

    def test_empty_inputs(self):
        results = solve_mclp_greedy({}, {}, [], [], max_new_facilities=5)
        assert results == [
            {
                "p": 0,
                "objective": 0.0,
                "selected_facilities": [],
                "covered_h3": [],
            }
        ]

    def test_solve_mclp_greedy_already_fully_covered(self):
        w = {"h3_cell_1": 100.0, "h3_cell_2": 200.0, "h3_cell_3": 150.0}
        IJ = {
            "h3_cell_1": ["fac_existing"],
            "h3_cell_2": ["fac_existing"],
            "h3_cell_3": ["fac_existing"],
        }
        result = solve_mclp_greedy(
            w=w,
            IJ=IJ,
            J_existing=["fac_existing"],
            J_potential=[],           # no candidates → greedy loop exits immediately
            max_new_facilities=5,
        )
        # Sentinel guarantees at least one row even at 100% coverage
        assert len(result) > 0

        first = result[0]
        assert first["p"] == 0
        assert first["objective"] == 450.0    # full baseline coverage
        assert first["selected_facilities"] == ["fac_existing"]
        assert set(first["covered_h3"]) == {"h3_cell_1", "h3_cell_2", "h3_cell_3"}

class TestDeduplicateColumns:

    def test_no_duplicates(self):
        cols = ["a", "b", "c"]
        result = deduplicate_columns(cols)
        assert result == ["a", "b", "c"]

    def test_case_insensitive_duplicate(self):
        cols = ["Name", "name", "value"]
        result = deduplicate_columns(cols)
        assert result[0] == "Name"
        assert result[1] == "name_dup"
        assert result[2] == "value"

    def test_multiple_duplicates(self):
        cols = ["ID", "id", "Id"]
        result = deduplicate_columns(cols)
        assert result[0] == "ID"
        assert "dup" in result[1]
        assert "dup" in result[2]
        # All should be unique (case-insensitive)
        assert len(set(c.lower() for c in result)) == 3

    def test_empty_list(self):
        result = deduplicate_columns([])
        assert result == []


from shared.core import should_load_country_cache


class TestShouldLoadCountryCache:

    def test_reuses_when_present_and_not_forcing(self):
        assert should_load_country_cache(force=False, country_cache_exists=True) is True

    def test_forcing_never_reuses(self):
        # #6: a forced run must re-query, never read a possibly-stale country cache.
        assert should_load_country_cache(force=True, country_cache_exists=True) is False

    def test_no_reuse_when_absent(self):
        assert should_load_country_cache(force=False, country_cache_exists=False) is False
