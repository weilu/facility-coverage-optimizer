# Databricks notebook source
# Pure functions that can be unit tested without Spark/Databricks dependencies
# These functions have no side effects and return the same output for the same input

import re
import numpy as np

# H3 resolution edge lengths in meters
H3_EDGE_LENGTH_M = {4: 22606, 5: 8544, 6: 3229, 7: 1220, 8: 461, 9: 174, 10: 66}


def get_k_rings(distance_meters: int, h3_resolution: int) -> int:
    """
    Calculate number of H3 rings needed to cover a given distance.

    Args:
        distance_meters: Target distance in meters
        h3_resolution: H3 resolution (4-10)

    Returns:
        Number of H3 rings (k value for h3_kring)
    """
    if h3_resolution not in H3_EDGE_LENGTH_M:
        raise ValueError(f"H3 resolution must be 4-10, got {h3_resolution}")
    return int(np.ceil(distance_meters / H3_EDGE_LENGTH_M[h3_resolution]))


def sanitize_col_name(name: str) -> str:
    """
    Converts an LGU display name into a Delta-safe column name.

    Rules applied (in order):
      1. Strip leading/trailing whitespace
      2. Replace any character that is not alphanumeric or underscore with underscore
      3. Collapse consecutive underscores to a single one
      4. Strip leading/trailing underscores
      5. Prefix with 'lgu_' so the name never starts with a digit

    Examples:
      "Kapiri Mposhi"  → "lgu_Kapiri_Mposhi"
      "Choma (East)"   → "lgu_Choma_East"
      "Lusaka"         → "lgu_Lusaka"
    """
    s = name.strip()
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return f"lgu_{s}"


def get_extract_table_names(
    catalog: str,
    schema: str,
    country: str,
    iso3: str,
    adm_level1: str | None,
    population_year: int,
) -> dict[str, str]:
    """
    Generate table names for extract pipeline.

    Args:
        catalog: UC catalog name
        schema: UC schema name
        country: Country name
        iso3: ISO 3-letter country code
        adm_level1: Optional admin level 1 region name
        population_year: Population data year

    Returns:
        Dictionary of table names for boundaries, population, facilities, lgu
    """
    if adm_level1 is not None:
        adm_suffix = f"_{adm_level1.lower().replace('-', '_').replace(' ', '_')}_province"
        return {
            "boundaries": f"{catalog}.{schema}.wb_boundaries_{iso3.lower()}{adm_suffix}",
            "population": f"{catalog}.{schema}.population_{iso3.lower()}_{population_year}{adm_suffix}",
            "facilities": f"{catalog}.{schema}.health_facilities_{iso3.lower()}_osm{adm_suffix}",
            "lgu": f"{catalog}.{schema}.wb_boundaries_lgu_{country.lower()}{adm_suffix}",
        }
    else:
        return {
            "boundaries": f"{catalog}.{schema}.wb_boundaries_{iso3.lower()}",
            "population": f"{catalog}.{schema}.population_{iso3.lower()}_{population_year}",
            "facilities": f"{catalog}.{schema}.health_facilities_{iso3.lower()}_osm",
            "lgu": f"{catalog}.{schema}.wb_boundaries_lgu_{country.lower()}",
        }


def get_transform_table_names(
    catalog: str,
    schema: str,
    country: str,
    iso3: str,
    adm_level1: str | None,
    population_year: int,
    distance_meters: int,
) -> dict[str, str]:
    """
    Generate table names for transform pipeline.

    Args:
        catalog: UC catalog name
        schema: UC schema name
        country: Country name
        iso3: ISO 3-letter country code
        adm_level1: Optional admin level 1 region name
        population_year: Population data year
        distance_meters: Distance threshold in meters

    Returns:
        Dictionary of table names for all transform outputs
    """
    distance_name = f"{int(distance_meters / 1000)}km"

    if adm_level1 is not None:
        adm_suffix = f"_{adm_level1.lower().replace('-', '_').replace(' ', '_')}_province"
        return {
            "boundaries": f"{catalog}.{schema}.wb_boundaries_{iso3.lower()}{adm_suffix}",
            "facilities": f"{catalog}.{schema}.health_facilities_{iso3.lower()}_osm{adm_suffix}",
            "population": f"{catalog}.{schema}.population_{iso3.lower()}_{population_year}",
            "population_aoi": f"{catalog}.{schema}.population_aoi_{iso3.lower()}_{population_year}{adm_suffix}_{distance_name}",
            "facilities_h3": f"{catalog}.{schema}.facilities_h3_{iso3.lower()}{adm_suffix}_{distance_name}",
            "facilities_coverage": f"{catalog}.{schema}.facilities_coverage_{iso3.lower()}{adm_suffix}_{distance_name}",
            "potential_locations": f"{catalog}.{schema}.potential_locations_{iso3.lower()}{adm_suffix}_{distance_name}",
            "potential_coverage": f"{catalog}.{schema}.potential_coverage_{iso3.lower()}{adm_suffix}_{distance_name}",
            "lgu": f"{catalog}.{schema}.wb_boundaries_lgu_{country.lower()}",
            "lgu_accessibility": f"{catalog}.{schema}.lgu_accessibility_results_{iso3.lower()}{adm_suffix}_{distance_name}",
        }
    else:
        return {
            "boundaries": f"{catalog}.{schema}.wb_boundaries_{iso3.lower()}",
            "facilities": f"{catalog}.{schema}.health_facilities_{iso3.lower()}_osm",
            "population": f"{catalog}.{schema}.population_{iso3.lower()}_{population_year}",
            "population_aoi": f"{catalog}.{schema}.population_aoi_{iso3.lower()}_{population_year}_{distance_name}",
            "facilities_h3": f"{catalog}.{schema}.facilities_h3_{iso3.lower()}_{distance_name}",
            "facilities_coverage": f"{catalog}.{schema}.facilities_coverage_{iso3.lower()}_{distance_name}",
            "potential_locations": f"{catalog}.{schema}.potential_locations_{iso3.lower()}_{distance_name}",
            "potential_coverage": f"{catalog}.{schema}.potential_coverage_{iso3.lower()}_{distance_name}",
            "lgu": f"{catalog}.{schema}.wb_boundaries_lgu_{country.lower()}",
            "lgu_accessibility": f"{catalog}.{schema}.lgu_accessibility_results_{iso3.lower()}_{distance_name}",
        }


def build_transform_combinations(
    adm_level1_list: list[str] | None,
    distances_meters: list[int],
) -> list[tuple[str | None, int]]:
    """
    Build list of (province, distance) combinations to process.

    Args:
        adm_level1_list: List of province names, empty list for all, None for country-level
        distances_meters: List of distances in meters

    Returns:
        List of (adm_level1, distance_meters) tuples
    """
    combinations = []
    provinces = adm_level1_list if adm_level1_list else [None]
    for adm_level1 in provinces:
        for distance_meters in distances_meters:
            combinations.append((adm_level1, distance_meters))
    return combinations


def solve_mclp_greedy(
    w: dict[str, float],
    IJ: dict[str, list[str]],
    J_existing: list[str],
    J_potential: list[str],
    max_new_facilities: int,
) -> list[dict]:
    """
    Greedy Maximum Covering Location Problem solver.

    This is a greedy approximation algorithm that iteratively selects
    the facility that provides maximum marginal population coverage.

    Args:
        w: Population weights by H3 cell {h3_index: population}
        IJ: Coverage mapping {h3_index: [facility_ids that cover it]}
        J_existing: List of existing facility IDs
        J_potential: List of potential new facility IDs
        max_new_facilities: Maximum number of new facilities to add

    Returns:
        List of results for p=1..max_new_facilities, each containing:
        - p: number of new facilities added
        - objective: total population covered
        - selected_facilities: list of all selected facility IDs
        - covered_h3: list of covered H3 cell IDs
    """
    # Build reverse index: facility -> set of H3 cells it covers
    facility_covers: dict[str, set[str]] = {}
    for h3_cell, fac_list in IJ.items():
        for fac in fac_list:
            if fac not in facility_covers:
                facility_covers[fac] = set()
            facility_covers[fac].add(h3_cell)

    # Initialize with existing facilities
    selected = set(J_existing)
    covered_h3: set[str] = set()
    for fac in J_existing:
        if fac in facility_covers:
            covered_h3.update(facility_covers[fac])

    current_coverage = sum(w.get(h3, 0) for h3 in covered_h3)

    results = [{
        "p": 0,
        "objective": 0.0,
        "selected_facilities": [],
        "covered_h3": [],
    }]
    
    candidates = set(J_potential) - selected

    for p in range(1, max_new_facilities + 1):
        best_fac = None
        best_gain = 0.0

        # Find facility with maximum marginal gain
        for fac in candidates:
            if fac not in facility_covers:
                continue
            new_cells = facility_covers[fac] - covered_h3
            gain = sum(w.get(h3, 0) for h3 in new_cells)
            if gain > best_gain:
                best_gain = gain
                best_fac = fac

        if best_fac is None or best_gain == 0:
            break

        # Add best facility
        selected.add(best_fac)
        candidates.remove(best_fac)
        covered_h3.update(facility_covers[best_fac])
        current_coverage += best_gain

        results.append({
            "p": p,
            "objective": current_coverage,
            "selected_facilities": list(selected),
            "covered_h3": list(covered_h3),
        })

    return results


def deduplicate_columns(columns: list[str]) -> list[str]:
    """
    Handle duplicate column names (case-insensitive) for Spark compatibility.

    Args:
        columns: List of column names

    Returns:
        List of deduplicated column names with _dup suffix for duplicates
    """
    cols_lower = [c.lower() for c in columns]
    seen: set[str] = set()
    new_cols = []

    for i, col in enumerate(columns):
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

    return new_cols
