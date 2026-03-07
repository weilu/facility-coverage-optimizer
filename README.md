# Health Facility Location Optimizer

Optimizes placement of new health facilities using Maximum Covering Location Problem (MCLP) to maximize population access within a travel distance threshold. Uses a greedy approximation algorithm for scalability.

See [docs/optimization_approach.md](docs/optimization_approach.md) for details on the algorithm choice.

## Scripts

| Script | Purpose |
|--------|---------|
| `01_extract.py` | Downloads and prepares source data (GADM boundaries, WorldPop raster, health facilities) |
| `02_transform.py` | Computes coverage, runs optimization, visualizes results |

## Configuration

Edit the configuration section at the top of each script:

**01_extract.py**
```python
COUNTRY = "Zambia"        # Country name
ADM_LEVEL1 = None         # Province filter (None = whole country)
ADM_LEVEL2 = None         # District filter
POPULATION_YEAR = 2025
```

**02_transform.py**
```python
DISTANCE_METERS = 10000       # Catchment radius
TARGET_NEW_FACILITIES = 7     # Number of new facilities to place
POTENTIAL_TYPE = "grid"       # "grid" or "kmeans"
GRID_SPACING = 0.03           # Grid density (degrees)
FORCE_RECOMPUTE = False       # Set True to invalidate cache
```

## Usage

Run in order on Databricks:
1. `01_extract.py` - extracts data to Unity Catalog tables
2. `02_transform.py` - runs optimization and displays results

## Caching

Expensive operations are cached to UC tables. Subsequent runs load from cache automatically.

| Cached Table | Contents |
|--------------|----------|
| `population_aoi_*` | Population filtered to AOI |
| `facilities_h3_*` | Existing facilities with coverage |
| `facilities_coverage_*` | Facility-population coverage pairs |
| `potential_locations_*` | Candidate locations with coverage |
| `potential_coverage_*` | Candidate-population coverage pairs |

Set `FORCE_RECOMPUTE = True` to recompute all cached results.

## Output

- Coverage statistics (current vs. maximum possible)
- Pareto frontier chart (facilities vs. coverage %)
- Interactive maps showing existing facilities, new placements, and population coverage
