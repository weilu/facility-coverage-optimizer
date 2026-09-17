# Health Facility Location Optimizer

Optimizes placement of new health facilities using Maximum Covering Location Problem (MCLP) to maximize population access within a travel distance threshold. Uses a greedy approximation algorithm for scalability.

See [docs/optimization_approach.md](docs/optimization_approach.md) for details on the algorithm choice.

## Project Structure

```
├── shared/
│   ├── settings.py    # Shared constants (catalog, schema, country)
│   ├── core.py        # Pure functions (no Spark dependencies)
│   └── env.py         # Environment detection & storage backends
├── extract/
│   ├── config.py      # Extract pipeline configuration
│   ├── 01a_download_worldpop.py
│   ├── 01b_download_wb.py
│   ├── 02_population.py
│   ├── 03_boundaries.py
│   └── 04_facilities.py
├── transform/
│   ├── config.py      # Transform pipeline configuration
│   ├── 01_prepare.py
│   ├── 02_coverage.py
│   ├── 03_optimize.py
│   └── 04_visualize.py
└── tests/
    ├── conftest.py           # spark fixture + UC-write guard + databricks-marker skip
    ├── test_core.py          # Unit tests for pure functions
    ├── test_env.py           # Tests for environment/storage
    ├── test_settings.py      # Country/ISO derivation + widget parsing
    ├── test_transform_ops.py # Grid (unit) + H3 transforms (@pytest.mark.databricks)
    └── run_tests.py          # On-cluster test gate (Databricks notebook)
```

## Setup

### Local Development (uv)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates venv automatically)
uv sync --extra dev --extra spark
```

### Databricks

Upload the `shared/`, `extract/`, and `transform/` directories to your Databricks workspace. Run notebooks in order within each pipeline.

## Configuration

### Shared Settings (`shared/settings.py`)

```python
UC_CATALOG = "prd_mega"
UC_SCHEMA = os.getenv("UC_SCHEMA", "sgpbpi163")  # Override for testing
COUNTRY = "Zambia"
ISO_2 = "ZM"
ISO_3 = "ZMB"
POPULATION_YEAR = 2025
```

### Extract Pipeline (`extract/config.py`)

```python
INCLUDE_ADM_LEVEL0 = True        # Include country-level (ADM0) processing
ADM_LEVEL1_LIST = []             # [] = all provinces, or ["Northern", "Southern"]
FACILITIES_SOURCE = "osm"        # "osm" or "file"
FORCE_RECOMPUTE = False
```

### Transform Pipeline (`transform/config.py`)

```python
INCLUDE_ADM_LEVEL0 = True        # Include country-level (ADM0) processing
ADM_LEVEL1_LIST = []             # [] = all provinces, or ["Northern", "Southern"]
DISTANCES_METERS = [2000, 4000, 5000, 10000]  # Catchment radii to analyze
TARGET_NEW_FACILITIES = 50
POTENTIAL_TYPE = "grid"   # "grid" or "kmeans"
GRID_SPACING = 0.03
FORCE_RECOMPUTE = False
ENABLE_VISUALIZATION = True      # Set false to skip 04_visualize.py
```

The `ENABLE_VISUALIZATION` setting can also be overridden via Databricks job parameters.

## Usage

### Databricks

Run tasks in order:

**Extract Pipeline:**
1. `extract/01a_download_worldpop.py` - Download WorldPop raster
2. `extract/01b_download_wb.py` - Download World Bank boundaries
3. `extract/02_population.py` - Extract population to UC table
4. `extract/03_boundaries.py` - Extract province/LGU boundaries
5. `extract/04_facilities.py` - Extract health facilities

**Transform Pipeline:**
1. `transform/01_prepare.py` - Prepare data, generate potential locations
2. `transform/02_coverage.py` - Compute H3-based coverage
3. `transform/03_optimize.py` - Run optimization, compute per-LGU accessibility metrics
4. `transform/04_visualize.py` - Generate Pareto charts and coverage maps (optional)

### Testing with a Different Schema

To test without affecting production tables, set `UC_SCHEMA` environment variable:

**Databricks Job/Workflow:**
Task settings → Environment variables → Add `UC_SCHEMA=sgpbpi163_dev`

**Local:**
```bash
export UC_SCHEMA=sgpbpi163_dev
```

## Cached Tables

Expensive operations are cached to UC tables. Set `FORCE_RECOMPUTE = True` to regenerate.

| Table Pattern | Contents |
|---------------|----------|
| `population_{iso3}_{year}` | Country-level population |
| `wb_boundaries_{iso3}_*` | Province boundaries |
| `health_facilities_{iso3}_*` | Health facilities |
| `*_population_aoi_*` | Population filtered to AOI |
| `*_facilities_h3_*` | Facilities with H3 index |
| `*_coverage_*` | Coverage pairs |
| `lgu_accessibility_results_*` | Final optimization results |
| `base_dashboard_data_*` | Aggregated metadata for frontend data app |

## Output

- Per-LGU accessibility metrics at each optimization step
- Coverage statistics (baseline vs. optimized)
- Pareto frontier chart (facilities vs. coverage %)
- Results table consumed by [pimpam-dash](../pimpam-dash) visualization app

## Running Tests

```bash
# Unit tests (everything runnable off-cluster; databricks-marked H3 tests auto-skip)
uv run pytest tests/ -v

# Explicitly exclude the Databricks-only H3 tests (what CI runs)
uv run pytest -m "not databricks" -v

# With coverage
uv run pytest tests/ --cov=shared --cov=extract --cov=transform

# The @pytest.mark.databricks tests (real H3 SQL) run on a cluster via the
# tests/run_tests.py notebook, which is the pipeline job's first (gate) task.
#
# On-cluster gate: this cluster does not mount /Workspace files to the driver, so
# the gate installs the project wheel (which bundles tests/) from a UC Volume and
# runs `pytest --pyargs tests`. Publish/refresh the wheel whenever code changes:
#   DATABRICKS_PROFILE=adb-6102124407836814 scripts/publish_wheel.sh dev-wei
# CI (.github/workflows/publish-wheel.yml) does this on merge to main (needs
# DATABRICKS_HOST + DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET repo secrets).
```
