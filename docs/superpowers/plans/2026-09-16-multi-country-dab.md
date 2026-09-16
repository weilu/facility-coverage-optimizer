# Multi-Country DAB Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the extract+transform pipeline as a Databricks Asset Bundle that runs any country by ISO3 alone and batch-runs many countries in one trigger.

**Architecture:** Centralize run-varying config as widgets/job-parameters in `shared/settings.py` (country derived from ISO3 via pycountry); standardize all table names on ISO3; merge the two jobs into one parameterized `pipeline` job gated by the test task, with a `batch` job fanning out over countries via `for_each` + `run_job`.

**Tech Stack:** Python, Databricks (notebooks as `.py` with `# Databricks notebook source`), Databricks Asset Bundles (`databricks.yml`), pytest, pycountry, PySpark, geopandas.

**Spec:** `docs/superpowers/specs/2026-09-16-multi-country-dab-design.md`

## Global Constraints

- Unity Catalog catalog is `prd_mega`. Canonical data schema is `sgpbpi163` (prod); dev target uses `pim`.
- Tests must never write to `prd_mega` (dev `pim` / prod `sgpbpi163`). The `tests/conftest.py` UC-write guard enforces this; do not weaken it.
- Notebook files (`extract/*.py`, `transform/*.py`) use the dual pattern: `# MAGIC %run "..."` cells for Databricks plus a `if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):` block of real imports for local/CI. Keep both in sync when adding imports.
- Importable logic lives in `shared/` modules (`core.py` = pure/no-Spark; `transform_ops.py` = Spark/geo transforms; `env.py` = storage/env). Do not put testable logic in the digit-prefixed notebooks.
- `H3_RESOLUTION` must be a single shared value (extraction and transform must match).
- Do NOT rename `health_facilities_{iso3}_osm` or `lgu_accessibility_results_{iso3}_*` — the dashboard depends on both.
- Comments explain WHY, never WHAT. Preserve existing comments when moving code.
- The test-framework slice (spec section 7) is already DONE (commit `c155fe2`); this plan covers spec sections 1–6.

---

### Task 1: Country/population parameterization in `shared/settings.py`

Derive country identity from a single `COUNTRY_ISO3` widget. ISO2 comes from pycountry (standard codes — WB's are non-standard for PSE/SRB/YEM). Because table names standardize on ISO3 (Task 3), the country display name is used only for logging.

**Files:**
- Modify: `shared/settings.py`
- Modify: `requirements.txt`, `pyproject.toml` (add `pycountry`)
- Test: `tests/test_settings.py` (create)

**Interfaces:**
- Produces: `shared.settings.resolve_iso2(iso3: str) -> str`, `shared.settings.resolve_country_name(iso3: str) -> str`, and module constants `ISO_3: str`, `ISO_2: str`, `COUNTRY: str`, `POPULATION_YEAR: int`. Consumed by `extract/config.py` and `transform/config.py` (already import `COUNTRY`, `ISO_3`, `POPULATION_YEAR`; `ISO_2` is imported by `extract/04_facilities.py`).

- [ ] **Step 1: Add pycountry to dependencies**

In `requirements.txt` add a line:
```
pycountry
```
In `pyproject.toml`, under `[project]` `dependencies`, add:
```
    "pycountry>=22.0.0",
```

- [ ] **Step 2: Write failing tests** — `tests/test_settings.py`

```python
"""Tests for country/ISO derivation and widget fallbacks in shared/settings.py."""

import pytest

from shared.settings import resolve_iso2, resolve_country_name

# The 36 countries already present in prd_mega.sgpbpi163.
ALL_ISO3 = [
    "AFG", "BEN", "BFA", "BGD", "CIV", "CMR", "DJI", "ETH", "GAB", "GIN",
    "GMB", "GNQ", "HTI", "IND", "JPN", "KHM", "LAO", "MLI", "MWI", "NER",
    "NGA", "NPL", "PAK", "PSE", "ROU", "SDN", "SEN", "SOM", "SRB", "SSD",
    "SYR", "TCD", "TGO", "UZB", "YEM", "ZMB",
]


class TestResolveIso2:
    def test_all_36_resolve(self):
        for iso3 in ALL_ISO3:
            assert len(resolve_iso2(iso3)) == 2

    @pytest.mark.parametrize("iso3,iso2", [("PSE", "PS"), ("SRB", "RS"), ("YEM", "YE")])
    def test_standard_codes_for_wb_oddities(self, iso3, iso2):
        # WB admin0 uses non-standard alpha-2 (GZ/YF/RY); OSM needs the real codes.
        assert resolve_iso2(iso3) == iso2

    def test_case_insensitive(self):
        assert resolve_iso2("lao") == "LA"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_iso2("ZZZ")


class TestResolveCountryName:
    def test_returns_nonempty_for_all(self):
        for iso3 in ALL_ISO3:
            assert resolve_country_name(iso3)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_settings.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_iso2'`.

- [ ] **Step 4: Rewrite `shared/settings.py`**

```python
# Databricks notebook source
# Shared application settings
# Imported by extract/config.py and transform/config.py

import pycountry

# Unity Catalog
UC_CATALOG = "prd_mega"
UC_SCHEMA_DEFAULT = "sgpbpi163"


def _get_widget(name: str, default: str) -> str:
    """Read a Databricks widget, falling back to a default off-cluster."""
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default


def resolve_iso2(iso3: str) -> str:
    """ISO 3166-1 alpha-2 for an alpha-3 code (standard codes, unlike WB admin0)."""
    country = pycountry.countries.get(alpha_3=iso3.upper())
    if country is None:
        raise ValueError(f"Unknown ISO3 country code: {iso3!r}")
    return country.alpha_2


def resolve_country_name(iso3: str) -> str:
    """Human-readable country name (display/logging only; table names use ISO3)."""
    country = pycountry.countries.get(alpha_3=iso3.upper())
    if country is None:
        raise ValueError(f"Unknown ISO3 country code: {iso3!r}")
    return getattr(country, "common_name", None) or country.name


UC_SCHEMA = _get_widget("UC_SCHEMA", UC_SCHEMA_DEFAULT)

# Country settings — derived from a single COUNTRY_ISO3 widget (default Laos)
ISO_3 = _get_widget("COUNTRY_ISO3", "LAO").upper()
ISO_2 = resolve_iso2(ISO_3)
COUNTRY = resolve_country_name(ISO_3)
POPULATION_YEAR = int(_get_widget("POPULATION_YEAR", "2025"))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_settings.py -v`
Expected: PASS (all).

- [ ] **Step 6: Run the full CI-safe suite (no regressions)**

Run: `pytest -m "not databricks" -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add shared/settings.py requirements.txt pyproject.toml tests/test_settings.py
git commit -m "feat: derive country/ISO2 from COUNTRY_ISO3 widget via pycountry"
```

---

### Task 2: Consolidate `FORCE_RECOMPUTE`, `INCLUDE_ADM_LEVEL0`, `H3_RESOLUTION` into `shared/settings.py`

These are duplicated across the two config files (and an H3 literal in `02_population`). Move them to settings; the config notebooks import them. `FORCE_RECOMPUTE`/`INCLUDE_ADM_LEVEL0` become widgets; `H3_RESOLUTION` stays a constant.

**Files:**
- Modify: `shared/settings.py`
- Modify: `extract/config.py` (remove `FORCE_RECOMPUTE`, `INCLUDE_ADM_LEVEL0`; import from settings)
- Modify: `transform/config.py` (remove `FORCE_RECOMPUTE`, `INCLUDE_ADM_LEVEL0`, `H3_RESOLUTION`; import from settings)
- Modify: `extract/02_population.py` (use shared `H3_RESOLUTION` instead of the `h3_resolution: int = 8` default)
- Test: `tests/test_settings.py` (extend)

**Interfaces:**
- Consumes: Task 1's `_get_widget`.
- Produces: `shared.settings.FORCE_RECOMPUTE: bool`, `shared.settings.INCLUDE_ADM_LEVEL0: bool`, `shared.settings.H3_RESOLUTION: int`, and `shared.settings._get_bool_widget(name, default)`.

- [ ] **Step 1: Write failing tests** — append to `tests/test_settings.py`

```python
class TestRunControlDefaults:
    def test_force_recompute_defaults_false(self):
        from shared.settings import FORCE_RECOMPUTE
        assert FORCE_RECOMPUTE is False

    def test_include_adm_level0_defaults_true(self):
        from shared.settings import INCLUDE_ADM_LEVEL0
        assert INCLUDE_ADM_LEVEL0 is True

    def test_h3_resolution_is_shared_constant(self):
        from shared.settings import H3_RESOLUTION
        assert H3_RESOLUTION == 8


class TestBoolWidget:
    @pytest.mark.parametrize("raw,expected", [
        ("true", True), ("True", True), ("1", True), ("yes", True),
        ("false", False), ("no", False), ("", False),
    ])
    def test_parses_truthy_strings(self, raw, expected):
        from shared.settings import _parse_bool
        assert _parse_bool(raw) is expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_settings.py -k "RunControl or BoolWidget" -v`
Expected: FAIL (ImportError for `FORCE_RECOMPUTE` / `_parse_bool`).

- [ ] **Step 3: Add to `shared/settings.py`**

After `resolve_country_name`, add:
```python
def _parse_bool(raw: str) -> bool:
    return str(raw).strip().lower() in ("true", "1", "yes")


def _get_bool_widget(name: str, default: bool) -> bool:
    try:
        return _parse_bool(dbutils.widgets.get(name))
    except Exception:
        return default
```
After `POPULATION_YEAR`, add:
```python
# Run-control (job parameters), consolidated from extract/config.py + transform/config.py
FORCE_RECOMPUTE = _get_bool_widget("FORCE_RECOMPUTE", False)
INCLUDE_ADM_LEVEL0 = _get_bool_widget("INCLUDE_ADM_LEVEL0", True)

# H3 resolution — single source; extraction and transform must match.
H3_RESOLUTION = 8
```

- [ ] **Step 4: Rewire `extract/config.py`**

Delete these lines (with their comments):
```python
# Set to True to recompute cached results even if tables exist
FORCE_RECOMPUTE = False

# Include country-level (ADM0) processing
INCLUDE_ADM_LEVEL0 = True
```
In the `%run`-paired local import block, extend the `from shared.settings import (...)` to include `FORCE_RECOMPUTE, INCLUDE_ADM_LEVEL0`. (These flow via `%run "../shared/settings"` on Databricks — confirm that cell exists; if `extract/config.py` imports settings through `%run "../shared/core"`/`./config` chains, add a `# MAGIC %run "../shared/settings"` cell to match the local import.)

- [ ] **Step 5: Rewire `transform/config.py`**

Delete `INCLUDE_ADM_LEVEL0 = True` (line ~88), `FORCE_RECOMPUTE = False` (line ~108), and `H3_RESOLUTION = 8  # Must match extraction resolution` (line ~105). Add `FORCE_RECOMPUTE, INCLUDE_ADM_LEVEL0, H3_RESOLUTION` to the local `from shared.settings import (...)` block (and ensure the matching `# MAGIC %run "../shared/settings"` cell exists). Keep `H3_EDGE_LENGTH_M` and `get_k_rings` imports from `shared.core` unchanged.

- [ ] **Step 6: Rewire `extract/02_population.py`**

Change the H3 default so it uses the shared value. In the local import block add `from shared.settings import H3_RESOLUTION`, and change the function signature `def ...(..., h3_resolution: int = 8, ...)` to `h3_resolution: int = H3_RESOLUTION` (or pass `H3_RESOLUTION` explicitly at the call site if the function is a helper). Verify with `grep -n "h3_resolution" extract/02_population.py` that no bare `8` literal remains for resolution.

- [ ] **Step 7: Run tests + compile checks**

Run:
```bash
pytest tests/test_settings.py -q
python -m py_compile extract/config.py transform/config.py extract/02_population.py shared/settings.py
pytest -m "not databricks" -q
```
Expected: tests pass; all compile.

- [ ] **Step 8: Commit**

```bash
git add shared/settings.py extract/config.py transform/config.py extract/02_population.py tests/test_settings.py
git commit -m "refactor: consolidate FORCE_RECOMPUTE/INCLUDE_ADM_LEVEL0/H3_RESOLUTION into settings"
```

---

### Task 3: Standardize LGU table naming on ISO3

Every table uses ISO3 except the LGU table, which uses the country name. Switch it to ISO3 in `shared/core.py` and the two config consumers. Update the existing `test_core.py` naming assertions.

**Files:**
- Modify: `shared/core.py` (4 `lgu` entries in `get_extract_table_names` and `get_transform_table_names`)
- Modify: `extract/config.py` (`COUNTRY_LGU_TABLE`)
- Modify: `transform/config.py` (`_get_adm_level1_names_from_uc`)
- Test: `tests/test_core.py` (update LGU assertions)

**Interfaces:**
- Produces: `get_extract_table_names(...)["lgu"] == f"{catalog}.{schema}.wb_boundaries_lgu_{iso3.lower()}"` (and province suffix variant); same for `get_transform_table_names`.

- [ ] **Step 1: Update tests first** — in `tests/test_core.py`

Find the LGU assertions in `TestGetExtractTableNames` / `TestGetTransformTableNames` (they currently expect `wb_boundaries_lgu_zambia`). Change the expected value to ISO3-based:
```python
        assert result["lgu"] == "prd.sgp.wb_boundaries_lgu_zmb"
```
(Use `iso3="ZMB"` in the call if not already; match the province-suffix variant to `wb_boundaries_lgu_zmb_northern_province` if such an assertion exists.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core.py -k "TableNames" -v`
Expected: FAIL (still produces `..._zambia`).

- [ ] **Step 3: Change `shared/core.py`**

In `get_extract_table_names`, both branches, change the `lgu` value from:
```python
            "lgu": f"{catalog}.{schema}.wb_boundaries_lgu_{_sanitize_adm_name(country)}{adm_suffix}",
```
```python
            "lgu": f"{catalog}.{schema}.wb_boundaries_lgu_{_sanitize_adm_name(country)}",
```
to (respectively, keeping/removing `adm_suffix` as in the original):
```python
            "lgu": f"{catalog}.{schema}.wb_boundaries_lgu_{iso3.lower()}{adm_suffix}",
```
```python
            "lgu": f"{catalog}.{schema}.wb_boundaries_lgu_{iso3.lower()}",
```
Apply the identical change to the two `lgu` entries in `get_transform_table_names`. Leave `_sanitize_adm_name` in place for the `adm_suffix` (province names). The `country` parameter becomes unused for naming — keep the signature (other callers/tests pass it) and add a one-line comment: `# country retained for signature compatibility; table names use iso3`.

- [ ] **Step 4: Change `extract/config.py`**

```python
COUNTRY_LGU_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_lgu_{ISO_3.lower()}"
```
Remove the now-unused `_sanitize_adm_name` import from `extract/config.py` **only if** nothing else in the file uses it (grep first: `grep -n "_sanitize_adm_name" extract/config.py`).

- [ ] **Step 5: Change `transform/config.py`**

In `_get_adm_level1_names_from_uc`:
```python
    lgu_table = f"{UC_CATALOG}.{UC_SCHEMA}.wb_boundaries_lgu_{COUNTRY_ISO3.lower()}"
```
(`COUNTRY_ISO3` is the local alias already imported for `ISO_3`.)

- [ ] **Step 6: Run tests + compile**

Run:
```bash
pytest tests/test_core.py -q
python -m py_compile shared/core.py extract/config.py transform/config.py
pytest -m "not databricks" -q
```
Expected: pass; compile OK.

- [ ] **Step 7: Commit**

```bash
git add shared/core.py extract/config.py transform/config.py tests/test_core.py
git commit -m "refactor: standardize LGU table naming on ISO3"
```

---

### Task 4: Fix the dead WB boundaries URL

The `ddh-published-v2/.../5/...` base URL 404s; the working path is `ddh-published/0038272/DR0095369/...`.

**Files:**
- Modify: `extract/config.py`

- [ ] **Step 1: Change the base URL**

Replace:
```python
WB_BOUNDARIES_BASE_URL = "https://datacatalogfiles.worldbank.org/ddh-published-v2/0038272/5/DR0095369/World%20Bank%20Official%20Boundaries%20(GeoJSON)"
```
with:
```python
WB_BOUNDARIES_BASE_URL = "https://datacatalogfiles.worldbank.org/ddh-published/0038272/DR0095369/World%20Bank%20Official%20Boundaries%20(GeoJSON)"
```

- [ ] **Step 2: Verify the three derived URLs resolve (HEAD 200)**

Run:
```bash
python - <<'PY'
import urllib.request
base = "https://datacatalogfiles.worldbank.org/ddh-published/0038272/DR0095369/World%20Bank%20Official%20Boundaries%20(GeoJSON)"
for lvl in (0, 1, 2):
    url = f"{base}/World%20Bank%20Official%20Boundaries%20-%20Admin%20{lvl}.geojson"
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        print(lvl, r.status)
PY
```
Expected: `0 200`, `1 200`, `2 200`.

- [ ] **Step 3: Commit**

```bash
git add extract/config.py
git commit -m "fix: point WB boundaries download at the live DDH URL"
```

---

### Task 5: Isolate `VOLUME_DIR` per schema via a `uc_volume` widget

`VOLUME_DIR` hardcodes the `sgpbpi163` volume, so dev/prod share caches. Expose the volume path segment as a widget (default preserves current behavior).

**Files:**
- Modify: `shared/settings.py` (add `UC_VOLUME`)
- Modify: `extract/config.py` (`VOLUME_DIR` uses it)
- Test: `tests/test_settings.py` (extend)

**Interfaces:**
- Produces: `shared.settings.UC_VOLUME: str` (default `"sgpbpi163/vgpbpi163"`). Consumed by `extract/config.py`'s `VOLUME_DIR`.

- [ ] **Step 1: Write failing test** — append to `tests/test_settings.py`

```python
class TestUcVolume:
    def test_default_preserves_current_volume(self):
        from shared.settings import UC_VOLUME
        assert UC_VOLUME == "sgpbpi163/vgpbpi163"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_settings.py -k UcVolume -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Add to `shared/settings.py`**

After `UC_SCHEMA`:
```python
# Volume path segment "schema/volume" for file caches (worldpop rasters, WB
# geojson, facilities input). Not derivable from UC_SCHEMA (volume name differs),
# so it is its own widget; default preserves existing cached data.
UC_VOLUME = _get_widget("UC_VOLUME", "sgpbpi163/vgpbpi163")
```

- [ ] **Step 4: Change `extract/config.py`**

Replace:
```python
VOLUME_DIR = f"/Volumes/{UC_CATALOG}/sgpbpi163/vgpbpi163"
```
with:
```python
VOLUME_DIR = f"/Volumes/{UC_CATALOG}/{UC_VOLUME}"
```
Add `UC_VOLUME` to the `from shared.settings import (...)` local block in `extract/config.py`.

- [ ] **Step 5: Run tests + compile**

Run:
```bash
pytest tests/test_settings.py -q
python -m py_compile extract/config.py shared/settings.py
```
Expected: pass; compile OK.

- [ ] **Step 6: Commit**

```bash
git add shared/settings.py extract/config.py tests/test_settings.py
git commit -m "refactor: make VOLUME_DIR schema-configurable via UC_VOLUME widget"
```

---

### Task 6: Review-item fixes #5/#6 in facility caching (extract a testable helper)

Fix #5 (function reads the module-global `adm_level1`) and #6 (`FORCE_RECOMPUTE` ignored for provinces) by extracting the cache decision into a pure, tested helper.

**Files:**
- Modify: `shared/core.py` (add `should_load_country_cache`)
- Modify: `extract/04_facilities.py` (use it; drop the `adm_level1` global reference)
- Test: `tests/test_core.py` (add helper tests)

**Interfaces:**
- Produces: `shared.core.should_load_country_cache(force: bool, country_cache_exists: bool) -> bool`.

- [ ] **Step 1: Write failing tests** — append to `tests/test_core.py`

```python
from shared.core import should_load_country_cache


class TestShouldLoadCountryCache:
    def test_reuses_when_present_and_not_forcing(self):
        assert should_load_country_cache(force=False, country_cache_exists=True) is True

    def test_forcing_never_reuses(self):
        # #6: a forced run must re-query, never read a possibly-stale country cache.
        assert should_load_country_cache(force=True, country_cache_exists=True) is False

    def test_no_reuse_when_absent(self):
        assert should_load_country_cache(force=False, country_cache_exists=False) is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_core.py -k ShouldLoadCountryCache -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Add the helper to `shared/core.py`**

```python
def should_load_country_cache(force: bool, country_cache_exists: bool) -> bool:
    """Whether to reuse the cached country-level OSM extract instead of re-querying.

    A forced run must never reuse a possibly-stale cache (#6); otherwise reuse it
    when it exists. Applies to both country and province passes — the country pass
    builds the cache in the same run, so there is no separate branch (#5).
    """
    return (not force) and country_cache_exists
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_core.py -k ShouldLoadCountryCache -v`
Expected: PASS.

- [ ] **Step 5: Rewire `extract/04_facilities.py`**

Add to the local import block: `from shared.core import should_load_country_cache` (and the matching `# MAGIC %run "../shared/core"` cell if not already present). Replace the branch:
```python
    # Perform country extraction once, province data can be cliped from country data
    if adm_level1 == None:
        extract_country_cache = force
    else:
        print(f"Switching to country level cache for processing {adm_level1}")
        # Always make sure country (adm_level=None) is run first in loop
        extract_country_cache = False

    # --- Country-level raw OSM cache (pre-boundary-filter) ---
    if country_raw_table and not extract_country_cache and table_exists(country_raw_table):
        print(f"Loading cached country-level OSM data from: {country_raw_table}")
```
with (note the corrected comment typo "cliped" → "clipped", and use of the `adm_level_name` param instead of the `adm_level1` global):
```python
    # Perform country extraction once; province data can be clipped from country data.
    # --- Country-level raw OSM cache (pre-boundary-filter) ---
    country_cache_exists = bool(country_raw_table) and table_exists(country_raw_table)
    if should_load_country_cache(force, country_cache_exists):
        print(f"Loading cached country-level OSM data from: {country_raw_table}")
```
Verify no other reference to `adm_level1` remains inside the function body: `grep -n "adm_level1" extract/04_facilities.py` should only show the module-level loop (outside the function) and the call-site.

- [ ] **Step 6: Run tests + compile**

Run:
```bash
pytest tests/test_core.py -q
python -m py_compile extract/04_facilities.py shared/core.py
pytest -m "not databricks" -q
```
Expected: pass; compile OK.

- [ ] **Step 7: Commit**

```bash
git add shared/core.py extract/04_facilities.py tests/test_core.py
git commit -m "fix: honor FORCE_RECOMPUTE for provinces and drop adm_level1 global (#5/#6)"
```

---

### Task 7: DAB restructure — merged `pipeline` job, `batch` fan-out, targets, delete runner

Merge the two jobs into one parameterized `pipeline` job gated by `run_tests`; add a `batch` job that `for_each`-runs the pipeline per country; add a `prod` target; delete `pipeline_runner.py`.

**Files:**
- Modify: `databricks.yml`
- Delete: `pipeline_runner.py`

**Interfaces:**
- Consumes: job parameters read by `shared/settings.py` widgets — `COUNTRY_ISO3`, `UC_SCHEMA`, `POPULATION_YEAR`, `FORCE_RECOMPUTE`, `INCLUDE_ADM_LEVEL0`, `UC_VOLUME`.

- [ ] **Step 1: Rewrite `databricks.yml`**

```yaml
bundle:
  name: health-facility-optimizer

variables:
  workspace_root:
    description: "Workspace root path for notebooks"
  uc_schema:
    description: "Unity Catalog schema (passed to notebooks via dbutils.widgets)"
  uc_volume:
    description: "Volume path segment 'schema/volume' for file caches"
    default: "sgpbpi163/vgpbpi163"
  cluster_id:
    description: "Existing cluster ID (find via: databricks clusters list)"
  countries:
    description: "JSON array of ISO3 codes for the batch job"
    default: '["AFG","BEN","BFA","BGD","CIV","CMR","DJI","ETH","GAB","GIN","GMB","GNQ","HTI","IND","JPN","KHM","LAO","MLI","MWI","NER","NGA","NPL","PAK","PSE","ROU","SDN","SEN","SOM","SRB","SSD","SYR","TCD","TGO","UZB","YEM","ZMB"]'

targets:
  dev-wei:
    workspace:
      profile: adb-6102124407836814
    variables:
      workspace_root: /Workspace/Users/wlu4@worldbank.org/facility-coverage-optimizer
      uc_schema: pim
      cluster_id: 0212-012600-43y69ksy  # ITSDA_DAP_TEAM_gpbpinfrastructureanalysispimpam
  prod:
    workspace:
      profile: adb-6102124407836814
    variables:
      workspace_root: /Workspace/Users/wlu4@worldbank.org/facility-coverage-optimizer
      uc_schema: sgpbpi163
      cluster_id: 0212-012600-43y69ksy  # ITSDA_DAP_TEAM_gpbpinfrastructureanalysispimpam

resources:
  jobs:
    pipeline:
      name: "HFO - Pipeline (single country)"
      parameters:
        - name: COUNTRY_ISO3
          default: LAO
        - name: UC_SCHEMA
          default: ${var.uc_schema}
        - name: UC_VOLUME
          default: ${var.uc_volume}
        - name: POPULATION_YEAR
          default: "2025"
        - name: FORCE_RECOMPUTE
          default: "false"
        - name: INCLUDE_ADM_LEVEL0
          default: "true"
      tasks:
        - task_key: run_tests
          existing_cluster_id: ${var.cluster_id}
          notebook_task:
            notebook_path: ${var.workspace_root}/tests/run_tests

        - task_key: download_worldpop
          existing_cluster_id: ${var.cluster_id}
          depends_on: [{ task_key: run_tests }]
          notebook_task:
            notebook_path: ${var.workspace_root}/extract/01a_download_worldpop
            base_parameters: &pipeline_params
              COUNTRY_ISO3: "{{job.parameters.COUNTRY_ISO3}}"
              UC_SCHEMA: "{{job.parameters.UC_SCHEMA}}"
              UC_VOLUME: "{{job.parameters.UC_VOLUME}}"
              POPULATION_YEAR: "{{job.parameters.POPULATION_YEAR}}"
              FORCE_RECOMPUTE: "{{job.parameters.FORCE_RECOMPUTE}}"
              INCLUDE_ADM_LEVEL0: "{{job.parameters.INCLUDE_ADM_LEVEL0}}"

        - task_key: download_wb
          existing_cluster_id: ${var.cluster_id}
          depends_on: [{ task_key: run_tests }]
          notebook_task:
            notebook_path: ${var.workspace_root}/extract/01b_download_wb
            base_parameters: *pipeline_params

        - task_key: population
          existing_cluster_id: ${var.cluster_id}
          depends_on: [{ task_key: download_worldpop }]
          notebook_task:
            notebook_path: ${var.workspace_root}/extract/02_population
            base_parameters: *pipeline_params

        - task_key: boundaries
          existing_cluster_id: ${var.cluster_id}
          depends_on: [{ task_key: download_wb }]
          notebook_task:
            notebook_path: ${var.workspace_root}/extract/03_boundaries
            base_parameters: *pipeline_params

        - task_key: facilities
          existing_cluster_id: ${var.cluster_id}
          depends_on: [{ task_key: boundaries }]
          notebook_task:
            notebook_path: ${var.workspace_root}/extract/04_facilities
            base_parameters: *pipeline_params

        - task_key: prepare
          existing_cluster_id: ${var.cluster_id}
          depends_on: [{ task_key: facilities }, { task_key: population }]
          notebook_task:
            notebook_path: ${var.workspace_root}/transform/01_prepare
            base_parameters: *pipeline_params

        - task_key: coverage
          existing_cluster_id: ${var.cluster_id}
          depends_on: [{ task_key: prepare }]
          notebook_task:
            notebook_path: ${var.workspace_root}/transform/02_coverage
            base_parameters: *pipeline_params

        - task_key: optimize
          existing_cluster_id: ${var.cluster_id}
          depends_on: [{ task_key: coverage }]
          notebook_task:
            notebook_path: ${var.workspace_root}/transform/03_optimize
            base_parameters: *pipeline_params

        - task_key: visualize
          existing_cluster_id: ${var.cluster_id}
          depends_on: [{ task_key: optimize }]
          notebook_task:
            notebook_path: ${var.workspace_root}/transform/04_visualize
            base_parameters:
              COUNTRY_ISO3: "{{job.parameters.COUNTRY_ISO3}}"
              UC_SCHEMA: "{{job.parameters.UC_SCHEMA}}"
              UC_VOLUME: "{{job.parameters.UC_VOLUME}}"
              POPULATION_YEAR: "{{job.parameters.POPULATION_YEAR}}"
              FORCE_RECOMPUTE: "{{job.parameters.FORCE_RECOMPUTE}}"
              INCLUDE_ADM_LEVEL0: "{{job.parameters.INCLUDE_ADM_LEVEL0}}"
              ENABLE_VISUALIZATION: "true"
              PIPELINE_START_EPOCH_MS: "{{job.start_time.epoch_ms}}"

    batch:
      name: "HFO - Batch (many countries)"
      parameters:
        - name: countries
          default: ${var.countries}
      tasks:
        - task_key: run_countries
          for_each_task:
            inputs: "{{job.parameters.countries}}"
            concurrency: 4
            task:
              task_key: pipeline_for_country
              run_job_task:
                job_id: ${resources.jobs.pipeline.id}
                job_parameters:
                  COUNTRY_ISO3: "{{input}}"
                  UC_SCHEMA: ${var.uc_schema}
                  UC_VOLUME: ${var.uc_volume}
```

- [ ] **Step 2: Delete the runner**

```bash
git rm pipeline_runner.py
```

- [ ] **Step 3: Validate the bundle for both targets**

Run:
```bash
databricks bundle validate -t dev-wei
databricks bundle validate -t prod
```
Expected: both report "Validation OK" (no schema errors). If `for_each_task`/`run_job_task` field names are rejected by the installed CLI version, consult `databricks bundle schema` and adjust field names, then re-run.

- [ ] **Step 4: Commit**

```bash
git add databricks.yml
git commit -m "feat: merge pipeline into one parameterized job with batch fan-out; add prod target; drop runner"
```

---

## Self-Review

**Spec coverage:**
- §1 country/population parameterization → Task 1. ✅
- §1 FORCE_RECOMPUTE/INCLUDE_ADM_LEVEL0 widgets, H3_RESOLUTION consolidation → Task 2. ✅
- §1 VOLUME_DIR isolation → Task 5. ✅
- §2 LGU naming → Task 3. ✅
- §3 WB URL fix → Task 4. ✅
- §4 DAB restructure (merge, batch, targets, delete runner, wire run_tests) → Task 7. ✅
- §5 review polish #5/#6 → Task 6. ✅
- §6 pycountry dep → Task 1 Step 1. ✅
- §7 test framework → already DONE (out of this plan's scope). ✅

**Placeholder scan:** No TBD/TODO; every code step has concrete content. Task 7's fallback ("consult `databricks bundle schema`") is a real recovery instruction, not a placeholder.

**Type consistency:** `resolve_iso2`/`resolve_country_name`/`_parse_bool`/`_get_bool_widget`/`UC_VOLUME`/`should_load_country_cache(force, country_cache_exists)` used consistently across tasks and match the settings/core producers. `get_*_table_names` keep their existing signatures (Task 3 keeps `country` param).

**Open verification notes for the executor:**
- Tasks 2/5/6 assume each notebook that imports a new symbol also has (or gains) the matching `# MAGIC %run "../shared/<module>"` cell. Confirm per file; add the cell if missing so Databricks resolves the symbol.
- Task 7's `for_each_task` + `run_job_task` YAML must be validated against the installed Databricks CLI (v1.1.0) schema; adjust field spelling if `bundle validate` complains.
