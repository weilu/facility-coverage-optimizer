# Multi-Country DAB Packaging — Design

Date: 2026-09-16
Status: Proposed (awaiting review)

## Goal

Package the extract + transform pipeline as a Databricks Asset Bundle (DAB) that
can run any country by ISO3 alone, and batch-run many countries in one trigger.
Concretely:

1. Delete the ad-hoc `pipeline_runner.py` (superseded by the DAB).
2. Parameterize the pipeline by country so it is no longer hardcoded to Laos.
3. Support multi-country batch runs.
4. Chain extract → transform into a single run.
5. Add a `prod` target and address open review items from PR #4.

Canonical data schema: `prd_mega.sgpbpi163` (where 36 countries already exist).

## Background / investigation findings

- **`COUNTRY` / `ISO_2` / `ISO_3` / `POPULATION_YEAR` are hardcoded** in
  `shared/settings.py`. Only `UC_SCHEMA` is a widget. This is the blocker for
  multi-country runs.
- **The WB boundaries download URL in `extract/config.py` is dead (404).** It
  points at `ddh-published-v2/0038272/5/DR0095369/…` which now returns
  `BlobNotFound`. The working URL (used by the sibling `mega-indicators` repo) is
  `ddh-published/0038272/DR0095369/…` (verified 200, 164MB). The pipeline
  currently only survives on the stale cached copy in the volume.
- **The WB dataset's `ISO_A2` codes are non-standard for 3 of the 36 countries**
  and fetching fresh data does NOT fix them (verified against the live file):
  PSE=`GZ` (should be `PS`), SRB=`YF` (should be `RS`), YEM=`RY` (should be `YE`).
  OSM Overpass needs the real ISO 3166-1 alpha-2 codes, so WB `ISO_A2` cannot be
  used verbatim.
- **`pycountry` (installed, v26.2.16) resolves ISO3→ISO2 correctly for all 36**,
  including PSE/SRB/YEM. It is NOT yet declared in `requirements.txt`/`pyproject.toml`.
- **The LGU table is the only table named by country** (`wb_boundaries_lgu_{country}`);
  every other table uses ISO3. `extract_boundaries_lgu` already filters by
  `ISO_A3 == country_iso3` (`extract/03_boundaries.py:115`) — the country *name*
  is used purely to build the table name. WB `NAM_0` for LAO is
  "Lao People's Democratic Republic", which does not match the existing
  `wb_boundaries_lgu_laos` table (operator used the short name).

## Design decisions (approved)

- **Country input:** pass `COUNTRY_ISO3` only.
- **ISO2:** derive via `pycountry` (WB admin0 used for boundaries, not for codes).
- **LGU naming:** standardize to `wb_boundaries_lgu_{iso3}` like every other table.
  This removes all country-name derivation from table naming and eliminates the
  Laos mismatch. The 36 existing country-named LGU tables will be regenerated
  under ISO3 names and the old ones dropped.
- **Targets:** `dev-wei` → schema `pim` (safe testing); new `prod` → `sgpbpi163`.
- **Batch default:** the 36 ISO3s already present in `sgpbpi163`.

## Changes

### 1. Parameterization & config consolidation — `shared/settings.py`

Centralize run-varying config in `shared/settings.py` using the existing
`_get_uc_schema()` widget-with-fallback pattern (so local, non-Databricks runs
still work), and have `extract/config.py` and `transform/config.py` import from
there instead of defining their own copies.

**Country / population (widgets):**

- Replace hardcoded `COUNTRY`/`ISO_2`/`ISO_3` with derivation from a
  `COUNTRY_ISO3` widget (default `"LAO"`, so local dev and existing behavior are
  preserved).
- `ISO_3 = <COUNTRY_ISO3 widget>`; `ISO_2 = pycountry.countries.get(alpha_3=ISO_3).alpha_2`.
- `COUNTRY` = a human-readable display name from `pycountry` (used only in log
  output now, not in table names).
- `POPULATION_YEAR` becomes a widget (default `2025`).

**Run-control (widgets), consolidated from the two config files:**

- **`FORCE_RECOMPUTE`** — widget (bool, default `False`), parsed like the existing
  `ENABLE_VISUALIZATION` widget. Currently duplicated in `extract/config.py:42`
  and `transform/config.py:108`; both import it from settings instead. Lets an
  operator force a re-run per country without editing code, and supports the
  review-item #6 fix.
- **`INCLUDE_ADM_LEVEL0`** — widget (bool, default `True`). Currently duplicated
  in `extract/config.py:45` and `transform/config.py:88`; both import it. Enables
  province-only reruns.

**Drift-prevention (shared constant, NOT a widget):**

- **`H3_RESOLUTION`** — define once in `shared/settings.py` (`= 8`). Today there
  are two independent literals that "must match": `transform/config.py:105` and
  the `h3_resolution: int = 8` default in `extract/02_population.py:62`. Both
  read the shared constant instead. Kept a constant (not a widget) so it cannot
  vary per run and break cross-country comparability.

**Volume isolation:**

- **`VOLUME_DIR`** (`extract/config.py:39`) currently hardcodes the `sgpbpi163`
  volume regardless of `UC_SCHEMA`, so dev (`pim`) and prod (`sgpbpi163`) share
  one volume for worldpop rasters, WB geojson caches, and the facilities input.
  The volume name (`vgpbpi163`) is not derivable from the schema name, so expose
  it as a bundle variable `uc_volume` (path segment `schema/volume`, default
  `sgpbpi163/vgpbpi163`) passed as a widget; `VOLUME_DIR =
  f"/Volumes/{UC_CATALOG}/{uc_volume}"`. Targets override it. Default preserves
  today's behavior (existing cached reference data stays reachable).
  Open question for review: whether the `pim` dev target has its own volume, or
  should keep pointing at the `sgpbpi163` volume for shared read-only reference
  data.

### 2. LGU naming standardization — `shared/core.py`, configs

- `shared/core.py`: change the `lgu` entries in `get_extract_table_names` and
  `get_transform_table_names` (4 occurrences) from
  `wb_boundaries_lgu_{_sanitize_adm_name(country)}` to
  `wb_boundaries_lgu_{iso3.lower()}`. Keep `_sanitize_adm_name` for province/adm1
  suffixes (those are region names, unaffected).
- `extract/config.py`: `COUNTRY_LGU_TABLE` → ISO3-based.
- `transform/config.py`: `_get_adm_level1_names_from_uc` lgu_table → ISO3-based.
- The `country` parameter of `get_*_table_names` becomes vestigial. Keep the
  signature as-is for now to avoid churn to callers and tests (do not change
  signatures per project rules); document that `country` is retained only for
  backward-compatible call sites.

### 3. Fix the WB download URL — `extract/config.py`

- Update `WB_BOUNDARIES_BASE_URL` to the working
  `ddh-published/0038272/DR0095369/…` path. (The `download_wb_geojson` cache-skip
  logic is unchanged; a fresh fetch requires the cached file to be absent.)

### 4. DAB restructure — `databricks.yml`

- **Merge** `extract_pipeline` + `transform_pipeline` into one `pipeline` job
  (delivers goal #4). Tasks: a `run_tests` gate task first, then the 5 extract
  notebooks (the first extract task `depends_on: run_tests`), then the 4
  transform notebooks, with `prepare` `depends_on` `facilities`. Use **job-level
  parameters** (`COUNTRY_ISO3`, `UC_SCHEMA`, `POPULATION_YEAR`, `FORCE_RECOMPUTE`,
  `INCLUDE_ADM_LEVEL0`, and the `uc_volume` variable) referenced by each task's
  `base_parameters` via `{{job.parameters.<name>}}`, replacing the per-task
  `notebook_defaults` anchor. (`H3_RESOLUTION` is a code constant, not a param.)
- **Add** a `batch` job: a single `for_each_task` iterating a JSON array of ISO3
  strings (default = the 36), each iteration a `run_job_task` → the `pipeline`
  job with `{{input}}` supplied as `COUNTRY_ISO3` (delivers goal #3; country
  parameterization from section 1 is what makes per-iteration params possible).
  Set a sensible `concurrency` so countries run in parallel without overloading
  the cluster.
- **Targets:** keep `dev-wei` (schema `pim`); add `prod` (schema `sgpbpi163`)
  with prod `workspace_root` and `cluster_id`.
- **Delete** `pipeline_runner.py`.

### 5. Review-item polish — `extract/04_facilities.py`

- **#5:** `extract_health_facilities_osm` branches on the `adm_level_name`
  parameter (`== "Country"`) instead of reading the module-global `adm_level1`.
- **#6:** honor `force` for provinces — do not read the country-level cache when
  `FORCE_RECOMPUTE` is set (or only when the country cache was refreshed in this
  run), so a forced province run cannot silently reuse stale data.

### 6. Dependency

- Add `pycountry` to `requirements.txt` and `pyproject.toml`.

### 7. Test gate & CI

- New `tests/run_tests.py` Databricks notebook — the `pipeline` job's first task;
  runs the full suite in an isolated, local-forced subprocess (see Testing).
- New `tests/conftest.py` — autouse guard asserting local env / temp backend so
  tests can never write to `prd_mega`.
- New `.github/workflows/tests.yml` — unit-tests-only CI on push/PR.
- `databricks.yml` — add the `run_tests` task; the first extract task
  `depends_on` it.

## Rollout / migration

Standardizing LGU naming orphans the 36 existing country-named LGU tables. Because
transform's province discovery reads the LGU table, the ISO3-named LGU table must
exist before transform runs for an existing country.

1. Merge code changes.
2. Run the `batch` job (or `03_boundaries` per country) for the 36 to create
   `wb_boundaries_lgu_{iso3}` tables. LGU extraction is a cheap Admin2 filter.
3. After verifying, drop the old `wb_boundaries_lgu_{country}` tables.

## Downstream consumers — dashboard (`../pimpam-dash`)

Verified the dashboard is unaffected by the LGU rename:

- It has **no references to `wb_boundaries_lgu_*`** (or any `wb_boundaries_*`) in
  code — only in `README.md` prose. The table we rename is intermediate and not
  consumed downstream.
- The tables it *does* read are already ISO3-named, from `prd_mega.sgpbpi163`:
  `health_facilities_{iso3}_osm` (+ `_{slug}_province`),
  `lgu_accessibility_results_{iso3}_{suffix}` (+ province variants), and
  `base_dashboard_data_{iso3}` (map center + boundary geometry + baselines).
- **The dashboard's map boundary IS the WB official boundary.** The AOI
  `geometry_wkt` in `base_dashboard_data_{iso3}` is populated by
  `transform/03_optimize.py:421-424` from `tables["boundaries"]` =
  `wb_boundaries_{iso3}` (`shared/core.py:157,170`), written at
  `03_optimize.py:454`, and read back in `pimpam-dash/queries.py:312-320,360`.
  The `gadm_boundaries_zmb` reference (`queries.py:443-461`,
  `get_gadm_boundary_wkt`) is dead legacy code — no callers.

**Constraint:** this work must NOT rename or alter `health_facilities_{iso3}_osm`
or `lgu_accessibility_results_{iso3}_*` — the dashboard depends on both. The
design does not touch them (both are already ISO3-named in `shared/core.py` and
left unchanged). Only `wb_boundaries_lgu_{country}` → `_{iso3}` changes.

## Testing

### New / updated tests

- **Unit:** country derivation in `settings.py` — ISO3 → ISO2/name for the 36,
  explicitly covering PSE/SRB/YEM (correct standard ISO2) and LAO.
- **Unit:** widget parsing/fallback in `settings.py` — `FORCE_RECOMPUTE` and
  `INCLUDE_ADM_LEVEL0` bool parsing and their local (no-widget) defaults; and
  that `H3_RESOLUTION` resolves to a single shared value used by both extract and
  transform.
- **Unit:** update `tests/test_core.py` naming assertions for ISO3-based LGU
  table names.
- **Validate:** `databricks bundle validate` for both `dev-wei` and `prod`.

### Pipeline test gate (`run_tests` task)

Runs the **full** suite (incl. `test_integration.py`) as the first task in the
`pipeline` job, gating every run. It must **never** read/write `prd_mega` (dev
`pim` or prod `sgpbpi163`) tables or volumes. Mechanism:

- Run pytest in an **isolated subprocess** on the driver
  (`subprocess.run([sys.executable, "-m", "pytest", "tests/", ...])`) with a
  **scrubbed environment**: `DATABRICKS_RUNTIME_VERSION` and the Databricks
  connection vars removed, and a temp `base_dir`. This forces
  `detect_environment()` → `LOCAL`, so `get_storage_backend()` returns the
  `LocalStorageBackend` (no UC writes), and the integration tests get their own
  in-process local Spark. Because it is a fresh process, the fixture's
  `spark.stop()` tears down only that local Spark — the cluster session is
  untouched. The task raises on non-zero pytest exit, failing the pipeline.
- `%pip install pytest` (already a declared dep) if the cluster image lacks it.

### Write-safety guard (defense in depth)

Add `tests/conftest.py` with an autouse, session-scoped fixture that **asserts
`shared.env.is_local()` and that the resolved storage backend's `base_dir` is a
temp path** before any test runs. If the suite is ever launched in a context that
resolves to a Databricks backend, it hard-fails instead of writing to a real
destination. Verified today: `test_integration.py` performs no UC/volume writes
(tempfile + local Spark only), and `test_env.py` writes only through an explicit
`LocalStorageBackend(temp_dir)`; this guard keeps that invariant enforced.

### CI (GitHub Actions)

Add a workflow (alongside the existing `compliance.yml`) that runs **unit tests
only** — `pytest tests/test_core.py tests/test_env.py` — on push/PR. Excludes
`test_integration.py` to avoid provisioning PySpark/Java in the runner (the
integration tests run in the on-cluster gate instead). There is no test CI today.

## Out of scope

- Rewriting the OSM extraction or optimization logic.
- `ADM_LEVEL1_LIST` stays a code default (`[]` = all provinces). It is a list
  (would need CSV-widget parsing), rarely overridden, and tangled with the
  commented-out per-country examples; batch-by-country does not need it.
- Analysis/algorithm knobs stay code defaults so cross-country results and the
  dashboard remain comparable: `DISTANCES_METERS`, `TARGET_NEW_FACILITIES`,
  `POTENTIAL_TYPE`/`GRID_SPACING`/`N_CLUSTERS`, `TARGET_ACCESS_RATE_PCT`,
  `VIZ_SAMPLE_SIZE`, `FACILITIES_SOURCE`.
- `MAPBOX_ACCESS_TOKEN` is a secret — if the travel API is ever enabled it must
  use Databricks secrets, never a plain widget/param. Not enabled today.
- Backfilling `potential_coverage` for countries that only have coverage tables.
