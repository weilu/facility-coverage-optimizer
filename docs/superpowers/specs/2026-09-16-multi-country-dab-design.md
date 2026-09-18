# Multi-Country DAB Packaging — Design

Date: 2026-09-16
Status: Code complete on branch `multi-country-dab`; dev verification pending.

## Implementation status

All code slices implemented and committed on branch `multi-country-dab`; local
suite green (73 passing, 2 databricks-marked deselected off-cluster); both bundle
targets `databricks bundle validate` OK.

- **Done — test framework & CI** (`c155fe2`): `shared/transform_ops.py`, real
  tests, cluster-aware `spark` fixture, UC-write guard, `databricks` marker +
  off-cluster auto-skip, CI workflow, `tests/run_tests.py` gate.
- **Done — Task 4 WB URL** (`aada296`): `ddh_bytes` volume-or-URL fallback.
- **Done — Task 1 country params** (`590de86`): ISO3→ISO2/name via pycountry.
- **Done — Task 2 config consolidation** (`0153033`): FORCE_RECOMPUTE /
  INCLUDE_ADM_LEVEL0 widgets, shared H3_RESOLUTION.
- **Done — Task 3 LGU naming** (`df37d87`): `wb_boundaries_lgu_{iso3}`.
- **Done — Task 5 VOLUME_DIR** (`298ca4d`): `UC_VOLUME` widget.
- **Done — Task 6 cache #5/#6** (`11bd2ed`): `should_load_country_cache`.
- **Done — Task 7 DAB** (`89498ee`): merged `pipeline` job + `batch` fan-out +
  `prod` target; deleted `pipeline_runner.py`.
- Created managed dev volume `prd_mega.pim.vpim`; `dev-wei` uses `pim` / `pim/vpim`.

**On-cluster test gate — WORKING (verified 2026-09-17):** cluster `0212` does not
mount `/Workspace` to the driver FS, so pytest can't discover/import code deployed
via sync/Repo/bundle/source:GIT. Solved with the prospects pattern — build a wheel
bundling `tests/`, publish to a UC Volume (`/Volumes/prd_mega/pim/vpim/dist/`),
`pip install` it on the cluster and run `pytest --pyargs tests`. Full suite passes
on-cluster incl. the H3 tests. Operational step: the wheel must be rebuilt+
published on code change (not yet wired into CI). See [[dev-cluster-wsfs-limitation]].

**CI wiring — DONE:** `scripts/publish_wheel.sh <target>` builds the wheel and
publishes it to the target's UC Volume `dist/`; `.github/workflows/publish-wheel.yml`
runs it on merge to main / on demand (needs `DATABRICKS_HOST` + OAuth secrets).

**Checkpoint C — VERIFIED (2026-09-18):** full single-country pipeline run for LAO
into `pim` on `dev-wei`. 9/10 tasks succeeded — `run_tests` gate, all 5 extract,
`prepare`/`coverage`/`optimize`. Produced 497 ISO3-named LAO tables in `pim`
(incl. `wb_boundaries_lgu_lao` — the Task 3 rename — and `base_dashboard_data_lao`),
none in prod `sgpbpi163`. Confirms country derivation, WB `ddh_bytes` fetch, ISO3
LGU naming, `UC_VOLUME` isolation, widgets, and the on-cluster gate together.
`visualize` (optional last stage) failed on a transient cluster driver restart —
infra, not code.

**Remaining before merge:** publish the wheel to the prod volume + add CI secrets
(prod automation); optionally repair-run `visualize`. Feature-specific tests were
written TDD-style with each slice.

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

### 7. Test gate & CI — DONE (commit `c155fe2`)

- **Extract testable transform ops into `shared/transform_ops.py`** (importable,
  `# Databricks notebook source` so `%run` works, side-effect-free):
  `generate_grid_in_polygon`, `add_facility_h3_index`, and
  `compute_coverage_h3_internal` (renamed from the notebook-private
  `_compute_coverage_h3_internal`). Needed because these lived in the
  un-importable `01_prepare.py` / `02_coverage.py` (digit-prefixed filenames +
  module-level side effects). The notebooks now `%run`/import them. pyspark is
  imported lazily inside the H3 helpers so the module stays PySpark-free.
- **Replace `tests/test_integration.py`** (library smoke-tests that imported no
  project code) with `tests/test_transform_ops.py`: a pure `generate_grid_in_polygon`
  test (CI) and `@pytest.mark.databricks` Spark tests calling the real H3
  transforms. `test_core.py` already covers `solve_mclp_greedy` / `get_k_rings`.
- `tests/conftest.py` — cluster-aware `spark` fixture (uses the existing cluster
  session, no `spark.stop()` on Databricks); autouse guard patching UC-write entry
  points (`DataFrameWriter.saveAsTable`, `gdf_to_uc_table`,
  `DatabricksStorageBackend.save_*`) so tests can never write to `prd_mega`;
  auto-skip of `databricks`-marked tests when `not is_databricks()`.
- `tests/run_tests.py` Databricks notebook — runs `pytest tests/` in the cluster
  session; raises on failure. (Wiring it into `databricks.yml` is deferred to
  section 4, which creates the merged `pipeline` job.)
- `.github/workflows/tests.yml` — `pytest -m "not databricks"` on push/PR,
  installing `.[dev]` (PySpark-free).
- `databricks` marker registered in `pyproject.toml`.
- **Deferred to section 4:** add the `run_tests` task to `databricks.yml`; the
  first extract task `depends_on` it.

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

### Audit finding — replace the integration tests

`tests/test_integration.py` imports **no project code** (only pyspark / pandas /
geopandas / numpy / shapely). Every test reimplements a simplified version of the
logic inline and asserts that the *library* behaves — so it would pass through any
real pipeline regression (`compute_coverage_h3`, `generate_grid_in_polygon`,
`solve_mclp_greedy`, etc. could all break silently). It is deleted and replaced
with tests that call the real functions. `test_core.py` (which does import real
code) stays.

A hard constraint shapes the split: the real Spark transforms use **Databricks-
native H3 SQL** (`h3_kring`, `h3_polyfillash3`, `h3_longlatash3`) that does not
exist in open-source PySpark. So H3 transforms are only meaningfully testable on a
cluster; everything else runs anywhere.

### Test groups

**Pure-Python unit tests (no Spark) — run in CI and on-cluster:**
- Country derivation in `settings.py` — ISO3 → ISO2/name for the 36, covering
  PSE/SRB/YEM and LAO.
- Widget parsing/fallback — `FORCE_RECOMPUTE`, `INCLUDE_ADM_LEVEL0`; single shared
  `H3_RESOLUTION`.
- `shared/core.py` naming — updated ISO3-based LGU assertions.
- Real algorithm coverage by calling actual functions:
  `shared.core.solve_mclp_greedy`, `get_k_rings` (both already in `test_core.py`),
  and `shared.transform_ops.generate_grid_in_polygon` (pure pandas/shapely).

**Databricks-only Spark tests (need real H3 SQL) — on-cluster gate only,
`@pytest.mark.databricks`, skipped when `not is_databricks()`:**
- `shared.transform_ops.add_facility_h3_index` (extracted from `01_prepare.py`).
  (`locations_pdf_to_spark` was left in the notebook — it depends on the
  `st_point_wkt` UDF and module-level `spark` — so it is not tested this slice.)
- `shared.transform_ops.compute_coverage_h3_internal` — the **pure transform**
  (returns DataFrames), NOT the `compute_coverage_h3` wrapper (which
  `saveAsTable`s) — on small synthetic input, asserting coverage numbers.
- population H3 indexing (`02_population`).

### Pipeline test gate (`run_tests` task)

First task in the `pipeline` job, gating every run. Runs `pytest tests/` in the
cluster **notebook process** so the Databricks-marked tests use the cluster's
native Spark session (real H3 SQL). The `spark` fixture becomes cluster-aware: on
Databricks it uses the existing session and does **not** call `spark.stop()`
(which would tear down the shared cluster session, breaking downstream tasks);
locally it builds and stops a `local[2]` session. `%pip install pytest` if the
cluster image lacks it (already a declared dep).

### Write-safety (no prod/dev writes) — hard requirement

- Spark tests call only the **pure transform functions that return DataFrames**
  and assert in-memory; they never call the caching wrappers that `saveAsTable`
  to UC.
- `tests/conftest.py` autouse session guard **patches the UC-write entry points**
  — `pyspark.sql.DataFrameWriter.saveAsTable` and the storage-backend `save_*` /
  `gdf_to_uc_table` helpers — to raise during the test session. Any accidental
  write to `prd_mega` (dev `pim` or prod `sgpbpi163`) fails the test instead of
  mutating data. This holds regardless of environment (works for the on-cluster
  gate, where forcing local mode is not possible because H3 tests need the real
  session).

### CI (GitHub Actions)

New workflow (alongside `compliance.yml`): runs the **pure-Python unit tests
only** — `pytest -m "not databricks"` — on push/PR, so the runner needs no
PySpark/Java. The Spark/H3 tests run in the on-cluster gate. There is no test CI
today.

### Validate

- `databricks bundle validate` for both `dev-wei` and `prod`.

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
