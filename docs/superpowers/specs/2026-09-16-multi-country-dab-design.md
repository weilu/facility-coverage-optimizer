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

### 1. Country parameterization — `shared/settings.py`

- Replace hardcoded `COUNTRY`/`ISO_2`/`ISO_3` with derivation from a
  `COUNTRY_ISO3` widget (default `"LAO"`, so local dev and existing behavior are
  preserved).
- `ISO_3 = <COUNTRY_ISO3 widget>`; `ISO_2 = pycountry.countries.get(alpha_3=ISO_3).alpha_2`.
- `COUNTRY` = a human-readable display name from `pycountry` (used only in log
  output now, not in table names).
- `POPULATION_YEAR` becomes a widget (default `2025`).
- Mirror the existing `_get_uc_schema()` widget-with-fallback pattern for each
  new widget so local (non-Databricks) runs still work.

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
  (delivers goal #4). Tasks: the 5 extract notebooks then the 4 transform
  notebooks, with `prepare` `depends_on` `facilities`. Use **job-level
  parameters** (`COUNTRY_ISO3`, `UC_SCHEMA`, `POPULATION_YEAR`) referenced by each
  task's `base_parameters` via `{{job.parameters.<name>}}`, replacing the
  per-task `notebook_defaults` anchor.
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

## Rollout / migration

Standardizing LGU naming orphans the 36 existing country-named LGU tables. Because
transform's province discovery reads the LGU table, the ISO3-named LGU table must
exist before transform runs for an existing country.

1. Merge code changes.
2. Run the `batch` job (or `03_boundaries` per country) for the 36 to create
   `wb_boundaries_lgu_{iso3}` tables. LGU extraction is a cheap Admin2 filter.
3. After verifying, drop the old `wb_boundaries_lgu_{country}` tables.

## Testing

- **Unit:** country derivation in `settings.py` — ISO3 → ISO2/name for the 36,
  explicitly covering PSE/SRB/YEM (correct standard ISO2) and LAO.
- **Unit:** update `tests/test_core.py` naming assertions for ISO3-based LGU
  table names.
- **Validate:** `databricks bundle validate` for both `dev-wei` and `prod`.
- Existing `tests/test_core.py` (other cases) stays green.

## Out of scope

- Rewriting the OSM extraction or optimization logic.
- Changing widgets other than country/population (e.g. `FORCE_RECOMPUTE`,
  `INCLUDE_ADM_LEVEL0`, `ADM_LEVEL1_LIST` keep their current config defaults).
- Backfilling `potential_coverage` for countries that only have coverage tables.
