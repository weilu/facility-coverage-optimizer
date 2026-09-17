# Databricks notebook source
# DAB test gate. The project wheel (with tests/ bundled — see pyproject) is
# published to a UC Volume; we pip-install it and run `pytest --pyargs tests`, so
# discovery and imports come from site-packages, NOT /Workspace. This cluster does
# not mount workspace files to the driver FS, so filesystem-based collection fails
# — the wheel + --pyargs approach (mirroring prospects_data_pipelines) sidesteps it.
# `dist_dir` is supplied by the DAB per target (dev vs prod volume).

# COMMAND ----------

import sys
import subprocess

dist_dir = (dbutils.widgets.get("dist_dir") or "").strip()
if not dist_dir:
    raise RuntimeError("'dist_dir' is not set — expected from the DAB job's base_parameters.")

# Install the project wheel (pulls its own deps: geopandas/shapely/pycountry/…) plus pytest.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--find-links", dist_dir,
     "health-facility-location-optimizer", "pytest"],
    check=True,
)

# COMMAND ----------

import io
import contextlib

import pytest

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    # --pyargs discovers the installed `tests` package; no workspace-file enumeration.
    exit_code = pytest.main(["-p", "no:cacheprovider", "-rA", "--pyargs", "tests"])
_output = _buf.getvalue()
print(_output)  # full output to the driver log
if exit_code != 0:
    raise RuntimeError(f"Test gate failed: pytest exit code {exit_code}\n{_output[-3500:]}")

print("All tests passed — pipeline may proceed.")
