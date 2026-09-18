# Databricks notebook source
# DAB test gate. This cluster doesn't mount /Workspace to the driver, so pytest
# can't discover files there; instead install the wheel (tests bundled) from the UC
# Volume (dist_dir, per target) and run `pytest --pyargs tests` from site-packages.

# COMMAND ----------

import glob
import sys
import subprocess

dist_dir = (dbutils.widgets.get("dist_dir") or "").strip()
if not dist_dir:
    raise RuntimeError("'dist_dir' is not set — expected from the DAB job's base_parameters.")

# Install the exact wheel by path so a public index can't substitute the project
# package for a higher-version one on this credentialed cluster.
wheels = sorted(glob.glob(f"{dist_dir}/health_facility_location_optimizer-*.whl"))
if not wheels:
    raise RuntimeError(f"No project wheel found in {dist_dir}")
wheel = wheels[-1]

# First install pulls deps (geopandas/pycountry/…) + pytest; the second forces the
# project itself fresh (static version, so a plain install would keep a cached copy).
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", wheel, "pytest"], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall", "--no-deps", wheel],
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
