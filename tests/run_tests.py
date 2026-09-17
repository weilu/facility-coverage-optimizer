# Databricks notebook source
# Pipeline test gate. Runs the full suite (including the `databricks`-marked H3
# tests) in-process on the cluster so those tests use the cluster's native Spark
# session and real H3 SQL. Raises on any failure so the pipeline job does not
# proceed with broken code.
#
# Wired into databricks.yml as the first task of the pipeline job (added when the
# extract+transform jobs are merged).

# COMMAND ----------

# MAGIC %pip install pytest pycountry

# COMMAND ----------

import os
import sys

import pytest

# Make the repo root (containing shared/, transform/, tests/) importable and the
# working directory, regardless of where the notebook is executed from.
_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_root, "shared")) and os.path.isdir(os.path.join(_root, "tests")):
        break
    _root = os.path.dirname(_root)

if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

# COMMAND ----------

# Run everything (no `-m 'not databricks'`) — the cluster is where H3 tests run.
exit_code = pytest.main(["tests", "-p", "no:cacheprovider"])
if exit_code != 0:
    raise RuntimeError(f"Test gate failed: pytest exit code {exit_code}")

print("All tests passed — pipeline may proceed.")
