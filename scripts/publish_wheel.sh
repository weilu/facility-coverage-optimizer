#!/usr/bin/env bash
# Build the project wheel and publish it to a target's UC Volume dist/ dir, where
# the on-cluster test gate (tests/run_tests.py) pip-installs it via --find-links.
# The wheel bundles tests/ (see pyproject) so pytest --pyargs can discover them
# without /Workspace file access — required because the dev cluster does not mount
# workspace files to the driver FS.
#
# Usage: scripts/publish_wheel.sh <dev-wei|prod>
# Auth: uses the databricks CLI's ambient auth. Locally, set DATABRICKS_PROFILE
#       (e.g. adb-6102124407836814); in CI, set DATABRICKS_HOST + OAuth secrets.
set -euo pipefail

TARGET="${1:?usage: scripts/publish_wheel.sh <dev-wei|prod>}"
case "$TARGET" in
  dev-wei) VOLUME="pim/vpim" ;;
  prod)    VOLUME="sgpbpi163/vgpbpi163" ;;
  *) echo "unknown target: $TARGET (expected dev-wei or prod)" >&2; exit 1 ;;
esac

PROFILE_ARG=()
[ -n "${DATABRICKS_PROFILE:-}" ] && PROFILE_ARG=(-p "${DATABRICKS_PROFILE}")

DIST_VOL="dbfs:/Volumes/prd_mega/${VOLUME}/dist"

uv build --wheel
WHL="$(ls -t dist/*.whl | head -1)"

databricks fs mkdirs "${DIST_VOL}" "${PROFILE_ARG[@]}" || true
databricks fs cp "${WHL}" "${DIST_VOL}/$(basename "${WHL}")" --overwrite "${PROFILE_ARG[@]}"
echo "published $(basename "${WHL}") -> ${DIST_VOL}"
