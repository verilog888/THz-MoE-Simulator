#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
bootstrap_python="${BOOTSTRAP_PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
  "$bootstrap_python" -m venv .venv
fi
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$repo_dir/.cache/pip}"
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.lock
.venv/bin/python -m pip install --disable-pip-version-check --no-build-isolation --no-deps -e .
.venv/bin/python -m pip check
