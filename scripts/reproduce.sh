#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_dir/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_dir/.cache}"
exec .venv/bin/python -m vdie_moe_sim.run_all --spec VDIE_MOE_SIM_SPEC.md --output results "$@"
