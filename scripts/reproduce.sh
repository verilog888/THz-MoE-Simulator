#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tov-matplotlib"
.venv/bin/python -m tov_sim.run_all --spec ToV_MoE_MINIMAL_SIM_SPEC.md --output "${1:-Results}" --workers "${TOV_WORKERS:-4}"
