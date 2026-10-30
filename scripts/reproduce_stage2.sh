#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_dir/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_dir/.cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$repo_dir/results"
run_dir="$(mktemp -d "$repo_dir/results/stage2-XXXXXX")"
"$python_bin" -m unittest discover -s tests -v 2>&1 | tee "$run_dir/tests.log"
for sample in tiny batching trace zipf; do
  for repeat in a b; do
    "$python_bin" -m vdie_moe_sim --stage stage2 --config "configs/stage2_${sample}.json" \
      --out "$run_dir/${sample}-${repeat}" --timeline > "$run_dir/${sample}-${repeat}.log" 2>&1
  done
done
"$python_bin" - "$run_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = {}
for sample in ("tiny", "batching", "trace", "zipf"):
    left, right = root / f"{sample}-a", root / f"{sample}-b"
    files = sorted(p.name for p in left.iterdir())
    if files != sorted(p.name for p in right.iterdir()):
        raise RuntimeError(f"different exported files: {sample}")
    for name in files:
        if (left / name).read_bytes() != (right / name).read_bytes():
            raise RuntimeError(f"non-reproducible output: {sample}/{name}")
    report[sample] = {"identical_files": files, "repeats": 2}
(root / "reproducibility.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Stage 2 tests and identical repeated exports passed. Artifacts: {root}")
PY
