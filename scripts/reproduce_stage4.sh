#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_dir/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_dir/.cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$repo_dir/results"
run_dir="$(mktemp -d "$repo_dir/results/stage4-XXXXXX")"
"$python_bin" -m unittest discover -s tests -v 2>&1 | tee "$run_dir/tests.log"
for sample in tiny concurrent limited zipf; do
  for repeat in a b; do
    echo "Running five schemes: $sample / $repeat"
    "$python_bin" -m vdie_moe_sim --stage stage4 --compare --config "configs/stage4_${sample}.json" \
      --out "$run_dir/${sample}-${repeat}" --timeline > "$run_dir/${sample}-${repeat}.log" 2>&1
  done
done
"$python_bin" - "$run_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = {}
for sample in ("tiny", "concurrent", "limited", "zipf"):
    left, right = root / f"{sample}-a", root / f"{sample}-b"
    files = sorted(p.relative_to(left).as_posix() for p in left.rglob("*") if p.is_file())
    if files != sorted(p.relative_to(right).as_posix() for p in right.rglob("*") if p.is_file()):
        raise RuntimeError(f"different exported files: {sample}")
    for name in files:
        if (left / name).read_bytes() != (right / name).read_bytes():
            raise RuntimeError(f"non-reproducible output: {sample}/{name}")
    fairness = json.loads((left / "fairness.json").read_text())
    for key in ("same_workload", "same_total_phy_budget", "same_compute_and_timing", "numerically_equal"):
        if fairness[key] is not True:
            raise RuntimeError(f"fairness/numeric failure: {sample}/{key}")
    report[sample] = {"identical_files": files, "repeats": 2, "schemes": ["A", "B", "C", "D", "Proposed"],
                      "fairness": fairness}
(root / "reproducibility.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Stage 4 tests, five-scheme fairness and repeated exports passed. Artifacts: {root}")
PY
