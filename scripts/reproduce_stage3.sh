#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_dir/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_dir/.cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$repo_dir/results"
run_dir="$(mktemp -d "$repo_dir/results/stage3-XXXXXX")"
"$python_bin" -m unittest discover -s tests -v 2>&1 | tee "$run_dir/tests.log"
for sample in tiny no_local_combine return_feedback zipf; do
  for repeat in a b; do
    "$python_bin" -m vdie_moe_sim --stage stage3 --config "configs/stage3_${sample}.json" \
      --out "$run_dir/${sample}-${repeat}" --timeline > "$run_dir/${sample}-${repeat}.log" 2>&1
  done
done
"$python_bin" - "$run_dir" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
report = {}
for sample in ("tiny", "no_local_combine", "return_feedback", "zipf"):
    left, right = root / f"{sample}-a", root / f"{sample}-b"
    files = sorted(p.name for p in left.iterdir())
    if files != sorted(p.name for p in right.iterdir()):
        raise RuntimeError(f"different exported files: {sample}")
    for name in files:
        if (left / name).read_bytes() != (right / name).read_bytes():
            raise RuntimeError(f"non-reproducible output: {sample}/{name}")
    report[sample] = {"identical_files": files, "repeats": 2}
on = json.loads((root / "tiny-a/manifest.json").read_text())
off = json.loads((root / "no_local_combine-a/manifest.json").read_text())
if on["workload_sha256"] != off["workload_sha256"] or on["config"]["phy"] != off["config"]["phy"]:
    raise RuntimeError("Local Combine comparison changed workload or total PHY budget")
np.testing.assert_allclose(np.load(root / "tiny-a/outputs.npy", allow_pickle=False),
                           np.load(root / "no_local_combine-a/outputs.npy", allow_pickle=False), rtol=1e-10, atol=1e-10)
report["local_combine_equivalence"] = {"same_workload": True, "same_phy_budget": True,
                                        "numerically_equal": True, "rtol": 1e-10, "atol": 1e-10}
(root / "reproducibility.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Stage 3 tests, repeated exports and Local Combine equivalence passed. Artifacts: {root}")
PY
