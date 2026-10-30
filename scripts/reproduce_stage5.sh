#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_dir/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_dir/.cache}"
samples=("$@")
if [[ ${#samples[@]} -eq 0 ]]; then samples=(smoke ablation paper_sample); fi
for sample in "${samples[@]}"; do
  case "$sample" in smoke|ablation|paper_sample) ;; *) echo "Expected smoke, ablation or paper_sample" >&2; exit 2 ;; esac
done
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$repo_dir/results"
run_dir="$(mktemp -d "$repo_dir/results/stage5-XXXXXX")"
echo "Artifacts: $run_dir"
"$python_bin" -m unittest discover -s tests -v 2>&1 | tee "$run_dir/tests.log"
"$python_bin" -m vdie_moe_sim --stage stage5 --plan configs/sweeps/stage5_paper_full.json --dry-run \
  > "$run_dir/paper_full.preview.json"
for sample in "${samples[@]}"; do
  for repeat in a b; do
    echo "Running sweep: $sample / $repeat"
    "$python_bin" -m vdie_moe_sim --stage stage5 --plan "configs/sweeps/stage5_${sample}.json" \
      --out "$run_dir/${sample}-${repeat}" > "$run_dir/${sample}-${repeat}.log" 2>&1
    echo "Auditing exported data: $sample / $repeat"
    "$python_bin" -m vdie_moe_sim --stage stage5 --audit --out "$run_dir/${sample}-${repeat}" \
      > "$run_dir/${sample}-${repeat}.audit.json"
  done
done
"$python_bin" - "$run_dir" "${samples[@]}" <<'PY'
import json
from pathlib import Path
import sys
from vdie_moe_sim.sweep import file_digest

root=Path(sys.argv[1])
report={}
for sample in sys.argv[2:]:
    left,right=root/f"{sample}-a",root/f"{sample}-b"
    files=sorted(p.relative_to(left).as_posix() for p in left.rglob('*') if p.is_file())
    if files!=sorted(p.relative_to(right).as_posix() for p in right.rglob('*') if p.is_file()):
        raise RuntimeError(f"different exported file sets: {sample}")
    for name in files:
        if file_digest(left/name)!=file_digest(right/name):
            raise RuntimeError(f"non-reproducible output: {sample}/{name}")
    manifest=json.loads((left/'sweep_manifest.json').read_text())
    audit=json.loads((root/f'{sample}-a.audit.json').read_text())
    if manifest['status']!='complete' or audit['status']!='passed':
        raise RuntimeError(f"incomplete/unaudited sweep: {sample}")
    report[sample]={'repeats':2,'runs_per_repeat':manifest['completed_runs'],'identical_file_pairs':len(files),
                    'identical_files':files,'offline_audit':audit}
(root/'reproducibility.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(f"Stage 5 tests, full CSV/NPY audits and repeated CSV/JSON/NPY/PNG/SVG exports passed. Artifacts: {root}")
PY
