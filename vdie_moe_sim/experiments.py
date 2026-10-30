"""Controlled five-scheme comparisons, also used by the Stage 5 sweep runner."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import stable_digest
from .scheduled_layer import SCHEMES, effective_config, run_scheduled


def fairness_config(config):
    value = config.to_dict()
    value.pop("name")
    value["combine"].pop("local_combine")
    value["scheduling"].pop("scheme")
    return value


def audit_comparison(results):
    if any(r.config.ablation is not None for r in results):
        raise ValueError("canonical five-scheme comparison cannot include ablations")
    if tuple(r.config.scheduling.scheme for r in results) != SCHEMES:
        raise ValueError("comparison must contain A/B/C/D/Proposed in order")
    reference = results[0]
    for r in results:
        if fairness_config(r.config) != fairness_config(reference.config) or r.workload.digest != reference.workload.digest:
            raise ValueError("comparison changed bandwidth, resources or workload")
        if [o.token_id for o in r.outputs] != [o.token_id for o in reference.outputs]:
            raise ValueError("comparison output tokens differ")
        if not all(np.allclose(x.vector, y.vector, rtol=1e-10, atol=1e-10) for x, y in zip(reference.outputs, r.outputs)):
            raise ValueError("comparison numerical outputs differ")
    return {"same_workload": True, "same_total_phy_budget": True, "same_compute_and_timing": True,
            "numerically_equal": True, "workload_sha256": reference.workload.digest,
            "controlled_config_sha256": stable_digest(fairness_config(reference.config)),
            "controlled_differences": ["scheduling.scheme", "combine.local_combine"],
            "rtol": 1e-10, "atol": 1e-10}


def run_comparison(config, directory, *, trace_base_dir=".", timeline=False):
    if config.ablation is not None:
        raise ValueError("use a Stage 5 ablation study instead of canonical --compare")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("comparison output directory must be empty")
    results = tuple(run_scheduled(effective_config(config, scheme), trace_base_dir=trace_base_dir) for scheme in SCHEMES)
    report = audit_comparison(results)
    for r in results:
        output = r.export(directory / r.config.scheduling.scheme)
        if timeline:
            from .plots import plot_scheduled_timeline
            plot_scheduled_timeline(r, output / "timeline.png")
    rows = [r.summary for r in results]
    reference_latency = rows[2]["layer_makespan_s"]
    for row in rows:
        row["speedup_vs_C"] = reference_latency / row["layer_makespan_s"]
    table = pd.DataFrame(rows)
    table.to_csv(directory / "comparison.csv", index=False, float_format="%.17g", lineterminator="\n")
    (directory / "fairness.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    from .plots import plot_scheme_comparison
    plot_scheme_comparison(directory / "comparison.csv", directory / "comparison.png")
    return table
