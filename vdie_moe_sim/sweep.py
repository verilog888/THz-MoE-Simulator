"""Deterministic, resumable sweeps with per-run artifacts and paired statistics."""

from dataclasses import replace
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd

from . import __version__
from .config import config_from_dict, load_json, stable_digest
from .experiments import fairness_config
from .offline import audit_run, read_rows
from .scheduled_layer import effective_config, run_scheduled
from .statistics import describe, paired_observations
from .sweep_plan import compile_plan, spec_from_dict


METRICS = ("layer_makespan_s", "token_latency_p50_s", "token_latency_p95_s", "token_latency_p99_s",
           "throughput_tokens_per_s", "thz_utilization", "mean_compute_utilization", "pepd",
           "pepd_payload_only", "bandwidth_waste", "wireless_payload_bytes", "peak_concurrent_flows")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def csv_text(rows):
    return pd.DataFrame(rows).to_csv(index=False,float_format="%.17g",lineterminator="\n")


def fingerprint(plan):
    source = {p.name:p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))}
    trace_hash = None
    if plan.base.workload.routing == "trace":
        trace_hash = file_digest(plan.trace_base_dir / plan.base.workload.trace_file)
    return {"spec":plan.spec.to_dict(),"base_config":plan.base.to_dict(),"source_sha256":stable_digest(source),
            "input_trace_sha256":trace_hash,"simulator_version":__version__,"python_version":platform.python_version(),
            "dependencies":{name:version(name) for name in ("numpy","pandas","matplotlib")}}


def run_config(point, seed, series):
    return effective_config(replace(point.config,workload=replace(point.config.workload,seed=seed)),series)


def run_path(point, seed, series):
    return Path("runs") / point.point_id / f"seed_{seed}" / series


def verify_run(path, expected_config, source_sha256):
    marker = load_json(path/"completed.json")
    if marker.get("status") != "complete" or marker.get("config_sha256") != expected_config.digest or marker.get("source_sha256") != source_sha256:
        raise ValueError(f"run checkpoint configuration/source mismatch: {path}")
    files = {p.name for p in path.iterdir() if p.is_file() and p.name != "completed.json"}
    if files != set(marker["files"]):
        raise ValueError(f"run checkpoint file set differs: {path}")
    for name,digest in marker["files"].items():
        if Path(name).name != name or file_digest(path/name) != digest:
            raise ValueError(f"run artifact checksum mismatch: {path}/{name}")
    manifest = load_json(path/"manifest.json")
    if manifest["config_sha256"] != expected_config.digest or manifest["source_sha256"] != source_sha256:
        raise ValueError("run manifest differs from checkpoint")
    return audit_run(path)


def collect_group(root, plan, point, seed, source_sha256):
    rows = []
    reference = None
    max_difference = 0.0
    for series in plan.spec.series:
        config = run_config(point,seed,series)
        path = root/run_path(point,seed,series)
        audited = verify_run(path,config,source_sha256)
        outputs = np.load(path/"outputs.npy",allow_pickle=False)
        if reference is None:
            reference = outputs
        elif reference.shape != outputs.shape or not np.allclose(reference,outputs,rtol=1e-10,atol=1e-10):
            raise ValueError("matched schemes produce different numerical outputs")
        max_difference = max(max_difference,float(np.max(np.abs(reference-outputs))))
        rows.append({"point_id":point.point_id,"routing_seed":seed,"series":series,
                     "run_directory":run_path(point,seed,series).as_posix(), **audited["summary"],
                     "mean_compute_utilization":audited["mean_compute_utilization"],
                     "routing_sha256":audited["routing_sha256"],
                     "controlled_config_sha256":stable_digest(fairness_config(config))})
    for field in ("workload_sha256","routing_sha256","controlled_config_sha256"):
        if len({r[field] for r in rows}) != 1:
            raise ValueError("within-point fairness mismatch: "+field)
    return rows, {"point_id":point.point_id,"routing_seed":seed,"series_count":len(rows),
                  "same_workload":True,"same_total_phy_budget":True,"same_compute_and_timing":True,"numerically_equal":True,
                  "workload_sha256":rows[0]["workload_sha256"],"controlled_config_sha256":rows[0]["controlled_config_sha256"],
                  "max_abs_output_difference":max_difference}


def aggregate(rows, paired, plan):
    result = []
    paired_by_key = {(r["point_id"],r["routing_seed"],r["series"]):r for r in paired}
    groups = {}
    for row in rows:
        groups.setdefault((row["point_id"],row["series"]),[]).append(row)
    for point in plan.points:
        for series in plan.spec.series:
            group = sorted(groups.get((point.point_id,series),[]),key=lambda r:r["routing_seed"])
            if [r["routing_seed"] for r in group] != sorted(plan.spec.seeds):
                raise ValueError("aggregate seed coverage incomplete or duplicated")
            for metric in (*METRICS,"speedup_vs_C","latency_difference_vs_C_s"):
                values = [r[metric] if metric in METRICS else paired_by_key[r["point_id"],r["routing_seed"],series][metric] for r in group]
                row = describe(values,plan.spec.statistics,key=[point.point_id,series,metric])
                result.append({"point_id":point.point_id,"series":series,"metric":metric,
                               "n_unique_workloads":len({r["workload_sha256"] for r in group}),**row})
    return result


def check_cross_point_routes(rows, plan):
    """PHY/compute-only changes must not perturb routing RNG or assignments."""
    points = {p.point_id:p for p in plan.points}
    groups = {}
    for row in rows:
        c = points[row["point_id"]].config
        w = c.to_dict()["workload"]
        w["seed"] = row["routing_seed"]
        # Source round-robin intentionally depends on Die count.
        key = stable_digest({"workload":w,"experts":c.model.num_experts,"top_k":c.model.top_k,
                             "dies_for_source":c.topology.num_dies if w["source_policy"] == "round_robin" else None})
        digest = groups.setdefault(key,row["routing_sha256"])
        if digest != row["routing_sha256"]:
            raise ValueError("a non-routing parameter changed the route trace")


def tables_for(rows,fairness,plan):
    if len(rows) != plan.run_count:
        raise ValueError("raw result count differs from compiled plan")
    check_cross_point_routes(rows,plan)
    paired = paired_observations(rows,plan.spec.series)
    return {"raw_results.csv":rows,"paired_results.csv":paired,"aggregate.csv":aggregate(rows,paired,plan),
            "fairness.csv":fairness,
            "points.csv":[{"point_id":p.point_id,"config_sha256":p.config.digest,
                           "coordinates_json":json.dumps(p.coordinates,sort_keys=True)} for p in plan.points],
            "views.csv":[{"axis":v["axis"],"value_json":json.dumps(v["value"]),"point_id":v["point_id"],
                          "changes_json":json.dumps(v["changes"],sort_keys=True),"view_order":v["view_order"]} for v in plan.views]}


def run_sweep(plan, directory, *, resume=False, progress=None):
    root = Path(directory)
    identity = fingerprint(plan)
    signature = stable_digest(identity)
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):
        if not resume:
            raise FileExistsError("sweep output must be empty; use --resume for the identical saved plan")
        saved = load_json(root/"sweep_manifest.json")
        if saved["run_signature"] != signature:
            raise ValueError("resume rejected: plan, input, code or dependency fingerprint changed")
    elif resume:
        raise ValueError("cannot resume an empty directory")
    manifest = {**identity,"run_signature":signature,"status":"running","stage":"stage5_sweep",
                "expected_runs":plan.run_count,"preview":plan.preview(),
                "statistics_scope":"routing-seed variability; exploratory bootstrap of seed means, not hardware uncertainty",
                "parameter_points":len(plan.points)}
    write_json(root/"sweep_manifest.json",manifest)
    rows, fairness = [], []
    try:
        for point_index,point in enumerate(plan.points):
            for seed_index,seed in enumerate(plan.spec.seeds):
                if progress:
                    progress(f"Point {point_index+1}/{len(plan.points)}, seed {seed} ({len(plan.spec.series)} schemes)")
                for series in plan.spec.series:
                    config = run_config(point,seed,series)
                    output = root/run_path(point,seed,series)
                    if output.exists():
                        continue  # collect_group verifies hashes, configuration and CSV before reuse.
                    output.parent.mkdir(parents=True,exist_ok=True)
                    temporary = Path(tempfile.mkdtemp(prefix=f".pending-{series}-",dir=output.parent))
                    result = run_scheduled(config,trace_base_dir=plan.trace_base_dir)
                    result.export(temporary)
                    timeline = plan.spec.timelines == "all" or (plan.spec.timelines == "representative" and point_index == seed_index == 0)
                    if timeline:
                        from .plots import plot_scheduled_timeline
                        plot_scheduled_timeline(result,temporary/"timeline.png")
                    del result
                    marker = {"status":"complete","config_sha256":config.digest,"source_sha256":identity["source_sha256"],
                              "files":{p.name:file_digest(p) for p in sorted(temporary.iterdir()) if p.is_file()}}
                    write_json(temporary/"completed.json",marker)
                    temporary.rename(output)
                group,check = collect_group(root,plan,point,seed,identity["source_sha256"])
                rows.extend(group)
                fairness.append(check)
        for name,table in tables_for(rows,fairness,plan).items():
            (root/name).write_text(csv_text(table))
        from .sweep_plots import plot_sweep
        plot_files = plot_sweep(root,plan.spec.study,plan.spec.statistics)
        manifest["status"] = "complete"
        manifest["plot_files"] = plot_files
        manifest["completed_runs"] = len(rows)
        manifest["aggregate_files"] = {p.name:file_digest(p) for p in sorted(root.glob("*.csv"))}
        manifest["plot_sha256"] = {p:file_digest(root/p) for p in plot_files}
        write_json(root/"sweep_manifest.json",manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(root/"sweep_manifest.json",manifest)
        raise
    return plan.preview()


def audit_sweep(directory):
    root = Path(directory)
    manifest = load_json(root/"sweep_manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("sweep is incomplete; no final aggregate can be trusted")
    plan = compile_plan(spec_from_dict(manifest["spec"]),config_from_dict(manifest["base_config"]))
    rows,fairness = [],[]
    for point in plan.points:
        for seed in plan.spec.seeds:
            group,check = collect_group(root,plan,point,seed,manifest["source_sha256"])
            rows.extend(group);fairness.append(check)
    tables = tables_for(rows,fairness,plan)
    if manifest["completed_runs"] != plan.run_count or manifest["expected_runs"] != plan.run_count:
        raise ValueError("sweep manifest run count mismatch")
    for name,table in tables.items():
        if (root/name).read_text() != csv_text(table) or file_digest(root/name) != manifest["aggregate_files"][name]:
            raise ValueError("aggregate differs from independently reloaded runs: "+name)
    for name,digest in manifest["plot_sha256"].items():
        if file_digest(root/name) != digest:
            raise ValueError("plot file checksum mismatch: "+name)
    return {"status":"passed","simulation_runs":len(rows),"parameter_points":len(plan.points),
            "matched_seed_groups":len(fairness),"aggregate_rows":len(tables["aggregate.csv"]),
            "checks":["all artifact checksums","offline CSV/NPY audits","same per-point budget/resources/trace",
                      "numerical equivalence","cross-point route consistency","paired statistics recomputed from runs"]}
