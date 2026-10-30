"""Audit exported CSV/NPY without running the event simulator again."""

from dataclasses import fields
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .batching import BatchExecution, ClosedBatch, ExpertInput
from .config import config_from_dict, load_json, stable_digest
from .execution_audit import audit_batches
from .layer_metrics import centralized_oracle, check_numeric, vector_digest
from .moe import Flow, Token
from .pipeline_metrics import _equal
from .radio import RadioSegment
from .radio_audit import RadioFlow, audit_radio
from .workload import Workload, validate_workload


def read_rows(path):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False).to_dict("records")


def optional_int(value):
    if value == "":
        return None
    converted = int(value)
    if float(value) != converted:
        raise ValueError("invalid optional integer")
    return converted


def audit_run(directory):
    directory = Path(directory)
    manifest = load_json(directory / "manifest.json")
    config = config_from_dict(manifest["config"])
    if config.digest != manifest["config_sha256"]:
        raise ValueError("manifest configuration hash mismatch")
    rows = read_rows(directory / "summary.csv")
    if len(rows) != 1:
        raise ValueError("each run must export exactly one summary")
    summary = rows[0]
    token_rows = read_rows(directory / "tokens.csv")
    csv_tokens = [Token(r["token_id"], r["source_die"], r["arrival_s"], r["activation_bytes"],
                        tuple(json.loads(r["expert_ids"])), tuple(json.loads(r["gate_weights"]))) for r in token_rows]
    trace = load_json(directory/"routing_trace.json")
    tokens = [Token(r["token_id"],r["source_die"],r["arrival_s"],config.model.activation_bytes,
                    tuple(r["expert_ids"]),tuple(r["gate_weights"])) for r in trace["tokens"]]
    if sorted(csv_tokens,key=lambda t:t.token_id) != sorted(tokens,key=lambda t:t.token_id):
        raise ValueError("routing trace differs from exported tokens")
    # JSON preserves 0.0 versus 0 for provenance; CSV intentionally does not.
    workload = Workload(tuple(sorted(tokens, key=lambda t: (t.arrival_s,t.token_id))))
    validate_workload(workload,config)
    if workload.digest != manifest["workload_sha256"] or summary["workload_sha256"] != workload.digest:
        raise ValueError("workload hash differs from exported tokens")
    if stable_digest(workload.to_trace()) != stable_digest(trace):
        raise ValueError("routing trace differs from exported tokens")
    if summary["config_sha256"] != config.digest:
        raise ValueError("summary configuration hash mismatch")
    requests = tuple(ExpertInput(**{f.name:r[f.name] for f in fields(ExpertInput)}) for r in read_rows(directory/"requests.csv"))
    executions = []
    for row in read_rows(directory/"batches.csv"):
        row["request_ids"] = tuple(json.loads(row["request_ids"]))
        batch = ClosedBatch(**{f.name:row[f.name] for f in fields(ClosedBatch)})
        executions.append(BatchExecution(batch, **{f.name:row[f.name] for f in fields(BatchExecution) if f.name != "batch"}))
    _, _, last_expert = audit_batches(config,requests,executions)
    if len(requests) != len(tokens)*config.model.top_k:
        raise ValueError("incorrect number of expert requests")
    flows = []
    for row in read_rows(directory/"flows.csv"):
        row["dsts"] = tuple(json.loads(row["dsts"]))
        flow = Flow(**{f.name:row[f.name] for f in fields(Flow)})
        flows.append(RadioFlow(flow, row["ready_s"],row["admitted_s"],row["completed_s"],
                               tuple(json.loads(row["expert_ids"])),optional_int(row["partial_id"]),optional_int(row["parent_id"])))
    segments = []
    for row in read_rows(directory/"radio_segments.csv"):
        row["subcarriers"] = tuple(json.loads(row["subcarriers"]))
        segments.append(RadioSegment(**row))
    decisions = read_rows(directory/"allocations.csv")
    for row in decisions:
        row["subcarriers"] = tuple(json.loads(row["subcarriers"]))
    radio = audit_radio(flows,segments,decisions,config,executions)
    completed = {r["token_id"]:r["completed_s"] for r in token_rows}
    horizon = max(completed.values()) - min(t.arrival_s for t in tokens)
    if horizon <= 0:
        raise ValueError("invalid observation horizon")
    latencies = []
    for r in token_rows:
        _equal(r["experts_done_s"],last_expert[r["token_id"]],"saved expert completion")
        _equal(r["completed_s"],r["source_combined_s"]+config.timing.source_to_base_s,"saved Base completion")
        latency = r["completed_s"]-r["arrival_s"]
        _equal(r["latency_s"],latency,"saved token latency")
        latencies.append(latency)
    for f in flows:
        token = next(t for t in tokens if t.token_id == f.flow.token_id)
        if f.ready_s < token.arrival_s or f.completed_s > completed[token.token_id]:
            raise ValueError("saved flow outside token lifetime")
    compute_rows = read_rows(directory/"compute_utilization.csv")
    if [r["die_id"] for r in compute_rows] != list(range(config.topology.num_dies)):
        raise ValueError("compute utilization misses/duplicates a Die")
    busy = []
    for die,row in enumerate(compute_rows):
        duration = math.fsum(x.end_s-x.start_s for x in executions if x.batch.die_id == die)
        _equal(row["busy_s"],duration,"offline compute busy time")
        _equal(row["utilization"],duration/horizon,"offline compute utilization")
        busy.append(duration/horizon)
    payload_bytes = sum(f.flow.payload_bytes for f in flows)
    capacity = radio["allocated_capacity_bits"]
    computed = {"layer_makespan_s":horizon,"num_tokens":len(tokens),"throughput_tokens_per_s":len(tokens)/horizon,
                "wireless_payload_bytes":payload_bytes,"thz_utilization":radio["occupied_subcarrier_seconds"]/(config.phy.num_subcarriers*horizon),
                "pepd":radio["active_flow_seconds"]/horizon,"pepd_payload_only":radio["payload_flow_seconds"]/horizon,
                "bandwidth_waste":max(0,1-payload_bytes*8/capacity) if capacity else 0,
                "dispatch_payload_bytes":sum(f.flow.payload_bytes for f in flows if f.flow.flow_type.startswith("dispatch")),
                "return_payload_bytes":sum(f.flow.payload_bytes for f in flows if f.flow.flow_type.startswith("return")), **radio}
    for p in (50,95,99):
        computed[f"token_latency_p{p}_s"] = float(np.percentile(latencies,p,method="linear"))
    for key,value in computed.items():
        _equal(summary[key],value,"offline summary "+key)
    outputs = np.load(directory/"outputs.npy",allow_pickle=False)
    validation = read_rows(directory/"output_validation.csv")
    if outputs.shape != (len(tokens),config.model.d_model) or len(validation) != len(tokens):
        raise ValueError("invalid output array or validation row count")
    for row,token in enumerate(sorted(tokens,key=lambda t:t.token_id)):
        v = validation[row]
        if v["output_row"] != row or v["token_id"] != token.token_id or v["vector_sha256"] != vector_digest(outputs[row]):
            raise ValueError("output row/hash mismatch")
        error = check_numeric(outputs[row],centralized_oracle(token,token.expert_ids,config.model.d_model,config.combine.verification_seed),"saved final output")
        _equal(v["max_abs_error"],error,"saved numeric error")
    _equal(summary["verification_max_abs_error"],max(v["max_abs_error"] for v in validation),"saved maximum numeric error")
    return {"status":"passed","summary":summary,"config":config,
            "mean_compute_utilization":float(np.mean(busy)), "routing_sha256":stable_digest(workload.to_trace()),
            "checks":["CSV bit/spectrum integrals","FIFO admission","CA allocation features","batch/compute audit",
                      "latency/percentiles/throughput","compute utilization","output array weighted-sum oracle"]}
