"""Audits and exports for the partial pipeline; never label it full-layer latency."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from . import __version__
from .batching import BatchExecution, ExpertInput, advance_time
from .compute import estimate_expert
from .config import SimulationConfig, stable_digest
from .dispatch import plan_dispatch
from .engine import Event, PipelineEventType as E
from .metrics import ASSUMPTIONS, audit_transmissions, summarize
from .moe import Interval, TokenCompletion, Transmission
from .placement import ExpertPlacement
from .thz_phy import THzLink
from .workload import Workload, validate_workload


def _equal(actual, expected, label):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-15):
        raise ValueError(f"{label} differs from causal/model expectation")


@dataclass(frozen=True)
class PipelineResult:
    config: SimulationConfig
    workload: Workload
    inputs: tuple[ExpertInput, ...]
    executions: tuple[BatchExecution, ...]
    transmissions: tuple[Transmission, ...]
    completions: tuple[TokenCompletion, ...]  # completed_s means experts-done in this result type.
    intervals: tuple[Interval, ...]
    events: tuple[Event, ...]
    ignored_deadlines: tuple[int, ...]
    radio_context: tuple[Transmission, ...] = ()  # Stage 3 may interleave Return with Dispatch.

    def audit(self) -> None:
        config = self.config
        validate_workload(self.workload, config)
        tokens = {t.token_id: t for t in self.workload.tokens}
        placement = ExpertPlacement.contiguous(config.model.num_experts, config.topology.num_dies)
        requests = {r.request_id: r for r in self.inputs}
        if len(requests) != len(self.inputs):
            raise ValueError("duplicate request ID")
        assignments = Counter((r.token_id, r.expert_id) for r in self.inputs)
        expected = Counter((t.token_id, e) for t in tokens.values() for e in t.expert_ids)
        if assignments != expected:
            raise ValueError("missing, duplicate or unexpected expert assignment")
        completions = {c.token.token_id: c for c in self.completions}
        if len(completions) != len(self.completions) or set(completions) != set(tokens):
            raise ValueError("missing or duplicate token completion")
        if any(c.token != tokens[c.token.token_id] for c in self.completions):
            raise ValueError("completion token differs from workload")
        gate_times = {t.token_id: advance_time(advance_time(t.arrival_s, config.timing.base_to_source_s),
                                             config.timing.gate_s) for t in tokens.values()}
        by_token = {}
        expected_events = {}
        expected_intervals = []
        for token in tokens.values():
            base_end = advance_time(token.arrival_s, config.timing.base_to_source_s)
            gate_end = gate_times[token.token_id]
            for kind, time_s in ((E.TOKEN_ARRIVAL, token.arrival_s), (E.BASE_TO_SOURCE_DONE, base_end),
                                  (E.GATE_DONE, gate_end)):
                expected_events[(kind, token.token_id, None)] = time_s
            expected_intervals.extend([
                (token.token_id, "base_to_source", token.arrival_s, base_end, -1),
                (token.token_id, "gate", base_end, gate_end, token.source_die)])
        for tx in self.transmissions:
            if tx.flow.token_id not in tokens or tx.flow.token_id in by_token:
                raise ValueError("dispatch must occur at most once per token")
            token = tokens[tx.flow.token_id]
            plan = plan_dispatch(token, placement)
            if (tx.flow.flow_type != "dispatch" or tx.flow.src != token.source_die
                    or tx.flow.dsts != plan.remote_dies or not plan.remote_dies
                    or tx.flow.payload_bytes != token.activation_bytes):
                raise ValueError("broadcast destinations/payload do not match assignments")
            if tx.subcarriers != tuple(range(config.phy.num_subcarriers)):
                raise ValueError("stage 2 FIFO dispatch must use the same full bandwidth budget")
            if tx.start_s < gate_times[token.token_id]:
                raise ValueError("dispatch precedes Gate")
            by_token[token.token_id] = tx
            expected_events[(E.DISPATCH_DONE, token.token_id, tx.flow.flow_id)] = tx.end_s
            expected_intervals.append((token.token_id, "dispatch", tx.start_s, tx.end_s, -1))
        expected_order = sorted((gate_times[t.token_id], t.token_id) for t in tokens.values()
                                if plan_dispatch(t, placement).remote_dies)
        if [tx.flow.token_id for tx in self.transmissions] != [token_id for _, token_id in expected_order]:
            raise ValueError("dispatch FIFO/order or broadcast count violated")
        previous_end = 0.0
        previous_radio_end = {}
        if self.radio_context:
            audit_transmissions(self.radio_context, THzLink(config.phy), config.topology.num_dies)
            if tuple(t for t in self.radio_context if t.flow.flow_type == "dispatch") != self.transmissions:
                raise ValueError("dispatch trace differs from shared radio context")
            for transmission in self.radio_context:
                previous_radio_end[transmission.flow.flow_id] = previous_end
                previous_end = transmission.end_s
        previous_end = 0.0
        for tx in self.transmissions:
            blocking_end = previous_radio_end.get(tx.flow.flow_id, previous_end)
            _equal(tx.start_s, max(blocking_end, gate_times[tx.flow.token_id]), "dispatch work conservation")
            previous_end = tx.end_s
        for request in self.inputs:
            token = tokens[request.token_id]
            die = placement.die_for(request.expert_id)
            local = die == token.source_die
            if request.die_id != die or request.local != local:
                raise ValueError("expert placement/local-input mismatch")
            ready = gate_times[token.token_id] if local else by_token[token.token_id].end_s
            _equal(request.ready_s, ready, "expert input readiness")
        from .execution_audit import audit_batches
        batch_events, batch_intervals, token_last = audit_batches(config, self.inputs, self.executions)
        expected_events.update(batch_events)
        expected_intervals.extend(batch_intervals)
        executions = {x.batch.batch_id: x for x in self.executions}
        for token_id, completion in completions.items():
            _equal(completion.completed_s, token_last[token_id], "token slowest-expert completion")
            expected_events[(E.TOKEN_EXPERTS_DONE, token_id, None)] = completion.completed_s
        actual_intervals = [(s.token_id, s.kind, s.start_s, s.end_s, s.die_id) for s in self.intervals]
        if Counter(actual_intervals) != Counter(expected_intervals):
            raise ValueError("missing, duplicate or inconsistent busy interval")
        keys = [(e.kind, e.token_id, e.ref_id) for e in self.events]
        if Counter(keys) != Counter(expected_events.keys()):
            raise ValueError("missing, duplicate or unexpected pipeline event")
        for event in self.events:
            _equal(event.time_s, expected_events[(event.kind, event.token_id, event.ref_id)], "event time")
        if any(a.time_s > b.time_s for a, b in zip(self.events, self.events[1:])):
            raise ValueError("non-monotonic pipeline events")
        if len(set(self.ignored_deadlines)) != len(self.ignored_deadlines):
            raise ValueError("deadline ignored multiple times")
        if any(i not in executions or config.batching.window_s == 0 for i in self.ignored_deadlines):
            raise ValueError("ignored deadline is not associated with a window")
        # Includes bit conservation, subcarrier collision and per-Die compute overlap checks.
        summarize(self.completions, self.transmissions, self.intervals, THzLink(config.phy), config.topology.num_dies)

    @property
    def summary(self) -> dict:
        self.audit()
        metrics, _ = summarize(self.completions, self.transmissions, self.intervals,
                               THzLink(self.config.phy), self.config.topology.num_dies)
        # Rename the observation endpoint explicitly; no full-layer latency/throughput is emitted.
        renames = {"layer_makespan_s": "dispatch_compute_makespan_s",
                   "throughput_tokens_per_s": "experts_done_rate_tokens_per_s",
                   **{f"token_latency_p{p}_s": f"experts_done_latency_p{p}_s" for p in (50, 95, 99)}}
        metrics = {renames.get(k, k): v for k, v in metrics.items() if k != "return_payload_bytes"}
        requests = {r.request_id: r for r in self.inputs}
        waits = [x.start_s - requests[i].ready_s for x in self.executions for i in x.batch.request_ids]
        batch_waits = [x.batch.ready_s - requests[i].ready_s for x in self.executions for i in x.batch.request_ids]
        queue_waits = [x.start_s - x.batch.ready_s for x in self.executions for _ in x.batch.request_ids]
        return {
            "stage": "stage2_dispatch_compute", "endpoint": "all_routed_experts_done_before_combine_return",
            "model_status": "uncalibrated", "scheme": "broadcast_fifo_full_band_reference",
            "config_sha256": self.config.digest, "workload_sha256": self.workload.digest,
            "seed": self.config.workload.seed, "num_dies": self.config.topology.num_dies,
            "num_experts": self.config.model.num_experts, "top_k": self.config.model.top_k,
            "total_bandwidth_hz": self.config.phy.total_bandwidth_hz,
            "num_subcarriers": self.config.phy.num_subcarriers,
            "expert_assignments": len(self.inputs), "local_assignments": sum(r.local for r in self.inputs),
            "remote_assignments": sum(not r.local for r in self.inputs), "dispatch_flows": len(self.transmissions),
            "compute_batches": len(self.executions), "weight_reads": sum(x.weight_reads for x in self.executions),
            "local_weight_bytes_read": sum(x.weight_reads * x.weight_bytes for x in self.executions),
            "mean_batch_size": len(self.inputs) / len(self.executions),
            "mean_input_to_compute_wait_s": float(np.mean(waits)),
            "mean_batch_formation_wait_s": float(np.mean(batch_waits)),
            "mean_closed_batch_queue_wait_s": float(np.mean(queue_waits)),
            "ignored_batch_deadlines": len(self.ignored_deadlines), **metrics,
        }

    def export(self, directory: str | Path) -> Path:
        summary = self.summary
        _, compute_rows = summarize(self.completions, self.transmissions, self.intervals,
                                    THzLink(self.config.phy), self.config.topology.num_dies)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError("output directory must be empty; choose a new run directory")

        def csv(name, rows, columns=None):
            pd.DataFrame(rows, columns=columns).to_csv(directory / name, index=False, float_format="%.17g", lineterminator="\n")

        csv("summary.csv", [summary])
        csv("tokens.csv", [{**asdict(c.token), "expert_ids": json.dumps(c.token.expert_ids),
                            "gate_weights": json.dumps(c.token.gate_weights), "experts_done_s": c.completed_s,
                            "experts_done_latency_s": c.completed_s - c.token.arrival_s} for c in self.completions])
        gate_times = {e.token_id: e.time_s for e in self.events if e.kind == E.GATE_DONE}
        csv("flows.csv", [{**asdict(t.flow), "dsts": json.dumps(t.flow.dsts), "ready_s": gate_times[t.flow.token_id],
                           "queue_wait_s": t.start_s - gate_times[t.flow.token_id], "start_s": t.start_s,
                           "payload_start_s": t.payload_start_s, "end_s": t.end_s,
                           "subcarriers": json.dumps(t.subcarriers), "num_subcarriers": len(t.subcarriers),
                           "rate_bps": t.rate_bps} for t in self.transmissions],
            columns=["flow_id", "token_id", "src", "dsts", "payload_bytes", "flow_type", "ready_s", "queue_wait_s",
                     "start_s", "payload_start_s", "end_s", "subcarriers", "num_subcarriers", "rate_bps"])
        by_request = {i: x for x in self.executions for i in x.batch.request_ids}
        csv("requests.csv", [{**asdict(r), "batch_id": by_request[r.request_id].batch.batch_id,
                              "batch_ready_s": by_request[r.request_id].batch.ready_s,
                              "compute_start_s": by_request[r.request_id].start_s,
                              "compute_end_s": by_request[r.request_id].end_s,
                              "batch_wait_s": by_request[r.request_id].batch.ready_s - r.ready_s,
                              "queue_wait_s": by_request[r.request_id].start_s - by_request[r.request_id].batch.ready_s}
                             for r in self.inputs])
        requests = {r.request_id: r for r in self.inputs}
        csv("batches.csv", [{**asdict(x.batch), "request_ids": json.dumps(x.batch.request_ids),
                             "token_ids": json.dumps([requests[i].token_id for i in x.batch.request_ids]),
                             "num_tokens": len(x.batch.request_ids), "start_s": x.start_s, "end_s": x.end_s,
                             "flops": x.flops, "weight_bytes": x.weight_bytes, "weight_reads": x.weight_reads,
                             "compute_s": x.compute_s, "memory_s": x.memory_s} for x in self.executions])
        csv("events.csv", [{"sequence": i, "time_s": e.time_s, "event_type": e.kind.value, "token_id": e.token_id,
                            "ref_id": e.ref_id, "ignored": e.kind == E.BATCH_DEADLINE and e.ref_id in self.ignored_deadlines}
                           for i, e in enumerate(self.events)])
        csv("intervals.csv", [asdict(i) for i in self.intervals])
        csv("compute_utilization.csv", compute_rows)
        assumptions = [a for a in ASSUMPTIONS if a["id"] not in ("A02", "A03", "A05", "A07", "A13", "A14")]
        assumptions += [
            {"id": "A02/A03", "status": "uncalibrated_reference", "model": "independent unlimited Base/Gate front-end delay per token, outside THz and expert compute resources"},
            {"id": "A05", "status": "user_accepted_reference_not_hardware_validation", "model": "half-open per-expert batch windows with early cap; closed-batch FIFO per shared Die; no preemption"},
            {"id": "A07", "status": "explicit_workload", "model": "weighted sampling without replacement; synthetic equal weights; trace weights retained unchanged; local inputs bypass RF"},
            {"id": "A13", "status": "partial_pipeline_metric", "model": "window earliest arrival to last expert completion; one busy interval per batch; not a full MoE layer"},
            {"id": "A14", "status": "stage2_limit", "model": "no numeric FFN or Local/Source Combine; no Return; FIFO full-band dispatch only"},
        ]
        package_dir = Path(__file__).parent
        source = {p.name: p.read_text(encoding="utf-8") for p in sorted(package_dir.glob("*.py"))}
        manifest = {
            "schema_version": 1, "simulator_version": __version__, "stage": "stage2_dispatch_compute",
            "endpoint": summary["endpoint"], "model_status": "uncalibrated; partial pipeline timing validation",
            "config": self.config.to_dict(), "config_sha256": self.config.digest,
            "workload_sha256": self.workload.digest, "input_trace_sha256": self.workload.trace_sha256,
            "source_sha256": stable_digest(source), "python_version": platform.python_version(),
            "dependencies": {name: version(name) for name in ("numpy", "pandas", "matplotlib")},
            "specification_sha256": "072f0287b6b79494564d0190f3143b348642f3328aaf9d5168eb385b29f40fdf",
            "assumptions": assumptions,
            "not_implemented": ["Local Combine", "Return", "Source Combine", "shared expert", "static OFDMA",
                                "CA-OFDMA", "five-scheme comparison", "parameter sweeps"],
        }
        for filename, data in (("manifest.json", manifest), ("routing_trace.json", self.workload.to_trace())):
            (directory / filename).write_text(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                                        indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return directory
