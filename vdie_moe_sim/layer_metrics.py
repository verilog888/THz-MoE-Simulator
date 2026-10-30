"""Independent dependency, numerical and resource audits for Stage 3."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .batching import advance_time
from .combine import CombineJob, FlowReady, PartialDelivery, PartialVector, TokenOutput, synthetic_expert_output
from .engine import Event, LayerEventType as L, PipelineEventType as E
from .metrics import summarize
from .moe import Interval, TokenCompletion, Transmission
from .pipeline_metrics import PipelineResult, _equal
from .placement import ExpertPlacement
from .thz_phy import THzLink


def vector_digest(vector) -> str:
    return hashlib.sha256(np.asarray(vector, dtype="<f8").tobytes()).hexdigest()


def centralized_oracle(token, expert_ids, dimension, seed):
    """One matrix contraction, independent of the event-path hierarchical reductions."""
    ids = tuple(sorted(expert_ids))
    weight_map = dict(zip(token.expert_ids, token.gate_weights))
    matrix = np.stack([synthetic_expert_output(token.token_id, e, dimension, seed) for e in ids])
    weights = np.array([weight_map[e] for e in ids], dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        return np.einsum("i,ij->j", weights, matrix)


def check_numeric(actual, expected, label):
    actual = np.asarray(actual)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"{label}: invalid vector shape or nonfinite value")
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{label}: does not match independent weighted-sum oracle")
    return float(np.max(np.abs(actual - expected)))


@dataclass(frozen=True)
class LayerResult:
    front: PipelineResult
    transmissions: tuple[Transmission, ...]
    flow_ready: tuple[FlowReady, ...]
    jobs: tuple[CombineJob, ...]
    partials: tuple[PartialVector, ...]
    deliveries: tuple[PartialDelivery, ...]
    outputs: tuple[TokenOutput, ...]
    completions: tuple[TokenCompletion, ...]
    intervals: tuple[Interval, ...]
    events: tuple[Event, ...]

    @property
    def config(self):
        return self.front.config

    @property
    def workload(self):
        return self.front.workload

    @property
    def executions(self):
        return self.front.executions

    def audit(self) -> list[dict]:
        self.front.audit()
        config = self.config
        tokens = {t.token_id: t for t in self.workload.tokens}
        placement = ExpertPlacement.contiguous(config.model.num_experts, config.topology.num_dies)
        if self.front.radio_context != self.transmissions:
            raise ValueError("front-end did not observe the complete shared RF queue")
        if tuple(e for e in self.events if isinstance(e.kind, E)) != self.front.events:
            raise ValueError("front-end events differ from layer events")
        requests = {r.request_id: r for r in self.front.inputs}
        expert_done = {(requests[i].token_id, requests[i].expert_id): x.end_s
                       for x in self.executions for i in x.batch.request_ids}
        group_ids = {t.token_id: {die: tuple(sorted(ids)) for die, ids in placement.group(t.expert_ids).items()}
                     for t in tokens.values()}
        expected_keys = set()
        for token in tokens.values():
            for expert in token.expert_ids:
                expected_keys.add(("weight", token.token_id, placement.die_for(expert), (expert,)))
            if config.combine.local_combine:
                for die, ids in group_ids[token.token_id].items():
                    if len(ids) > 1:
                        expected_keys.add(("local_reduce", token.token_id, die, ids))
            expected_keys.add(("source_reduce", token.token_id, token.source_die, tuple(sorted(token.expert_ids))))
        by_key = {(j.kind, j.token_id, j.die_id, j.expert_ids): j for j in self.jobs}
        if len(by_key) != len(self.jobs) or set(by_key) != expected_keys:
            raise ValueError("missing, duplicate or unexpected Combine job")
        if sorted(j.job_id for j in self.jobs) != list(range(len(self.jobs))):
            raise ValueError("Combine job IDs are not unique/complete")
        weight_jobs = {(j.token_id, j.expert_ids[0]): j for j in self.jobs if j.kind == "weight"}
        local_jobs = {(j.token_id, j.die_id): j for j in self.jobs if j.kind == "local_reduce"}
        source_jobs = {j.token_id: j for j in self.jobs if j.kind == "source_reduce"}
        partials = {p.partial_id: p for p in self.partials}
        expected_parts = []
        for token in tokens.values():
            if config.combine.local_combine:
                expected_parts.extend((token.token_id, die, ids) for die, ids in group_ids[token.token_id].items())
            else:
                expected_parts.extend((token.token_id, placement.die_for(e), (e,)) for e in token.expert_ids)
        if (len(partials) != len(self.partials)
                or Counter((p.token_id, p.die_id, p.expert_ids) for p in self.partials) != Counter(expected_parts)):
            raise ValueError("partial result partition does not cover each expert exactly once")
        expected_events = {(e.kind, e.token_id, e.ref_id): e.time_s for e in self.front.events}
        expected_intervals = list(self.front.intervals)
        for partial in self.partials:
            if config.combine.local_combine and len(partial.expert_ids) > 1:
                ready = local_jobs[(partial.token_id, partial.die_id)].end_s
            else:
                ready = weight_jobs[(partial.token_id, partial.expert_ids[0])].end_s
            _equal(partial.ready_s, ready, "partial readiness")
            check_numeric(partial.vector, centralized_oracle(tokens[partial.token_id], partial.expert_ids,
                          config.model.d_model, config.combine.verification_seed), "local weighted partial")
            expected_events[(L.PARTIAL_READY, partial.token_id, partial.partial_id)] = partial.ready_s
        deliveries = {d.partial_id: d for d in self.deliveries}
        if len(deliveries) != len(self.deliveries) or set(deliveries) != set(partials):
            raise ValueError("missing or duplicate partial delivery")
        ready_by_id = {r.flow_id: r for r in self.flow_ready}
        flows = {t.flow.flow_id: t for t in self.transmissions}
        if (len(ready_by_id) != len(self.flow_ready) or set(ready_by_id) != set(flows)
                or [t.flow.flow_id for t in self.transmissions] != list(range(len(self.transmissions)))):
            raise ValueError("shared RF FIFO IDs/order or readiness records invalid")
        expected_return_ids = set()
        delivered_by_token = defaultdict(list)
        for partial in self.partials:
            delivery = deliveries[partial.partial_id]
            token = tokens[partial.token_id]
            if partial.die_id == token.source_die:
                if delivery.flow_id is not None:
                    raise ValueError("Source-local partial must not use RF")
                _equal(delivery.available_s, partial.ready_s, "local delivery")
            else:
                if delivery.flow_id not in flows or delivery.flow_id in expected_return_ids:
                    raise ValueError("missing or shared Return flow")
                tx = flows[delivery.flow_id]
                if (tx.flow.flow_type != "return" or tx.flow.token_id != token.token_id
                        or tx.flow.src != partial.die_id or tx.flow.dsts != (token.source_die,)
                        or tx.flow.payload_bytes != token.activation_bytes):
                    raise ValueError("Return endpoint/token/payload mismatch")
                ready = ready_by_id[delivery.flow_id]
                if ready.partial_id != partial.partial_id:
                    raise ValueError("Return associated with wrong partial")
                _equal(ready.ready_s, partial.ready_s, "Return readiness")
                _equal(delivery.available_s, tx.end_s, "remote delivery")
                expected_return_ids.add(delivery.flow_id)
                expected_events[(L.RETURN_DONE, token.token_id, tx.flow.flow_id)] = tx.end_s
                expected_intervals.append(Interval(token.token_id, "return", "THz shared spectrum", tx.start_s, tx.end_s))
            delivered_by_token[token.token_id].append(delivery.available_s)
        if expected_return_ids != {t.flow.flow_id for t in self.transmissions if t.flow.flow_type == "return"}:
            raise ValueError("unexpected Return traffic")
        gate_times = {e.token_id: e.time_s for e in self.front.events if e.kind == E.GATE_DONE}
        previous_end, previous_ready = 0.0, 0.0
        for tx in self.transmissions:
            ready = ready_by_id[tx.flow.flow_id]
            if tx.flow.flow_type == "dispatch":
                if ready.partial_id is not None:
                    raise ValueError("Dispatch cannot carry a returned partial")
                _equal(ready.ready_s, gate_times[tx.flow.token_id], "Dispatch readiness")
            if ready.ready_s < previous_ready:
                raise ValueError("RF FIFO reordered ready flows")
            _equal(tx.start_s, max(previous_end, ready.ready_s), "shared RF work conservation")
            if tx.subcarriers != tuple(range(config.phy.num_subcarriers)):
                raise ValueError("stage 3 FIFO must share the single full bandwidth budget")
            previous_end, previous_ready = tx.end_s, ready.ready_s
        jobs_by_die = defaultdict(list)
        durations = {"weight": config.combine.weight_s, "local_reduce": config.combine.local_reduce_s,
                     "source_reduce": config.timing.source_combine_s}
        event_kinds = {"weight": L.WEIGHT_DONE, "local_reduce": L.LOCAL_REDUCE_DONE,
                       "source_reduce": L.SOURCE_COMBINE_DONE}
        for job in self.jobs:
            if job.kind == "weight":
                ready = expert_done[(job.token_id, job.expert_ids[0])]
            elif job.kind == "local_reduce":
                ready = max(weight_jobs[(job.token_id, e)].end_s for e in job.expert_ids)
            else:
                ready = max(delivered_by_token[job.token_id])
            _equal(job.ready_s, ready, f"{job.kind} dependency")
            if job.start_s < job.ready_s:
                raise ValueError("Combine started before its inputs were ready")
            _equal(job.end_s, advance_time(job.start_s, durations[job.kind]), "Combine duration")
            expected_events[(event_kinds[job.kind], job.token_id, job.job_id)] = job.end_s
            expected_intervals.append(Interval(job.token_id, job.kind, f"Combine die {job.die_id}",
                                               job.start_s, job.end_s, job.die_id))
            jobs_by_die[job.die_id].append(job)
        for jobs in jobs_by_die.values():
            previous_end = 0.0
            for job in sorted(jobs, key=lambda j: (j.ready_s, j.job_id)):
                start = max(job.ready_s, previous_end) if config.combine.resource_model == "per_die_fifo" else job.ready_s
                _equal(job.start_s, start, "Combine FIFO/resource arbitration")
                previous_end = job.end_s
        completions = {c.token.token_id: c for c in self.completions}
        outputs = {o.token_id: o for o in self.outputs}
        if (len(completions) != len(self.completions) or len(outputs) != len(self.outputs)
                or set(completions) != set(tokens) or set(outputs) != set(tokens)):
            raise ValueError("missing or duplicate final completion/output")
        validation_rows = []
        for row, token_id in enumerate(sorted(tokens)):
            token, output, completion = tokens[token_id], outputs[token_id], completions[token_id]
            if completion.token != token:
                raise ValueError("final token differs from workload")
            source_end = source_jobs[token_id].end_s
            _equal(output.ready_s, source_end, "Source output readiness")
            base_end = advance_time(source_end, config.timing.source_to_base_s)
            _equal(completion.completed_s, base_end, "final Base completion")
            expected_events[(L.SOURCE_TO_BASE_DONE, token_id, None)] = base_end
            expected_intervals.append(Interval(token_id, "source_to_base", "Source to Base", source_end, base_end))
            oracle = centralized_oracle(token, token.expert_ids, config.model.d_model, config.combine.verification_seed)
            error = check_numeric(output.vector, oracle, "final weighted sum")
            validation_rows.append({"output_row": row, "token_id": token_id, "max_abs_error": error,
                                    "reference_max_abs": float(np.max(np.abs(oracle))),
                                    "vector_sha256": vector_digest(output.vector)})
        if Counter(self.intervals) != Counter(expected_intervals):
            raise ValueError("missing, duplicated or inconsistent layer intervals")
        keys = [(e.kind, e.token_id, e.ref_id) for e in self.events]
        if Counter(keys) != Counter(expected_events.keys()):
            raise ValueError("missing, duplicate or unexpected layer event")
        for event in self.events:
            _equal(event.time_s, expected_events[(event.kind, event.token_id, event.ref_id)], "layer event")
        if any(a.time_s > b.time_s for a, b in zip(self.events, self.events[1:])):
            raise ValueError("non-monotonic layer events")
        summarize(self.completions, self.transmissions, self.intervals, THzLink(config.phy), config.topology.num_dies)
        return validation_rows

    @property
    def summary(self):
        validation = self.audit()
        raw, _ = summarize(self.completions, self.transmissions, self.intervals,
                           THzLink(self.config.phy), self.config.topology.num_dies)
        remote_assignments = sum(not r.local for r in self.front.inputs)
        return {"stage": "stage3_end_to_end", "endpoint": "all_tokens_returned_to_base",
                "model_status": "uncalibrated", "scheme": "broadcast_shared_fifo_reference",
                "local_combine": self.config.combine.local_combine,
                "combine_resource_model": self.config.combine.resource_model,
                "config_sha256": self.config.digest, "workload_sha256": self.workload.digest,
                "seed": self.config.workload.seed, "num_dies": self.config.topology.num_dies,
                "num_experts": self.config.model.num_experts, "top_k": self.config.model.top_k,
                "total_bandwidth_hz": self.config.phy.total_bandwidth_hz,
                "num_subcarriers": self.config.phy.num_subcarriers,
                "expert_assignments": len(self.front.inputs), "compute_batches": len(self.executions),
                "local_assignments": len(self.front.inputs) - remote_assignments,
                "remote_assignments": remote_assignments, "partial_vectors": len(self.partials),
                "dispatch_flows": sum(t.flow.flow_type == "dispatch" for t in self.transmissions),
                "return_flows": sum(t.flow.flow_type == "return" for t in self.transmissions),
                "return_bytes_saved_by_local_combine": remote_assignments * self.config.model.activation_bytes - raw["return_payload_bytes"],
                "combine_jobs": len(self.jobs), "mean_combine_queue_wait_s": float(np.mean([j.start_s - j.ready_s for j in self.jobs])),
                "verification_max_abs_error": max(r["max_abs_error"] for r in validation), **raw}

    def combine_utilization(self):
        origin = min(c.token.arrival_s for c in self.completions)
        horizon = max(c.completed_s for c in self.completions) - origin
        rows = []
        for die in range(self.config.topology.num_dies):
            spans = sorted((j.start_s, j.end_s) for j in self.jobs if j.die_id == die)
            occupied, end = 0.0, origin
            for start, stop in spans:
                occupied += max(0, stop - max(start, end))
                end = max(end, stop)
            active_s = sum(stop - start for start, stop in spans)
            rows.append({"die_id": die, "occupied_s": occupied, "utilization": occupied / horizon,
                         "active_job_s": active_s, "mean_active_jobs": active_s / horizon})
        return rows

    def export(self, directory: str | Path):
        summary = self.summary
        validation = self.audit()
        # Reuse audited request/batch/trace exports, then replace endpoint-specific tables.
        directory = self.front.export(directory)

        def csv(name, rows, columns=None):
            pd.DataFrame(rows, columns=columns).to_csv(directory / name, index=False, float_format="%.17g", lineterminator="\n")

        csv("summary.csv", [summary])
        experts_done = {c.token.token_id: c.completed_s for c in self.front.completions}
        output_by_id = {o.token_id: o for o in self.outputs}
        csv("tokens.csv", [{**asdict(c.token), "expert_ids": json.dumps(c.token.expert_ids),
                            "gate_weights": json.dumps(c.token.gate_weights), "experts_done_s": experts_done[c.token.token_id],
                            "source_combined_s": output_by_id[c.token.token_id].ready_s,
                            "completed_s": c.completed_s, "latency_s": c.completed_s - c.token.arrival_s}
                           for c in self.completions])
        ready = {r.flow_id: r for r in self.flow_ready}
        csv("flows.csv", [{**asdict(t.flow), "dsts": json.dumps(t.flow.dsts), "ready_s": ready[t.flow.flow_id].ready_s,
                           "partial_id": ready[t.flow.flow_id].partial_id, "queue_wait_s": t.start_s - ready[t.flow.flow_id].ready_s,
                           "start_s": t.start_s, "payload_start_s": t.payload_start_s, "end_s": t.end_s,
                           "subcarriers": json.dumps(t.subcarriers), "num_subcarriers": len(t.subcarriers), "rate_bps": t.rate_bps}
                          for t in self.transmissions],
            columns=["flow_id", "token_id", "src", "dsts", "payload_bytes", "flow_type", "ready_s", "partial_id", "queue_wait_s",
                     "start_s", "payload_start_s", "end_s", "subcarriers", "num_subcarriers", "rate_bps"])
        csv("combine_jobs.csv", [{**asdict(j), "expert_ids": json.dumps(j.expert_ids), "queue_wait_s": j.start_s - j.ready_s}
                                 for j in self.jobs])
        deliveries = {d.partial_id: d for d in self.deliveries}
        csv("partials.csv", [{"partial_id": p.partial_id, "token_id": p.token_id, "die_id": p.die_id,
                              "expert_ids": json.dumps(p.expert_ids), "ready_s": p.ready_s,
                              "available_at_source_s": deliveries[p.partial_id].available_s,
                              "return_flow_id": deliveries[p.partial_id].flow_id, "vector_sha256": vector_digest(p.vector)}
                             for p in self.partials])
        csv("events.csv", [{"sequence": i, "time_s": e.time_s, "event_type": e.kind.value, "token_id": e.token_id,
                            "ref_id": e.ref_id, "ignored": e.kind == E.BATCH_DEADLINE and e.ref_id in self.front.ignored_deadlines}
                           for i, e in enumerate(self.events)])
        csv("intervals.csv", [asdict(i) for i in self.intervals])
        _, compute_rows = summarize(self.completions, self.transmissions, self.intervals,
                                    THzLink(self.config.phy), self.config.topology.num_dies)
        csv("compute_utilization.csv", compute_rows)
        csv("combine_utilization.csv", self.combine_utilization())
        csv("output_validation.csv", validation)
        np.save(directory / "outputs.npy", np.stack([output_by_id[i].vector for i in sorted(output_by_id)]), allow_pickle=False)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update({"stage": "stage3_end_to_end", "endpoint": summary["endpoint"],
                         "model_status": "uncalibrated; synthetic float64 weighted-combine verification",
                         "not_implemented": ["real FFN", "BF16/FP8 numerical quantization", "shared expert",
                                             "static OFDMA", "CA-OFDMA", "five-scheme comparison", "parameter sweeps"],
                         "numerical_verification": {"generator": "assignment-local NumPy SeedSequence uniform(-1,1)",
                             "seed": self.config.combine.verification_seed, "dtype": "float64", "dimension": self.config.model.d_model,
                             "rtol": 1e-10, "atol": 1e-10, "max_abs_error": summary["verification_max_abs_error"],
                             "output_row_mapping": "output_validation.csv"}})
        manifest["assumptions"] = [a for a in manifest["assumptions"] if a["id"] not in ("A02/A03", "A13", "A14")]
        manifest["assumptions"] += [
            {"id": "A02", "status": "uncalibrated_reference", "model": "independent fixed Base/Gate delays outside THz/expert/Combine resources"},
            {"id": "A03", "status": "user_selected_reference_not_hardware_validation", "model": self.config.combine.resource_model,
             "details": "weight/local/source reductions share one FIFO unit per Die in default mode, separate from expert compute; delays configured"},
            {"id": "A13", "status": "end_to_end_metric", "model": "window earliest arrival to final Base completion; Dispatch and Return share one RF budget"},
            {"id": "A14", "status": "synthetic_semantic_check", "model": "float64 weighted vectors; activation precision affects byte counts only; no real FFN or quantization"},
        ]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return directory
