"""Full-layer dataflow audit and integral metrics for the five controlled schemes."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from . import __version__
from .batching import advance_time
from .config import stable_digest, uses_broadcast, scheme_label
from .engine import PipelineEventType as E, LayerEventType as L, RadioEventType as R
from .execution_audit import audit_batches
from .layer_metrics import centralized_oracle, check_numeric, vector_digest
from .metrics import summarize
from .pipeline_metrics import _equal
from .placement import ExpertPlacement
from .radio_audit import RadioFlow, audit_radio
from .thz_phy import THzLink
from .workload import validate_workload


STAGE4_ASSUMPTIONS = [
    {"id": "A01/A04/A14", "status": "uncalibrated_reference", "model": "routed experts only; synthetic float64 vectors; no real FFN, quantization, channel errors or hardware calibration"},
    {"id": "A02/A03", "status": "user_selected_reference", "model": "independent fixed Base/Gate delays; per-Die FIFO weight/local/source Combine separate from expert compute unless explicitly configured independent"},
    {"id": "A05", "status": "user_selected_reference", "model": "half-open expert batch windows, early cap, shared Die compute FIFO"},
    {"id": "A06", "status": "user_selected_ideal_radio", "model": "multiple TX/RX per endpoint on globally disjoint subcarriers; no spatial reuse or power/SNR coupling"},
    {"id": "A08/A11", "status": "user_selected_reference", "model": "one spectrum-occupying setup per hop; no setup restart or extra retuning cost; event-driven reallocation; CA refresh interval capped by epoch_s"},
    {"id": "A09", "status": "explicit_reference_parameterization", "model": "Q=remaining bytes/q_ref; W=age since ready/w_ref including active service; I=fraction of destination Dies with neither running nor closed queued batches (dispatch only); C=1 for Return; coefficients configurable, default all 1"},
    {"id": "A10", "status": "user_selected_reference", "model": "FIFO non-preemptive admission up to Nsc flows; static uses floor(Nsc/F) and leaves remainder idle; CA uses paper max(1,floor(quota)) then largest remainder, repairing overflow with one-per-flow reservation; all-zero priority falls back to equal"},
    {"id": "A12", "status": "user_selected_relay_reference", "model": "A per-remote-expert Source->virtual Base->expert and expert->Base->Source; each hop charged to same THz budget, Base pure forwarding; B per-expert direct P2P; C/D/Proposed deduplicated broadcast plus Local Combine; local hits bypass radio in all schemes"},
    {"id": "A13", "status": "integral_metric", "model": "earliest arrival to final Base; radio occupancy and PEPD integrate setup and payload segments; capacity uses net PHY rate; bytes counted once per hop"},
]


@dataclass(frozen=True)
class ScheduledResult:
    config: object
    workload: object
    inputs: tuple
    executions: tuple
    flows: tuple
    segments: tuple
    decisions: tuple
    jobs: tuple
    partials: tuple
    deliveries: tuple
    outputs: tuple
    experts_completions: tuple
    completions: tuple
    intervals: tuple
    events: tuple
    ignored_deadlines: tuple
    ignored_radio: tuple

    @classmethod
    def from_simulation(cls, s):
        flows = tuple(RadioFlow(x.flow, x.ready_s, x.admitted_s, x.completed_s, x.expert_ids, x.partial_id, x.parent_id)
                      for x in s.radio.states.values())
        ordered = lambda mapping: tuple(mapping[i] for i in sorted(mapping))
        return cls(s.config, s.workload, tuple(s.inputs), tuple(s.executions), flows, tuple(s.radio.segments),
                   tuple(dict(d) for d in s.radio.decisions), tuple(s.jobs), tuple(s.partials), ordered(s.deliveries),
                   ordered(s.outputs), ordered(s.experts_completions), ordered(s.completions), tuple(s.intervals),
                   tuple(s.events), tuple(s.ignored_deadlines), tuple(s.radio.ignored))

    def audit(self):
        c = self.config
        validate_workload(self.workload, c)
        tokens = {t.token_id: t for t in self.workload.tokens}
        placement = ExpertPlacement.contiguous(c.model.num_experts, c.topology.num_dies)
        relay, p2p = c.scheduling.scheme == "A", not uses_broadcast(c)
        expected_local = c.ablation.local_combine if c.ablation else c.scheduling.scheme not in ("A", "B")
        if c.combine.local_combine != expected_local:
            raise ValueError("scheme/Local Combine mismatch")
        radio_metrics = audit_radio(self.flows, self.segments, self.decisions, c, self.executions)
        flows = {r.flow.flow_id: r for r in self.flows}
        expected_events, expected_intervals = {}, []
        gates = {}
        for t in tokens.values():
            base_end = advance_time(t.arrival_s, c.timing.base_to_source_s)
            gates[t.token_id] = advance_time(base_end, c.timing.gate_s)
            for kind, time in ((E.TOKEN_ARRIVAL, t.arrival_s), (E.BASE_TO_SOURCE_DONE, base_end), (E.GATE_DONE, gates[t.token_id])):
                expected_events[(kind, t.token_id, None)] = time
            expected_intervals.extend([(t.token_id, "base_to_source", t.arrival_s, base_end, -1),
                                       (t.token_id, "gate", base_end, gates[t.token_id], t.source_die)])
        for r in self.flows:
            if r.flow.token_id not in tokens or r.flow.payload_bytes != c.model.activation_bytes:
                raise ValueError("wireless token/payload mismatch")
            expected_events[(R.DONE, r.flow.token_id, r.flow.flow_id)] = r.completed_s
        used, input_ready = set(), {}

        def check_path(last, src, dsts, kind, ready, experts=(), partial_id=None):
            f = last.flow
            if (f.flow_id in used or f.flow_type != kind or f.dsts != dsts
                    or last.expert_ids != experts or last.partial_id != partial_id):
                raise ValueError("logical flow route/association mismatch")
            used.add(f.flow_id)
            if relay:
                if last.parent_id not in flows:
                    raise ValueError("missing relay parent hop")
                first = flows[last.parent_id]
                if (first.flow.flow_id in used or first.flow.src != src or first.flow.dsts != (c.topology.num_dies,)
                        or first.flow.flow_type != kind + "_to_base" or first.flow.token_id != f.token_id
                        or first.expert_ids != experts or first.partial_id != partial_id or first.parent_id is not None
                        or f.src != c.topology.num_dies):
                    raise ValueError("invalid relay hop chain")
                used.add(first.flow.flow_id)
                _equal(first.ready_s, ready, "first relay hop readiness")
                _equal(last.ready_s, first.completed_s, "relay forwarding dependency")
            else:
                if f.src != src or last.parent_id is not None:
                    raise ValueError("invalid direct flow source/parent")
                _equal(last.ready_s, ready, "direct flow readiness")

        dispatch = defaultdict(list)
        for r in self.flows:
            if r.flow.flow_type == "dispatch":
                dispatch[r.flow.token_id].append(r)
        for token in tokens.values():
            remote = tuple(sorted(e for e in token.expert_ids if placement.die_for(e) != token.source_die))
            expected = [(e,) for e in remote] if p2p else ([remote] if remote else [])
            actual = {r.expert_ids: r for r in dispatch[token.token_id]}
            if len(actual) != len(dispatch[token.token_id]) or set(actual) != set(expected):
                raise ValueError("Dispatch assignment partition/count mismatch")
            for ids in expected:
                record = actual[ids]
                dies = tuple(sorted({placement.die_for(e) for e in ids}))
                check_path(record, token.source_die, dies, "dispatch", gates[token.token_id], ids)
                for expert in ids:
                    input_ready[token.token_id, expert] = record.completed_s
            for e in token.expert_ids:
                if placement.die_for(e) == token.source_die:
                    input_ready[token.token_id, e] = gates[token.token_id]
        requests = {r.request_id: r for r in self.inputs}
        if len(requests) != len(self.inputs) or Counter((r.token_id, r.expert_id) for r in self.inputs) != Counter(input_ready.keys()):
            raise ValueError("missing/duplicate expert assignment")
        for r in self.inputs:
            if r.die_id != placement.die_for(r.expert_id) or r.local != (r.die_id == tokens[r.token_id].source_die):
                raise ValueError("input placement/local flag mismatch")
            _equal(r.ready_s, input_ready[r.token_id, r.expert_id], "expert input dependency")
        batch_events, batch_intervals, token_last = audit_batches(c, self.inputs, self.executions)
        expected_events.update(batch_events)
        expected_intervals.extend(batch_intervals)
        if len(self.experts_completions) != len(tokens) or {x.token.token_id for x in self.experts_completions} != set(tokens):
            raise ValueError("invalid expert completions")
        for x in self.experts_completions:
            _equal(x.completed_s, token_last[x.token.token_id], "slowest expert completion")
            expected_events[(E.TOKEN_EXPERTS_DONE, x.token.token_id, None)] = x.completed_s
        expert_done = {(requests[i].token_id, requests[i].expert_id): x.end_s for x in self.executions for i in x.batch.request_ids}
        groups = {t.token_id: {d: tuple(sorted(ids)) for d, ids in placement.group(t.expert_ids).items()} for t in tokens.values()}
        expected_jobs, expected_parts = set(), []
        for t in tokens.values():
            for e in t.expert_ids:
                expected_jobs.add(("weight", t.token_id, placement.die_for(e), (e,)))
            expected_jobs.add(("source_reduce", t.token_id, t.source_die, tuple(sorted(t.expert_ids))))
            for die, ids in groups[t.token_id].items():
                if c.combine.local_combine:
                    expected_parts.append((t.token_id, die, ids))
                    if len(ids) > 1:
                        expected_jobs.add(("local_reduce", t.token_id, die, ids))
                else:
                    expected_parts.extend((t.token_id, die, (e,)) for e in ids)
        jobs = {(j.kind, j.token_id, j.die_id, j.expert_ids): j for j in self.jobs}
        if len(jobs) != len(self.jobs) or set(jobs) != expected_jobs or sorted(j.job_id for j in self.jobs) != list(range(len(jobs))):
            raise ValueError("Combine job coverage/IDs mismatch")
        weights = {(j.token_id, j.expert_ids[0]): j for j in self.jobs if j.kind == "weight"}
        locals_ = {(j.token_id, j.die_id): j for j in self.jobs if j.kind == "local_reduce"}
        sources = {j.token_id: j for j in self.jobs if j.kind == "source_reduce"}
        partials, deliveries = {p.partial_id: p for p in self.partials}, {d.partial_id: d for d in self.deliveries}
        if (len(partials) != len(self.partials) or len(deliveries) != len(self.deliveries) or set(deliveries) != set(partials)
                or Counter((p.token_id, p.die_id, p.expert_ids) for p in self.partials) != Counter(expected_parts)):
            raise ValueError("partial/result coverage mismatch")
        available = defaultdict(list)
        for p in self.partials:
            token, delivery = tokens[p.token_id], deliveries[p.partial_id]
            ready = locals_[p.token_id, p.die_id].end_s if len(p.expert_ids) > 1 else weights[p.token_id, p.expert_ids[0]].end_s
            _equal(p.ready_s, ready, "weighted partial readiness")
            check_numeric(p.vector, centralized_oracle(token, p.expert_ids, c.model.d_model, c.combine.verification_seed), "partial")
            expected_events[(L.PARTIAL_READY, p.token_id, p.partial_id)] = p.ready_s
            if p.die_id == token.source_die:
                if delivery.flow_id is not None:
                    raise ValueError("local partial must not transmit")
                _equal(delivery.available_s, p.ready_s, "local partial delivery")
            else:
                if delivery.flow_id not in flows:
                    raise ValueError("missing Return")
                r = flows[delivery.flow_id]
                if r.flow.token_id != p.token_id:
                    raise ValueError("Return token mismatch")
                check_path(r, p.die_id, (token.source_die,), "return", p.ready_s, partial_id=p.partial_id)
                _equal(delivery.available_s, r.completed_s, "Return delivery")
            available[p.token_id].append(delivery.available_s)
        if used != set(flows):
            raise ValueError("unexpected extra wireless flow")
        by_die = defaultdict(list)
        durations = {"weight": c.combine.weight_s, "local_reduce": c.combine.local_reduce_s, "source_reduce": c.timing.source_combine_s}
        kinds = {"weight": L.WEIGHT_DONE, "local_reduce": L.LOCAL_REDUCE_DONE, "source_reduce": L.SOURCE_COMBINE_DONE}
        for j in self.jobs:
            ready = (expert_done[j.token_id, j.expert_ids[0]] if j.kind == "weight" else
                     max(weights[j.token_id, e].end_s for e in j.expert_ids) if j.kind == "local_reduce" else max(available[j.token_id]))
            _equal(j.ready_s, ready, "Combine dependency")
            _equal(j.end_s, advance_time(j.start_s, durations[j.kind]), "Combine duration")
            by_die[j.die_id].append(j)
            expected_events[(kinds[j.kind], j.token_id, j.job_id)] = j.end_s
            expected_intervals.append((j.token_id, j.kind, j.start_s, j.end_s, j.die_id))
        for jobs_on_die in by_die.values():
            end = 0.0
            for j in sorted(jobs_on_die, key=lambda j: (j.ready_s, j.job_id)):
                _equal(j.start_s, max(end, j.ready_s) if c.combine.resource_model == "per_die_fifo" else j.ready_s, "Combine resource")
                end = j.end_s
        outputs, completions = {o.token_id: o for o in self.outputs}, {x.token.token_id: x for x in self.completions}
        if len(outputs) != len(self.outputs) or len(completions) != len(self.completions) or set(outputs) != set(tokens) or set(completions) != set(tokens):
            raise ValueError("output/completion coverage mismatch")
        validations = []
        for row, token_id in enumerate(sorted(tokens)):
            t, o, completion = tokens[token_id], outputs[token_id], completions[token_id]
            if completion.token != t:
                raise ValueError("completion token changed")
            _equal(o.ready_s, sources[token_id].end_s, "output readiness")
            _equal(completion.completed_s, advance_time(o.ready_s, c.timing.source_to_base_s), "final Base completion")
            expected_events[(L.SOURCE_TO_BASE_DONE, token_id, None)] = completion.completed_s
            expected_intervals.append((token_id, "source_to_base", o.ready_s, completion.completed_s, -1))
            error = check_numeric(o.vector, centralized_oracle(t, t.expert_ids, c.model.d_model, c.combine.verification_seed), "final output")
            validations.append({"output_row": row, "token_id": token_id, "max_abs_error": error, "vector_sha256": vector_digest(o.vector)})
        actual = [(i.token_id, i.kind, i.start_s, i.end_s, i.die_id) for i in self.intervals]
        if Counter(actual) != Counter(expected_intervals):
            raise ValueError("layer interval coverage mismatch")
        causal_events = [e for e in self.events if e.kind not in (R.REBALANCE, R.WAKE)]
        if Counter((e.kind, e.token_id, e.ref_id) for e in causal_events) != Counter(expected_events.keys()):
            raise ValueError("layer event coverage mismatch")
        for e in causal_events:
            _equal(e.time_s, expected_events[e.kind, e.token_id, e.ref_id], "causal event time")
        if any(a.time_s > b.time_s for a, b in zip(self.events, self.events[1:])):
            raise ValueError("nonmonotonic events")
        if len(set(self.ignored_deadlines)) != len(self.ignored_deadlines) or any(
                (E.BATCH_DEADLINE, next((e.token_id for e in self.events if e.kind == E.BATCH_DEADLINE and e.ref_id == i), -1), i)
                not in expected_events for i in self.ignored_deadlines):
            raise ValueError("invalid ignored deadline")
        raw, compute_rows = summarize(self.completions, (), self.intervals, THzLink(c.phy), c.topology.num_dies)
        for r in self.flows:
            if r.ready_s < tokens[r.flow.token_id].arrival_s or r.completed_s > completions[r.flow.token_id].completed_s:
                raise ValueError("flow outside token lifetime")
        return validations, raw, compute_rows, radio_metrics

    @property
    def summary(self):
        validation, raw, _, radio = self.audit()
        duration = raw["layer_makespan_s"]
        payload = sum(r.flow.payload_bytes for r in self.flows)
        capacity = radio["allocated_capacity_bits"]
        raw.update({"thz_utilization": radio["occupied_subcarrier_seconds"] / (self.config.phy.num_subcarriers * duration),
                    "pepd": radio["active_flow_seconds"] / duration,
                    "pepd_payload_only": radio["payload_flow_seconds"] / duration,
                    "wireless_payload_bytes": payload,
                    "dispatch_payload_bytes": sum(r.flow.payload_bytes for r in self.flows if r.flow.flow_type.startswith("dispatch")),
                    "return_payload_bytes": sum(r.flow.payload_bytes for r in self.flows if r.flow.flow_type.startswith("return")),
                    "bandwidth_waste": max(0.0, 1 - payload * 8 / capacity) if capacity else 0.0,
                    **radio})
        return {"stage": "stage4_five_schemes", "scheme": scheme_label(self.config),
                "study": "ablation" if self.config.ablation else "five_scheme", "broadcast": uses_broadcast(self.config),
                "policy": self.config.scheduling.policy, "model_status": "uncalibrated_reference",
                "endpoint": "all_tokens_returned_to_base", "config_sha256": self.config.digest,
                "workload_sha256": self.workload.digest, "total_bandwidth_hz": self.config.phy.total_bandwidth_hz,
                "num_subcarriers": self.config.phy.num_subcarriers, "expert_assignments": len(self.inputs),
                "compute_batches": len(self.executions), "local_combine": self.config.combine.local_combine,
                "wireless_hops": len(self.flows), "dispatch_flows": sum(r.flow.flow_type == "dispatch" for r in self.flows),
                "return_flows": sum(r.flow.flow_type == "return" for r in self.flows),
                "relay_extra_hops": sum(r.parent_id is not None for r in self.flows),
                "verification_max_abs_error": max(v["max_abs_error"] for v in validation),
                "ignored_radio_events": len(self.ignored_radio), **raw}

    def export(self, directory):
        summary = self.summary
        validation, _, compute_rows, _ = self.audit()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError("output directory must be empty")
        def csv(name, rows, columns=None):
            pd.DataFrame(rows, columns=columns).to_csv(directory / name, index=False, float_format="%.17g", lineterminator="\n")
        csv("summary.csv", [summary])
        done = {c.token.token_id: c.completed_s for c in self.experts_completions}
        outputs = {o.token_id: o for o in self.outputs}
        csv("tokens.csv", [{**asdict(x.token), "expert_ids": json.dumps(x.token.expert_ids), "gate_weights": json.dumps(x.token.gate_weights),
                            "experts_done_s": done[x.token.token_id], "source_combined_s": outputs[x.token.token_id].ready_s,
                            "completed_s": x.completed_s, "latency_s": x.completed_s-x.token.arrival_s} for x in self.completions])
        flow_columns = ["flow_id", "token_id", "src", "dsts", "payload_bytes", "flow_type", "ready_s", "admitted_s", "completed_s", "queue_wait_s", "expert_ids", "partial_id", "parent_id"]
        csv("flows.csv", [{**asdict(r.flow), "dsts": json.dumps(r.flow.dsts), "ready_s": r.ready_s, "admitted_s": r.admitted_s,
             "completed_s": r.completed_s, "queue_wait_s": r.admitted_s-r.ready_s, "expert_ids": json.dumps(r.expert_ids),
             "partial_id": r.partial_id, "parent_id": r.parent_id} for r in self.flows], flow_columns)
        csv("radio_segments.csv", [{**asdict(s), "subcarriers": json.dumps(s.subcarriers)} for s in self.segments],
            ["flow_id", "start_s", "end_s", "phase", "subcarriers", "rate_bps", "sent_bits"])
        csv("allocations.csv", [{**d, "subcarriers": json.dumps(d["subcarriers"])} for d in self.decisions],
            ["generation", "time_s", "flow_id", "subcarriers", "remaining_bits", "q", "w", "idle", "critical", "priority", "repaired_overflow"])
        executions = {i: x for x in self.executions for i in x.batch.request_ids}
        csv("requests.csv", [{**asdict(r), "batch_id": executions[r.request_id].batch.batch_id,
             "batch_ready_s": executions[r.request_id].batch.ready_s, "compute_start_s": executions[r.request_id].start_s,
             "compute_end_s": executions[r.request_id].end_s} for r in self.inputs])
        csv("batches.csv", [{**asdict(x.batch), "request_ids": json.dumps(x.batch.request_ids),
             **{k: v for k, v in asdict(x).items() if k != "batch"}} for x in self.executions])
        csv("combine_jobs.csv", [{**asdict(j), "expert_ids": json.dumps(j.expert_ids), "queue_wait_s": j.start_s-j.ready_s} for j in self.jobs])
        deliveries = {d.partial_id: d for d in self.deliveries}
        csv("partials.csv", [{"partial_id": p.partial_id, "token_id": p.token_id, "die_id": p.die_id,
            "expert_ids": json.dumps(p.expert_ids), "ready_s": p.ready_s, "available_at_source_s": deliveries[p.partial_id].available_s,
            "return_flow_id": deliveries[p.partial_id].flow_id, "vector_sha256": vector_digest(p.vector)} for p in self.partials])
        ignored = set(self.ignored_radio)
        csv("events.csv", [{"sequence": i, "time_s": e.time_s, "event_type": e.kind.value, "token_id": e.token_id,
            "ref_id": e.ref_id, "ignored": e in ignored or (e.kind == E.BATCH_DEADLINE and e.ref_id in self.ignored_deadlines)} for i, e in enumerate(self.events)])
        csv("intervals.csv", [asdict(i) for i in self.intervals])
        csv("compute_utilization.csv", compute_rows)
        # Use the same union-of-intervals implementation as Stage 3.
        from .layer_metrics import LayerResult
        csv("combine_utilization.csv", LayerResult.combine_utilization(self))
        csv("output_validation.csv", validation)
        np.save(directory / "outputs.npy", np.stack([outputs[i].vector for i in sorted(outputs)]), allow_pickle=False)
        source = {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))}
        manifest = {"schema_version": 1, "simulator_version": __version__, "stage": "stage4_five_schemes",
            "model_status": "uncalibrated; ideal retuning; synthetic float64 semantics", "config": self.config.to_dict(),
            "config_sha256": self.config.digest, "workload_sha256": self.workload.digest, "source_sha256": stable_digest(source),
            "input_trace_sha256": self.workload.trace_sha256, "python_version": platform.python_version(),
            "dependencies": {name: version(name) for name in ("numpy", "pandas", "matplotlib")},
            "specification_sha256": "072f0287b6b79494564d0190f3143b348642f3328aaf9d5168eb385b29f40fdf",
            "assumptions": STAGE4_ASSUMPTIONS + ([{"id": "A15", "status": "explicit_ablation", "model": "broadcast and Local Combine varied independently; C/D/Proposed select PHY policies only, not canonical baselines"}] if self.config.ablation else []),
            "virtual_base_endpoint": self.config.topology.num_dies if self.config.scheduling.scheme == "A" else None,
            "not_implemented": ["real FFN", "BF16/FP8 numerical quantization", "shared expert", "calibrated retuning/power/channel model"],
            "numerical_verification": {"rtol": 1e-10, "atol": 1e-10, "output_row_mapping": "output_validation.csv",
                                       "max_abs_error": summary["verification_max_abs_error"]}}
        for name, value in (("manifest.json", manifest), ("routing_trace.json", self.workload.to_trace())):
            (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return directory
