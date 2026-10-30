"""Trace auditing and integral metrics, with an explicit observation window."""

from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from . import __version__
from .config import SimulationConfig, integer, number, stable_digest
from .engine import Event, EventType
from .moe import Interval, TokenCompletion, Transmission
from .ofdma import Allocation, validate_allocations
from .thz_phy import THzLink


ASSUMPTIONS = [
    {"id": "A01", "status": "user_accepted_scope_not_hardware_validation", "model": "routed experts only; shared expert excluded"},
    {"id": "A02", "status": "uncalibrated", "model": "Base transfers are fixed independent delays outside THz budget"},
    {"id": "A03", "status": "uncalibrated", "model": "Gate and Source Combine use configured delays; no numerical FFN"},
    {"id": "A04", "status": "uncalibrated", "model": "compute, memory bandwidth, PHY efficiency and setup are sensitivity inputs"},
    {"id": "A05", "status": "stage1_limit", "model": "one token and one remote expert; no batch formation or compute contention"},
    {"id": "A06", "status": "user_selected_not_exercised", "model": "ideal multiple TX/RX per Die on globally disjoint subcarriers; no spatial reuse"},
    {"id": "A07", "status": "synthetic_workload", "model": "unique expert assignments; normalized synthetic weights; contiguous balanced placement"},
    {"id": "A08", "status": "model_convention", "model": "setup reserves spectrum; header/FEC/guard represented only by PHY efficiency"},
    {"id": "A13", "status": "metric_convention", "model": "window earliest arrival to final Base completion; PEPD and occupancy include setup; linear percentiles"},
    {"id": "A14", "status": "stage1_limit", "model": "singleton Local Combine bypass; no weighted-vector correctness or physical-channel calibration"},
]


def audit_transmissions(transmissions: tuple[Transmission, ...], link: THzLink,
                        num_dies: int) -> int:
    """Reject overbooking, invalid endpoints and missing/excess payload capacity."""
    endpoints = []
    flow_ids = set()
    for tx in transmissions:
        flow = tx.flow
        if flow.flow_id in flow_ids:
            raise ValueError("duplicate transmitted flow_id")
        flow_ids.add(flow.flow_id)
        validate_allocations((Allocation(flow.flow_id, tx.subcarriers),), link.config.num_subcarriers)
        integer("payload_bytes", flow.payload_bytes)
        for die in (flow.src, *flow.dsts):
            integer("flow endpoint", die, 0)
            if die >= num_dies:
                raise ValueError("flow endpoint outside topology")
        if not flow.dsts or len(set(flow.dsts)) != len(flow.dsts) or flow.src in flow.dsts:
            raise ValueError("wireless destinations must be nonempty, unique and remote")
        if flow.flow_type not in ("dispatch", "return"):
            raise ValueError("unknown flow type")
        for label, value in (("start", tx.start_s), ("payload start", tx.payload_start_s), ("end", tx.end_s)):
            number(label, value, zero=True)
        if not tx.start_s <= tx.payload_start_s < tx.end_s:
            raise ValueError("invalid transmission causal interval")
        number("rate_bps", tx.rate_bps)
        expected_rate = len(tx.subcarriers) * link.subcarrier_rate_bps
        if not math.isclose(tx.rate_bps, expected_rate, rel_tol=1e-12):
            raise ValueError("rate exceeds or differs from allocated subcarrier capacity")
        if not math.isclose(tx.payload_start_s - tx.start_s, link.config.setup_s,
                            rel_tol=1e-9, abs_tol=1e-15):
            raise ValueError("setup time differs from configured setup")
        sent = (tx.end_s - tx.payload_start_s) * tx.rate_bps
        if not math.isclose(sent, flow.payload_bytes * 8, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("payload bit conservation violated")
        # Half-open intervals: release before acquisition at the same instant.
        endpoints.extend(((tx.start_s, 1, flow.flow_id, tx.subcarriers),
                          (tx.end_s, -1, flow.flow_id, tx.subcarriers)))
    active = set()
    occupied = set()
    peak = 0
    for _, action, flow_id, carriers in sorted(endpoints):
        if action == -1:
            active.remove(flow_id)
            occupied.difference_update(carriers)
        else:
            if occupied.intersection(carriers):
                raise ValueError("overlapping transmissions collide on subcarriers")
            occupied.update(carriers)
            active.add(flow_id)
            peak = max(peak, len(active))
    return peak


def summarize(completions: tuple[TokenCompletion, ...], transmissions: tuple[Transmission, ...],
              intervals: tuple[Interval, ...], link: THzLink, num_dies: int) -> tuple[dict, list[dict]]:
    if not completions:
        raise ValueError("cannot summarize a zero-token run")
    tokens = {c.token.token_id: c for c in completions}
    if len(tokens) != len(completions):
        raise ValueError("duplicate token completion")
    for completion in completions:
        number("arrival", completion.token.arrival_s, zero=True)
        number("completion", completion.completed_s, zero=True)
        if completion.completed_s < completion.token.arrival_s:
            raise ValueError("token completed before arrival")
    origin = min(c.token.arrival_s for c in completions)
    end = max(c.completed_s for c in completions)
    duration = end - origin
    if duration <= 0:
        raise ValueError("zero observation window has undefined throughput")
    peak = audit_transmissions(transmissions, link, num_dies)
    for tx in transmissions:
        token = tokens.get(tx.flow.token_id)
        if token is None or tx.start_s < token.token.arrival_s or tx.end_s > token.completed_s:
            raise ValueError("flow outside its token lifetime")
    busy = [[] for _ in range(num_dies)]
    for span in intervals:
        number("interval start", span.start_s, zero=True)
        number("interval end", span.end_s, zero=True)
        token = tokens.get(span.token_id)
        if (token is None or span.start_s < token.token.arrival_s
                or span.end_s > token.completed_s or span.end_s < span.start_s):
            raise ValueError("invalid interval / interval outside token lifetime")
        if span.kind == "expert_compute":
            integer("compute die", span.die_id, 0)
            if span.die_id >= num_dies:
                raise ValueError("compute die outside topology")
            busy[span.die_id].append((span.start_s, span.end_s))
    compute_rows = []
    for die, spans in enumerate(busy):
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("shared Die compute resource is overbooked")
        busy_s = math.fsum(stop - start for start, stop in spans)
        compute_rows.append({"die_id": die, "busy_s": busy_s, "utilization": busy_s / duration})
    latencies = [c.completed_s - c.token.arrival_s for c in completions]
    p50, p95, p99 = np.percentile(latencies, [50, 95, 99], method="linear")
    occupied_sc_s = math.fsum(len(t.subcarriers) * (t.end_s - t.start_s) for t in transmissions)
    active_flow_s = math.fsum(t.end_s - t.start_s for t in transmissions)
    payload_flow_s = math.fsum(t.end_s - t.payload_start_s for t in transmissions)
    capacity = math.fsum(t.rate_bps * (t.end_s - t.start_s) for t in transmissions)
    payload_bytes = sum(t.flow.payload_bytes for t in transmissions)
    waste = 1.0 - payload_bytes * 8 / capacity if capacity else 0.0
    if waste < -1e-9 or waste > 1 + 1e-9:
        raise ValueError("invalid bandwidth waste / capacity accounting")
    summary = {
        "num_tokens": len(completions), "window_start_s": origin, "window_end_s": end,
        "layer_makespan_s": duration, "token_latency_p50_s": float(p50),
        "token_latency_p95_s": float(p95), "token_latency_p99_s": float(p99),
        "throughput_tokens_per_s": len(completions) / duration,
        "thz_utilization": occupied_sc_s / (link.config.num_subcarriers * duration),
        "pepd": active_flow_s / duration, "pepd_payload_only": payload_flow_s / duration,
        "peak_concurrent_flows": peak, "occupied_subcarrier_seconds": occupied_sc_s,
        "wireless_payload_bytes": payload_bytes, "allocated_capacity_bits": capacity,
        "bandwidth_waste": min(1.0, max(0.0, waste)),
        "dispatch_payload_bytes": sum(t.flow.payload_bytes for t in transmissions if t.flow.flow_type == "dispatch"),
        "return_payload_bytes": sum(t.flow.payload_bytes for t in transmissions if t.flow.flow_type == "return"),
    }
    return summary, compute_rows


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    completions: tuple[TokenCompletion, ...]
    transmissions: tuple[Transmission, ...]
    intervals: tuple[Interval, ...]
    events: tuple[Event, ...]

    @property
    def workload_digest(self) -> str:
        return stable_digest([asdict(c.token) for c in self.completions])

    @property
    def summary(self) -> dict:
        self.audit_causality()
        metrics, _ = summarize(self.completions, self.transmissions, self.intervals,
                               THzLink(self.config.phy), self.config.topology.num_dies)
        return {"stage": "stage1_reference", "scheme": "isolated_full_band_reference",
                "model_status": "uncalibrated", "config_sha256": self.config.digest,
                "workload_sha256": self.workload_digest, "seed": self.config.workload.seed,
                "total_bandwidth_hz": self.config.phy.total_bandwidth_hz,
                "num_subcarriers": self.config.phy.num_subcarriers, **metrics}

    def audit_causality(self) -> None:
        """Check the stage-1 dependency graph independently of queue execution."""
        if len(self.completions) != 1 or len(self.transmissions) != 2:
            raise ValueError("reference trace requires one completion and two transmissions")
        token = self.completions[0].token
        if [e.kind for e in self.events] != list(EventType):
            raise ValueError("missing, reordered or duplicate reference events")
        if any(e.token_id != token.token_id for e in self.events):
            raise ValueError("event belongs to a different token")
        times = [token.arrival_s, *(e.time_s for e in self.events)]
        if any(not math.isfinite(t) for t in times) or any(a > b for a, b in zip(times, times[1:])):
            raise ValueError("event causality violated")
        event_time = {e.kind.value: e.time_s for e in self.events}
        boundaries = {
            "base_to_source": (token.arrival_s, event_time["BASE_TO_SOURCE_DONE"]),
            "gate": (event_time["BASE_TO_SOURCE_DONE"], event_time["GATE_DONE"]),
            "dispatch": (event_time["DISPATCH_READY"], event_time["DISPATCH_DONE"]),
            "expert_compute": (event_time["DISPATCH_DONE"], event_time["EXPERT_COMPUTE_DONE"]),
            "local_combine_singleton_bypass": (event_time["EXPERT_COMPUTE_DONE"], event_time["LOCAL_COMBINE_DONE"]),
            "return": (event_time["RETURN_READY"], event_time["RETURN_DONE"]),
            "source_combine": (event_time["RETURN_DONE"], event_time["SOURCE_COMBINE_DONE"]),
            "source_to_base": (event_time["SOURCE_COMBINE_DONE"], event_time["SOURCE_TO_BASE_DONE"]),
        }
        if len(self.intervals) != len(boundaries) or {s.kind for s in self.intervals} != set(boundaries):
            raise ValueError("missing or duplicate reference intervals")
        for span in self.intervals:
            if span.token_id != token.token_id or (span.start_s, span.end_s) != boundaries[span.kind]:
                raise ValueError("interval disagrees with causal event boundaries")
        dispatch, returned = self.transmissions
        if (dispatch.flow.flow_type, returned.flow.flow_type) != ("dispatch", "return"):
            raise ValueError("incorrect reference flow sequence")
        if (dispatch.flow.src != token.source_die or returned.flow.dsts != (token.source_die,)
                or dispatch.flow.dsts != (returned.flow.src,)):
            raise ValueError("return must reach the original Source from the expert Die")
        for tx in self.transmissions:
            if ((tx.start_s, tx.end_s) != boundaries[tx.flow.flow_type]
                    or tx.flow.payload_bytes != token.activation_bytes):
                raise ValueError("flow disagrees with token payload or causal timing")
        if self.completions[0].completed_s != event_time["SOURCE_TO_BASE_DONE"]:
            raise ValueError("completion does not include the final Base transfer")

    def export(self, directory: str | Path) -> Path:
        # Validate before writing; a stale output directory must never look like a new run.
        summary = self.summary
        _, compute_rows = summarize(self.completions, self.transmissions, self.intervals,
                                    THzLink(self.config.phy), self.config.topology.num_dies)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError("output directory must be empty; choose a new run directory")

        def csv(name, rows):
            pd.DataFrame(rows).to_csv(directory / name, index=False, float_format="%.17g", lineterminator="\n")

        csv("summary.csv", [summary])
        csv("tokens.csv", [{**asdict(c.token), "expert_ids": json.dumps(c.token.expert_ids),
                            "gate_weights": json.dumps(c.token.gate_weights), "completed_s": c.completed_s,
                            "latency_s": c.completed_s - c.token.arrival_s} for c in self.completions])
        csv("flows.csv", [{**asdict(t.flow), "dsts": json.dumps(t.flow.dsts),
                           "start_s": t.start_s, "payload_start_s": t.payload_start_s, "end_s": t.end_s,
                           "subcarriers": json.dumps(t.subcarriers), "num_subcarriers": len(t.subcarriers),
                           "rate_bps": t.rate_bps} for t in self.transmissions])
        csv("events.csv", [{"sequence": i, "time_s": e.time_s, "event_type": e.kind.value,
                            "token_id": e.token_id} for i, e in enumerate(self.events)])
        csv("intervals.csv", [asdict(i) for i in self.intervals])
        csv("compute_utilization.csv", compute_rows)
        package_dir = Path(__file__).parent
        source = {p.name: p.read_text(encoding="utf-8") for p in sorted(package_dir.glob("*.py"))}
        manifest = {
            "schema_version": 1, "simulator_version": __version__, "stage": "stage1_reference",
            "model_status": "uncalibrated; analytical timing validation only",
            "config": self.config.to_dict(), "config_sha256": self.config.digest,
            "workload_sha256": self.workload_digest, "source_sha256": stable_digest(source),
            "python_version": platform.python_version(),
            "dependencies": {name: version(name) for name in ("numpy", "pandas", "matplotlib")},
            "specification_sha256": "072f0287b6b79494564d0190f3143b348642f3328aaf9d5168eb385b29f40fdf",
            "assumptions": ASSUMPTIONS,
            "not_implemented": ["multi-token/multi-expert", "numeric local combine", "shared expert",
                                "TDMA queue", "static OFDMA", "CA-OFDMA", "five-scheme comparison", "parameter sweeps"],
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                                           indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return directory
