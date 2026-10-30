"""Offline checks of spectrum, fractional-bit conservation and admission policy."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import heapq
import math

from .config import integer, number
from .moe import Flow
from .ofdma import Allocation, validate_allocations
from .pipeline_metrics import _equal
from .radio import allocate_counts
from .thz_phy import THzLink


@dataclass(frozen=True)
class RadioFlow:
    flow: Flow
    ready_s: float
    admitted_s: float
    completed_s: float
    expert_ids: tuple[int, ...]
    partial_id: int | None
    parent_id: int | None


def audit_radio(records, segments, decisions, config, executions):
    link = THzLink(config.phy)
    flows = {r.flow.flow_id: r for r in records}
    if sorted(flows) != list(range(len(records))):
        raise ValueError("missing or duplicate radio flow ID")
    pieces, endpoints = defaultdict(list), []
    for record in records:
        f = record.flow
        for label, value in (("ready", record.ready_s), ("admitted", record.admitted_s), ("completed", record.completed_s)):
            number(label, value, zero=True)
        if not record.ready_s <= record.admitted_s < record.completed_s:
            raise ValueError("invalid radio flow lifetime")
        integer("payload bytes", f.payload_bytes)
        max_endpoint = config.topology.num_dies if config.scheduling.scheme == "A" else config.topology.num_dies - 1
        if (not f.dsts or len(set(f.dsts)) != len(f.dsts) or f.src in f.dsts
                or any(type(d) is not int or not 0 <= d <= max_endpoint for d in (f.src, *f.dsts))):
            raise ValueError("invalid wireless endpoints")
    for index, segment in enumerate(segments):
        if segment.flow_id not in flows:
            raise ValueError("segment for unknown flow")
        r = flows[segment.flow_id]
        if not r.admitted_s <= segment.start_s < segment.end_s <= r.completed_s:
            raise ValueError("segment outside admitted flow lifetime")
        validate_allocations((Allocation(segment.flow_id, segment.subcarriers),), config.phy.num_subcarriers)
        _equal(segment.rate_bps, len(segment.subcarriers) * link.subcarrier_rate_bps, "segment rate")
        if segment.phase not in ("setup", "payload"):
            raise ValueError("invalid radio segment phase")
        expected_bits = 0.0 if segment.phase == "setup" else (segment.end_s - segment.start_s) * segment.rate_bps
        if not math.isclose(segment.sent_bits, expected_bits, rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError("radio segment bit conservation")
        pieces[segment.flow_id].append(segment)
        endpoints.extend(((segment.start_s, 1, index), (segment.end_s, -1, index)))
    active, occupied, peak = set(), set(), 0
    for _, action, index in sorted(endpoints):
        segment = segments[index]
        if action == -1:
            active.remove(segment.flow_id)
            occupied.difference_update(segment.subcarriers)
        else:
            if segment.flow_id in active or occupied.intersection(segment.subcarriers):
                raise ValueError("flow or subcarrier overlaps")
            active.add(segment.flow_id)
            occupied.update(segment.subcarriers)
            peak = max(peak, len(active))
    cumulative = {}
    for flow_id, record in flows.items():
        spans = sorted(pieces[flow_id], key=lambda p: p.start_s)
        end, setup, sent, payload_seen = record.admitted_s, 0.0, 0.0, False
        times, totals = [], []
        for s in spans:
            _equal(s.start_s, end, "uninterrupted admitted service")
            end = s.end_s
            if s.phase == "setup":
                if payload_seen:
                    raise ValueError("setup restarted after payload")
                setup += s.end_s - s.start_s
            else:
                payload_seen = True
            sent += s.sent_bits
            times.append(s.end_s)
            totals.append(sent)
        _equal(end, record.completed_s, "flow completion boundary")
        _equal(setup, config.phy.setup_s, "one setup per hop")
        _equal(sent, record.flow.payload_bytes * 8, "complete hop payload")
        cumulative[flow_id] = (times, totals)
    # Non-preemptive FIFO admission: a service slot is freed only at completion.
    slots = [0.0] * (1 if config.scheduling.policy == "tdma" else config.phy.num_subcarriers)
    for record in sorted(records, key=lambda r: (r.ready_s, r.flow.flow_id)):
        available = heapq.heappop(slots)
        _equal(record.admitted_s, max(available, record.ready_s), "FIFO admission/work conservation")
        heapq.heappush(slots, record.completed_s)
    groups, by_flow = defaultdict(list), defaultdict(list)
    for d in decisions:
        if d["flow_id"] not in flows:
            raise ValueError("allocation for unknown flow")
        groups[d["generation"]].append(d)
        by_flow[d["flow_id"]].append(d)
    for rows in groups.values():
        time = rows[0]["time_s"]
        expected_active = sorted(i for i, f in flows.items() if f.admitted_s <= time < f.completed_s)
        if [d["flow_id"] for d in rows] != expected_active or any(d["time_s"] != time for d in rows):
            raise ValueError("allocation active set incomplete or duplicated")
        priorities = []
        for d in rows:
            record = flows[d["flow_id"]]
            times, totals = cumulative[d["flow_id"]]
            index = bisect_right(times, time) - 1
            remaining = record.flow.payload_bytes * 8 - (totals[index] if index >= 0 else 0)
            _equal(d["remaining_bits"], remaining, "allocation remaining bits")
            cfg = config.scheduling
            q, w = remaining / 8 / cfg.q_ref_bytes, (time - record.ready_s) / cfg.w_ref_s
            busy = {x.batch.die_id for x in executions if x.batch.ready_s <= time < x.end_s}
            idle = sum(die not in busy for die in record.flow.dsts) / len(record.flow.dsts) if record.flow.flow_type == "dispatch" else 0.0
            critical = float(record.flow.flow_type.startswith("return"))
            for field, value in (("q", q), ("w", w), ("idle", idle), ("critical", critical)):
                _equal(d[field], value, "CA feature " + field)
            priority = cfg.alpha*q + cfg.beta*w + cfg.gamma*idle + cfg.delta*critical
            _equal(d["priority"], priority, "CA priority")
            # Independent feature/priority checks above tolerate arithmetic roundoff.
            # Re-summing segment bits may differ from incremental subtraction by an
            # ULP, which can reorder tied fractional remainders. Replay the exact
            # audited decision input for the discrete integer allocation.
            priorities.append(d["priority"])
        counts, repaired = allocate_counts(config.scheduling.policy, config.phy.num_subcarriers, priorities)
        offset = 0
        for row, count in zip(rows, counts):
            if row["subcarriers"] != tuple(range(offset, offset + count)) or row["repaired_overflow"] != repaired:
                raise ValueError("scheduler allocation differs from configured rule")
            offset += count
    for flow_id, spans in pieces.items():
        rows = sorted(by_flow[flow_id], key=lambda d: d["time_s"])
        if not rows or rows[0]["time_s"] != flows[flow_id].admitted_s:
            raise ValueError("missing initial allocation")
        times = [d["time_s"] for d in rows]
        for span in spans:
            i = bisect_right(times, span.start_s) - 1
            if i < 0 or span.subcarriers != rows[i]["subcarriers"]:
                raise ValueError("segment differs from scheduler allocation")
            if i + 1 < len(rows) and span.end_s > times[i+1]:
                raise ValueError("segment crosses a reallocation boundary")
    return {"peak_concurrent_flows": peak,
            "occupied_subcarrier_seconds": math.fsum(len(s.subcarriers)*(s.end_s-s.start_s) for s in segments),
            "active_flow_seconds": math.fsum(s.end_s-s.start_s for s in segments),
            "payload_flow_seconds": math.fsum(s.end_s-s.start_s for s in segments if s.phase == "payload"),
            "allocated_capacity_bits": math.fsum(s.rate_bps*(s.end_s-s.start_s) for s in segments)}
