"""Event-driven shared-spectrum service with piecewise rates and stale-event guards.

Active flows retain admission until done. Setup is paid once per hop; retuning is
ideal and free. These are explicit reference assumptions, not PHY measurements.
"""

from collections import deque
from dataclasses import dataclass
import math

from .batching import advance_time
from .config import integer, number
from .engine import RadioEventType as R
from .moe import Flow
from .ofdma import Allocation, validate_allocations
from .thz_phy import THzLink


def allocate_counts(policy, total, priorities):
    """Paper floor/equal rule; CA preserves its rule except infeasible max(1,...)."""
    integer("total", total)
    if policy not in ("tdma", "static", "ca"):
        raise ValueError("unknown radio policy")
    for value in priorities:
        number("priority", value, zero=True)
    count = len(priorities)
    if not count:
        return (), False
    if count > total or (policy == "tdma" and count > 1):
        raise ValueError("admit no more flows than available subcarriers")
    if policy == "tdma":
        return (total,), False
    if policy == "static":
        return (total // count,) * count, False
    largest = max(priorities)
    scaled = [p / largest for p in priorities] if largest else [1.0] * count
    weights = [p / math.fsum(scaled) for p in scaled]
    quota = [total * p for p in weights]
    counts = [max(1, math.floor(q)) for q in quota]
    repaired = sum(counts) > total
    if repaired:
        quota = [(total - count) * p for p in weights]
        counts = [1 + math.floor(q) for q in quota]
    order = sorted(range(count), key=lambda i: (-(quota[i] - math.floor(quota[i])), i))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    if sum(counts) != total or min(counts) < 1:
        raise RuntimeError("integer allocation failed budget conservation")
    return tuple(counts), repaired


@dataclass
class FlowState:
    flow: Flow
    ready_s: float
    expert_ids: tuple[int, ...]
    partial_id: int | None
    parent_id: int | None
    remaining_bits: float
    setup_left_s: float
    admitted_s: float | None = None
    completed_s: float | None = None


@dataclass(frozen=True)
class RadioSegment:
    flow_id: int
    start_s: float
    end_s: float
    phase: str
    subcarriers: tuple[int, ...]
    rate_bps: float
    sent_bits: float


class SharedRadio:
    def __init__(self, loop, phy, scheduling, on_done, idle_fraction=lambda flow: 0.0):
        self.loop, self.phy, self.scheduling = loop, phy, scheduling
        self.link = THzLink(phy)
        self.on_done, self.idle_fraction = on_done, idle_fraction
        self.states, self.active, self.pending = {}, {}, deque()
        self.allocations = {}
        self.segments, self.decisions, self.ignored = [], [], []
        self.last_s = 0.0
        self.generation = 0
        self.kicked_at = None

    def enqueue(self, token_id, src, dsts, payload_bytes, kind, *, expert_ids=(), partial_id=None, parent_id=None):
        flow = Flow(len(self.states), token_id, src, tuple(dsts), payload_bytes, kind)
        self.states[flow.flow_id] = FlowState(flow, self.loop.now_s, tuple(expert_ids), partial_id, parent_id,
                                             float(payload_bytes * 8), self.phy.setup_s)
        self.pending.append(flow.flow_id)
        self.kick()
        return flow.flow_id

    def kick(self):
        if self.kicked_at != self.loop.now_s:
            self.kicked_at = self.loop.now_s
            self.loop.schedule(self.loop.now_s, R.REBALANCE, priority=1)

    def advance(self):
        now = self.loop.now_s
        for flow_id, carriers in self.allocations.items():
            state = self.active[flow_id]
            rate = len(carriers) * self.link.subcarrier_rate_bps
            setup = min(state.setup_left_s, now - self.last_s)
            middle = min(now, self.last_s + setup)
            if middle > self.last_s:
                self.segments.append(RadioSegment(flow_id, self.last_s, middle, "setup", carriers, rate, 0.0))
            state.setup_left_s = max(0.0, state.setup_left_s - setup)
            if now > middle:
                sent = (now - middle) * rate
                if sent > state.remaining_bits and not math.isclose(sent, state.remaining_bits, rel_tol=1e-9, abs_tol=1e-8):
                    raise RuntimeError("radio progressed beyond flow completion")
                sent = min(sent, state.remaining_bits)
                self.segments.append(RadioSegment(flow_id, middle, now, "payload", carriers, rate, sent))
                state.remaining_bits = max(0.0, state.remaining_bits - sent)
                if state.remaining_bits <= state.flow.payload_bytes * 8 * 1e-12:
                    state.remaining_bits = 0.0
        self.last_s = now
        done = [i for i, s in self.active.items() if s.remaining_bits == 0 and s.setup_left_s == 0]
        for flow_id in done:
            state = self.active.pop(flow_id)
            self.allocations.pop(flow_id)
            state.completed_s = now
            self.on_done(state)
        return bool(done)

    def handle(self, event):
        if event.kind == R.WAKE:
            if event.ref_id != self.generation:
                self.ignored.append(event)
                return
            self.advance()
            self.kick()
            return
        if event.kind != R.REBALANCE:
            raise ValueError("unknown radio event")
        self.kicked_at = None
        if self.advance():
            self.kick()  # Process causal zero-time completions before allocation.
            return
        limit = 1 if self.scheduling.policy == "tdma" else self.phy.num_subcarriers
        while self.pending and len(self.active) < limit:
            flow_id = self.pending.popleft()
            state = self.states[flow_id]
            state.admitted_s = self.loop.now_s
            self.active[flow_id] = state
        self.generation += 1
        ids = sorted(self.active)  # FIFO tie-breaking, stable carrier identities.
        features, priorities = [], []
        cfg = self.scheduling
        for flow_id in ids:
            state = self.active[flow_id]
            q = state.remaining_bits / 8 / cfg.q_ref_bytes
            w = (self.loop.now_s - state.ready_s) / cfg.w_ref_s
            idle = self.idle_fraction(state)
            critical = float(state.flow.flow_type.startswith("return"))
            priority = cfg.alpha * q + cfg.beta * w + cfg.gamma * idle + cfg.delta * critical
            number("CA priority", priority, zero=True)
            features.append((q, w, idle, critical))
            priorities.append(priority)
        counts, repaired = allocate_counts(cfg.policy, self.phy.num_subcarriers, priorities)
        self.allocations = {}
        offset = 0
        for flow_id, count, priority, feature in zip(ids, counts, priorities, features):
            carriers = tuple(range(offset, offset + count))
            offset += count
            self.allocations[flow_id] = carriers
            self.decisions.append({"generation": self.generation, "time_s": self.loop.now_s,
                "flow_id": flow_id, "subcarriers": carriers, "remaining_bits": self.active[flow_id].remaining_bits,
                "q": feature[0], "w": feature[1], "idle": feature[2], "critical": feature[3],
                "priority": priority, "repaired_overflow": repaired})
        validate_allocations(tuple(Allocation(i, sc) for i, sc in self.allocations.items()), self.phy.num_subcarriers)
        if self.active:
            duration = min(s.setup_left_s + s.remaining_bits / (len(self.allocations[i]) * self.link.subcarrier_rate_bps)
                           for i, s in self.active.items())
            if cfg.policy == "ca":
                duration = min(duration, cfg.epoch_s)
            self.loop.schedule(advance_time(self.loop.now_s, duration), R.WAKE, ref_id=self.generation)

    @property
    def unfinished(self):
        return bool(self.active or self.pending)
