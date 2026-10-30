"""Per-expert finite windows; closed batches later compete for shared Die compute."""

from dataclasses import dataclass, field
import math
from .config import BatchingConfig, integer, number


def advance_time(start: float, duration: float) -> float:
    stop = start + duration
    if not math.isfinite(stop) or (duration > 0 and stop <= start):
        raise ValueError("time precision/overflow: duration cannot be represented")
    return stop


@dataclass(frozen=True)
class ExpertInput:
    request_id: int
    token_id: int
    expert_id: int
    die_id: int
    ready_s: float
    local: bool

    def __post_init__(self):
        for name in ("request_id", "token_id", "expert_id", "die_id"):
            integer(name, getattr(self, name), 0)
        number("expert ready_s", self.ready_s, zero=True)
        if type(self.local) is not bool:
            raise ValueError("local must be a boolean")


@dataclass(frozen=True)
class ClosedBatch:
    batch_id: int
    expert_id: int
    die_id: int
    request_ids: tuple[int, ...]
    opened_s: float
    ready_s: float
    close_order: int
    close_reason: str


@dataclass(frozen=True)
class BatchExecution:
    batch: ClosedBatch
    start_s: float
    end_s: float
    flops: int
    weight_bytes: int
    weight_reads: int
    compute_s: float
    memory_s: float


@dataclass
class _OpenBatch:
    batch_id: int
    expert_id: int
    die_id: int
    opened_s: float
    deadline_s: float
    requests: list[int] = field(default_factory=list)


class ExpertBatcher:
    """Window [first arrival, deadline); cap closes early; zero window is singleton.

    ``admit`` returns closed batches and new deadline events to register.
    ``expire`` tolerates stale deadlines after an early cap closure.
    """

    def __init__(self, config: BatchingConfig):
        self.config = config
        self._by_expert: dict[int, _OpenBatch] = {}
        self._by_id: dict[int, _OpenBatch] = {}
        self._next_id = 0
        self._close_order = 0
        self._seen: set[int] = set()
        self._latest_s = 0.0

    @property
    def pending(self) -> int:
        return sum(len(batch.requests) for batch in self._by_id.values())

    def _close(self, batch: _OpenBatch, time_s: float, reason: str) -> ClosedBatch:
        del self._by_id[batch.batch_id]
        del self._by_expert[batch.expert_id]
        closed = ClosedBatch(batch.batch_id, batch.expert_id, batch.die_id, tuple(batch.requests),
                             batch.opened_s, time_s, self._close_order, reason)
        self._close_order += 1
        return closed

    def admit(self, request: ExpertInput) -> tuple[tuple[ClosedBatch, ...], tuple[tuple[float, int], ...]]:
        if request.request_id in self._seen or request.ready_s < self._latest_s:
            raise ValueError("duplicate or out-of-order expert request")
        self._seen.add(request.request_id)
        self._latest_s = request.ready_s
        closed = []
        current = self._by_expert.get(request.expert_id)
        # Enforce half-open windows even if an arrival wins a same-time FIFO tie.
        if current is not None and request.ready_s >= current.deadline_s:
            closed.append(self._close(current, current.deadline_s, "window"))
            current = None
        deadlines = []
        if current is None:
            current = _OpenBatch(self._next_id, request.expert_id, request.die_id, request.ready_s,
                                 advance_time(request.ready_s, self.config.window_s))
            self._next_id += 1
            self._by_expert[request.expert_id] = current
            self._by_id[current.batch_id] = current
            if self.config.window_s > 0:
                deadlines.append((current.deadline_s, current.batch_id))
        if current.die_id != request.die_id:
            raise ValueError("an expert cannot span multiple Dies")
        current.requests.append(request.request_id)
        if self.config.window_s == 0:
            closed.append(self._close(current, request.ready_s, "immediate"))
        elif len(current.requests) == self.config.max_tokens:
            closed.append(self._close(current, request.ready_s, "cap"))
        return tuple(closed), tuple(deadlines)

    def expire(self, batch_id: int, time_s: float) -> ClosedBatch | None:
        current = self._by_id.get(batch_id)
        if current is None:
            return None
        if time_s != current.deadline_s:
            raise ValueError("batch deadline fired at incorrect time")
        return self._close(current, time_s, "window")
