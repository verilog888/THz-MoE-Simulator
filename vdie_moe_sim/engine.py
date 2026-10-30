"""Deterministic heapq discrete-event engine, with same-time FIFO ordering."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import heapq
import itertools
from .config import integer, number


class EventType(str, Enum):
    BASE_TO_SOURCE_DONE = "BASE_TO_SOURCE_DONE"
    GATE_DONE = "GATE_DONE"
    DISPATCH_READY = "DISPATCH_READY"
    DISPATCH_DONE = "DISPATCH_DONE"
    EXPERT_COMPUTE_DONE = "EXPERT_COMPUTE_DONE"
    LOCAL_COMBINE_DONE = "LOCAL_COMBINE_DONE"
    RETURN_READY = "RETURN_READY"
    RETURN_DONE = "RETURN_DONE"
    SOURCE_COMBINE_DONE = "SOURCE_COMBINE_DONE"
    SOURCE_TO_BASE_DONE = "SOURCE_TO_BASE_DONE"


class PipelineEventType(str, Enum):
    TOKEN_ARRIVAL = "TOKEN_ARRIVAL"
    BASE_TO_SOURCE_DONE = "BASE_TO_SOURCE_DONE"
    GATE_DONE = "GATE_DONE"
    DISPATCH_DONE = "DISPATCH_DONE"
    BATCH_DEADLINE = "BATCH_DEADLINE"
    EXPERT_BATCH_DONE = "EXPERT_BATCH_DONE"
    TOKEN_EXPERTS_DONE = "TOKEN_EXPERTS_DONE"


class LayerEventType(str, Enum):
    WEIGHT_DONE = "WEIGHT_DONE"
    LOCAL_REDUCE_DONE = "LOCAL_REDUCE_DONE"
    PARTIAL_READY = "PARTIAL_READY"
    RETURN_DONE = "RETURN_DONE"
    SOURCE_COMBINE_DONE = "SOURCE_COMBINE_DONE"
    SOURCE_TO_BASE_DONE = "SOURCE_TO_BASE_DONE"


class RadioEventType(str, Enum):
    REBALANCE = "RADIO_REBALANCE"
    WAKE = "RADIO_WAKE"
    DONE = "RADIO_FLOW_DONE"


@dataclass(frozen=True)
class Event:
    time_s: float
    kind: EventType | PipelineEventType | LayerEventType | RadioEventType
    token_id: int = 0
    ref_id: int | None = None


class EventLoop:
    def __init__(self):
        self.now_s = 0.0
        self.processed = 0
        self._sequence = itertools.count()
        self._queue: list[tuple[float, int, int, Event]] = []

    def schedule(self, time_s: float, kind: EventType | PipelineEventType | LayerEventType, token_id: int = 0,
                 *, ref_id: int | None = None, priority: int = 0) -> None:
        number("event time", time_s, zero=True)
        integer("token_id", token_id, 0)
        if not isinstance(kind, (EventType, PipelineEventType, LayerEventType, RadioEventType)):
            raise ValueError("event kind must be an EventType")
        if ref_id is not None:
            integer("event ref_id", ref_id, 0)
        if time_s < self.now_s:
            raise ValueError("cannot schedule an event in the past")
        integer("event priority", priority, 0)
        heapq.heappush(self._queue, (time_s, priority, next(self._sequence), Event(time_s, kind, token_id, ref_id)))

    def run(self, handler: Callable[[Event], None], *, max_events: int = 100_000) -> None:
        integer("max_events", max_events)
        processed_this_run = 0
        while self._queue:
            if processed_this_run >= max_events:
                raise RuntimeError("event limit exceeded; possible zero-time event cycle")
            time_s, _, _, event = heapq.heappop(self._queue)
            if time_s < self.now_s:
                raise RuntimeError("event causality violated")
            self.now_s = time_s
            handler(event)
            self.processed += 1
            processed_this_run += 1
