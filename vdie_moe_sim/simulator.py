"""Stage 1 only: one token, one remote routed expert, no contention."""

import math

from .compute import estimate_expert
from .config import SimulationConfig
from .engine import Event, EventLoop, EventType as E
from .metrics import SimulationResult
from .moe import Flow, Interval, TokenCompletion, Transmission
from .placement import ExpertPlacement
from .scheduler import IsolatedFullBandAllocator
from .thz_phy import THzLink
from .topology import Topology
from .workload import make_reference_token


def run_reference(config: SimulationConfig) -> SimulationResult:
    topology = Topology.from_config(config)
    placement = ExpertPlacement.contiguous(config.model.num_experts, topology.num_dies)
    token = make_reference_token(config, placement)
    destination = placement.die_for(token.expert_ids[0])
    topology.check_die(destination)
    topology.check_die(token.source_die)
    link = THzLink(config.phy)
    allocator = IsolatedFullBandAllocator(config.phy.num_subcarriers)
    compute = estimate_expert(config.model, config.compute, 1)
    loop = EventLoop()
    events: list[Event] = []
    intervals: list[Interval] = []
    transmissions: list[Transmission] = []
    completions: list[TokenCompletion] = []

    def finish_time(start, duration):
        stop = start + duration
        if not math.isfinite(stop) or (duration > 0 and stop <= start):
            raise ValueError("time precision/overflow: duration cannot be represented at this arrival")
        return stop

    def stage(kind, duration, next_event, resource, die=-1):
        stop = finish_time(loop.now_s, duration)
        intervals.append(Interval(token.token_id, kind, resource, loop.now_s, stop, die))
        loop.schedule(stop, next_event, token.token_id)

    def transmit(kind, src, dst, next_event):
        flow = Flow(len(transmissions), token.token_id, src, (dst,), token.activation_bytes, kind)
        allocation, = allocator.allocate((flow,))
        estimate = link.transfer(flow.payload_bytes, len(allocation.subcarriers))
        payload_start = finish_time(loop.now_s, estimate.setup_s)
        stop = finish_time(payload_start, estimate.payload_s)
        transmissions.append(Transmission(flow, loop.now_s, payload_start, stop,
                                          allocation.subcarriers, estimate.rate_bps))
        intervals.append(Interval(token.token_id, kind, f"THz {src}->{dst}", loop.now_s, stop))
        loop.schedule(stop, next_event, token.token_id)

    def handle(event):
        events.append(event)
        if event.kind == E.BASE_TO_SOURCE_DONE:
            stage("gate", config.timing.gate_s, E.GATE_DONE, f"Gate die {token.source_die}", token.source_die)
        elif event.kind == E.GATE_DONE:
            loop.schedule(loop.now_s, E.DISPATCH_READY, token.token_id)
        elif event.kind == E.DISPATCH_READY:
            transmit("dispatch", token.source_die, destination, E.DISPATCH_DONE)
        elif event.kind == E.DISPATCH_DONE:
            stage("expert_compute", compute.duration_s, E.EXPERT_COMPUTE_DONE,
                  f"Compute die {destination}", destination)
        elif event.kind == E.EXPERT_COMPUTE_DONE:
            # One output only: a zero-time causal marker, NOT an implemented multi-expert reduction.
            stage("local_combine_singleton_bypass", 0, E.LOCAL_COMBINE_DONE,
                  f"Local combine die {destination}", destination)
        elif event.kind == E.LOCAL_COMBINE_DONE:
            loop.schedule(loop.now_s, E.RETURN_READY, token.token_id)
        elif event.kind == E.RETURN_READY:
            transmit("return", destination, token.source_die, E.RETURN_DONE)
        elif event.kind == E.RETURN_DONE:
            stage("source_combine", config.timing.source_combine_s, E.SOURCE_COMBINE_DONE,
                  f"Source combine die {token.source_die}", token.source_die)
        elif event.kind == E.SOURCE_COMBINE_DONE:
            stage("source_to_base", config.timing.source_to_base_s, E.SOURCE_TO_BASE_DONE, "Source to Base")
        elif event.kind == E.SOURCE_TO_BASE_DONE:
            completions.append(TokenCompletion(token, loop.now_s))
        else:
            raise RuntimeError(f"unhandled event: {event.kind}")

    # Arrival is the observation origin, not an extra waiting time in token latency.
    initial_end = finish_time(token.arrival_s, config.timing.base_to_source_s)
    intervals.append(Interval(token.token_id, "base_to_source", "Base to Source", token.arrival_s, initial_end))
    loop.schedule(initial_end, E.BASE_TO_SOURCE_DONE, token.token_id)
    loop.run(handle)
    if [event.kind for event in events] != list(E) or len(completions) != 1:
        raise RuntimeError("incomplete or duplicate reference-path events")
    result = SimulationResult(config, tuple(completions), tuple(transmissions), tuple(intervals), tuple(events))
    result.summary  # Audit payload, lifetimes and occupancy before returning a successful result.
    return result
