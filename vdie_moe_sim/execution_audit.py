"""Shared independent checks for expert batching and shared Die execution."""

from collections import Counter, defaultdict

from .batching import advance_time
from .compute import estimate_expert
from .engine import PipelineEventType as E


def audit_batches(config, inputs, execution_trace):
    from .pipeline_metrics import _equal
    requests = {r.request_id: r for r in inputs}
    expected_events, expected_intervals = {}, []
    executions = {x.batch.batch_id: x for x in execution_trace}
    if len(executions) != len(execution_trace):
        raise ValueError("batch executed more than once")
    members = Counter(r for x in execution_trace for r in x.batch.request_ids)
    if members != Counter(requests.keys()):
        raise ValueError("lost or duplicated batch membership")
    if sorted(x.batch.close_order for x in execution_trace) != list(range(len(execution_trace))):
        raise ValueError("batch close order is not unique/complete")
    per_die = defaultdict(list)
    per_expert = defaultdict(list)
    token_last = defaultdict(float)
    for execution in execution_trace:
        batch = execution.batch
        members = [requests[i] for i in batch.request_ids]
        if not 1 <= len(members) <= config.batching.max_tokens:
            raise ValueError("batch size exceeds configured cap")
        if any(r.expert_id != batch.expert_id or r.die_id != batch.die_id for r in members):
            raise ValueError("batch mixes experts or Dies")
        _equal(batch.opened_s, min(r.ready_s for r in members), "batch opened time")
        deadline = advance_time(batch.opened_s, config.batching.window_s)
        if config.batching.window_s == 0:
            if batch.close_reason != "immediate" or len(members) != 1:
                raise ValueError("zero-window batches must be singleton")
            _equal(batch.ready_s, batch.opened_s, "immediate batch closure")
        else:
            if any(r.ready_s >= deadline for r in members):
                raise ValueError("request included outside half-open batch window")
            if batch.close_reason == "cap":
                if len(members) != config.batching.max_tokens:
                    raise ValueError("early closed batch has not reached cap")
                _equal(batch.ready_s, max(r.ready_s for r in members), "cap closure")
            elif batch.close_reason == "window":
                if len(members) == config.batching.max_tokens:
                    raise ValueError("full batch should close at cap")
                _equal(batch.ready_s, deadline, "deadline closure")
            else:
                raise ValueError("unknown batch close reason")
            expected_events[(E.BATCH_DEADLINE, members[0].token_id, batch.batch_id)] = deadline
        if execution.start_s < batch.ready_s:
            raise ValueError("batch execution precedes closure")
        estimate = estimate_expert(config.model, config.compute, len(members))
        if (execution.flops, execution.weight_bytes, execution.weight_reads) != (
                estimate.flops, estimate.weight_bytes, estimate.weight_reads):
            raise ValueError("batch FLOP/weight accounting mismatch")
        _equal(execution.compute_s, estimate.compute_s, "compute model")
        _equal(execution.memory_s, estimate.memory_s, "memory model")
        _equal(execution.end_s, advance_time(execution.start_s, estimate.duration_s), "compute completion")
        per_die[batch.die_id].append(execution)
        per_expert[batch.expert_id].append(batch)
        expected_events[(E.EXPERT_BATCH_DONE, members[0].token_id, batch.batch_id)] = execution.end_s
        expected_intervals.append((members[0].token_id, "expert_compute", execution.start_s,
                                   execution.end_s, batch.die_id))
        for request in members:
            token_last[request.token_id] = max(token_last[request.token_id], execution.end_s)
    for executions_on_die in per_die.values():
        previous_end = 0.0
        for execution in sorted(executions_on_die, key=lambda x: (x.batch.ready_s, x.batch.close_order)):
            _equal(execution.start_s, max(previous_end, execution.batch.ready_s), "shared Die FIFO/work conservation")
            previous_end = execution.end_s
    for expert, batches in per_expert.items():
        batches.sort(key=lambda b: b.close_order)
        if any(a.ready_s > b.opened_s for a, b in zip(batches, batches[1:])):
            raise ValueError("an expert has overlapping batch windows")
        admitted = sorted((r for r in inputs if r.expert_id == expert), key=lambda r: (r.ready_s, r.request_id))
        if [i for b in batches for i in b.request_ids] != [r.request_id for r in admitted]:
            raise ValueError("per-expert request FIFO violated")
    return expected_events, expected_intervals, token_last
