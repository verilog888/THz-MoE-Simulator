"""Stage 3: weighted Local Combine, shared-RF Return, Source Combine, Base."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .batching import advance_time
from .combine import (CombineJob, PartialDelivery, PartialVector, TokenOutput,
                      sum_vectors, synthetic_expert_output, weighted_vector)
from .config import SimulationConfig
from .engine import LayerEventType as L
from .layer_metrics import LayerResult
from .moe import Interval, TokenCompletion
from .pipeline import PipelineSimulation


@dataclass(frozen=True)
class _ReductionTask:
    job_id: int
    token_id: int
    die_id: int
    kind: str
    expert_ids: tuple[int, ...]
    ready_s: float


class LayerSimulation(PipelineSimulation):
    def __init__(self, config, *, trace_base_dir="."):
        super().__init__(config, trace_base_dir=trace_base_dir)
        self.groups = {t.token_id: {d: tuple(sorted(ids)) for d, ids in self.placement.group(t.expert_ids).items()}
                       for t in self.workload.tokens}
        self.weights = {(t.token_id, e): w for t in self.workload.tokens for e, w in zip(t.expert_ids, t.gate_weights)}
        self.expected_partials = {t.token_id: len(self.groups[t.token_id]) if config.combine.local_combine
                                  else len(t.expert_ids) for t in self.workload.tokens}
        self.experts_completions = {}
        self.reduction_queues = [deque() for _ in range(config.topology.num_dies)]
        self.reduction_busy = [None] * config.topology.num_dies
        self.next_job_id = 0
        self.jobs = []
        self.jobs_by_id = {}
        self.finished_jobs = set()
        self.raw_vectors = {}
        self.weighted_vectors = {}
        self.group_scheduled = set()
        self.source_scheduled = set()
        self.partials = []
        self.deliveries = {}
        self.inboxes = {t.token_id: [] for t in self.workload.tokens}
        self.outputs = {}

    def enqueue_reduction(self, token_id, die_id, kind, expert_ids):
        task = _ReductionTask(self.next_job_id, token_id, die_id, kind, tuple(sorted(expert_ids)), self.loop.now_s)
        self.next_job_id += 1
        if self.config.combine.resource_model == "independent":
            self.launch_reduction(task)
        else:
            self.reduction_queues[die_id].append(task)
            self.start_reduction(die_id)

    def start_reduction(self, die_id):
        if self.reduction_busy[die_id] is None and self.reduction_queues[die_id]:
            self.launch_reduction(self.reduction_queues[die_id].popleft())

    def launch_reduction(self, task):
        durations = {"weight": self.config.combine.weight_s,
                     "local_reduce": self.config.combine.local_reduce_s,
                     "source_reduce": self.config.timing.source_combine_s}
        kinds = {"weight": L.WEIGHT_DONE, "local_reduce": L.LOCAL_REDUCE_DONE,
                 "source_reduce": L.SOURCE_COMBINE_DONE}
        end = advance_time(self.loop.now_s, durations[task.kind])
        job = CombineJob(task.job_id, task.token_id, task.die_id, task.kind, task.expert_ids,
                         task.ready_s, self.loop.now_s, end)
        self.jobs.append(job)
        self.jobs_by_id[job.job_id] = job
        if self.config.combine.resource_model == "per_die_fifo":
            self.reduction_busy[job.die_id] = job.job_id
        self.intervals.append(Interval(job.token_id, job.kind, f"Combine die {job.die_id}",
                                       job.start_s, job.end_s, job.die_id))
        self.loop.schedule(end, kinds[job.kind], job.token_id, ref_id=job.job_id)

    def on_expert_done(self, request):
        key = (request.token_id, request.expert_id)
        self.raw_vectors[key] = synthetic_expert_output(*key, self.config.model.d_model,
                                                        self.config.combine.verification_seed)
        self.enqueue_reduction(request.token_id, request.die_id, "weight", (request.expert_id,))

    def on_experts_done(self, token_id):
        if token_id in self.experts_completions:
            raise RuntimeError("duplicate experts completion")
        self.experts_completions[token_id] = TokenCompletion(self.tokens[token_id], self.loop.now_s)

    def radio_completion_kind(self, flow):
        return L.RETURN_DONE if flow.flow_type == "return" else super().radio_completion_kind(flow)

    def on_flow_done(self, transmission):
        if transmission.flow.flow_type == "dispatch":
            super().on_flow_done(transmission)
        else:
            ready = self.flow_ready[transmission.flow.flow_id]
            if ready.partial_id is None:
                raise RuntimeError("Return has no associated partial")
            self.deliver(self.partials[ready.partial_id], transmission.flow.flow_id)

    def produce_partial(self, token_id, die_id, expert_ids, vector):
        partial = PartialVector(len(self.partials), token_id, die_id, tuple(sorted(expert_ids)), self.loop.now_s, vector)
        self.partials.append(partial)
        self.loop.schedule(self.loop.now_s, L.PARTIAL_READY, token_id, ref_id=partial.partial_id)

    def deliver(self, partial, flow_id):
        if partial.partial_id in self.deliveries:
            raise RuntimeError("partial delivered twice")
        self.deliveries[partial.partial_id] = PartialDelivery(partial.partial_id, self.loop.now_s, flow_id)
        inbox = self.inboxes[partial.token_id]
        inbox.append(partial.partial_id)
        if len(inbox) == self.expected_partials[partial.token_id]:
            if partial.token_id in self.source_scheduled:
                raise RuntimeError("Source Combine scheduled twice")
            self.source_scheduled.add(partial.token_id)
            token = self.tokens[partial.token_id]
            self.enqueue_reduction(token.token_id, token.source_die, "source_reduce", token.expert_ids)

    def finish_reduction(self, event):
        job = self.jobs_by_id[event.ref_id]
        if job.job_id in self.finished_jobs:
            raise RuntimeError("Combine job completed twice")
        if self.config.combine.resource_model == "per_die_fifo" and self.reduction_busy[job.die_id] != job.job_id:
            raise RuntimeError("Combine resource completion mismatch")
        self.finished_jobs.add(job.job_id)
        dimension = self.config.model.d_model
        if job.kind == "weight":
            expert, = job.expert_ids
            key = (job.token_id, expert)
            value = weighted_vector(self.raw_vectors.pop(key), self.weights[key], dimension)
            if not self.config.combine.local_combine:
                self.produce_partial(job.token_id, job.die_id, (expert,), value)
            else:
                self.weighted_vectors[key] = value
                group_key = (job.token_id, job.die_id)
                members = self.groups[job.token_id][job.die_id]
                if group_key not in self.group_scheduled and all((job.token_id, e) in self.weighted_vectors for e in members):
                    self.group_scheduled.add(group_key)
                    if len(members) == 1:
                        self.produce_partial(job.token_id, job.die_id, members, self.weighted_vectors.pop(key))
                    else:
                        self.enqueue_reduction(job.token_id, job.die_id, "local_reduce", members)
        elif job.kind == "local_reduce":
            value = sum_vectors((self.weighted_vectors.pop((job.token_id, e)) for e in job.expert_ids), dimension)
            self.produce_partial(job.token_id, job.die_id, job.expert_ids, value)
        elif job.kind == "source_reduce":
            partials = sorted((self.partials[i] for i in self.inboxes[job.token_id]), key=lambda p: (p.die_id, p.expert_ids))
            value = sum_vectors((p.vector for p in partials), dimension)  # No second gate weighting.
            self.outputs[job.token_id] = TokenOutput(job.token_id, self.loop.now_s, value)
            end = advance_time(self.loop.now_s, self.config.timing.source_to_base_s)
            self.intervals.append(Interval(job.token_id, "source_to_base", "Source to Base", self.loop.now_s, end))
            self.loop.schedule(end, L.SOURCE_TO_BASE_DONE, job.token_id)
        if self.config.combine.resource_model == "per_die_fifo":
            self.reduction_busy[job.die_id] = None
            self.start_reduction(job.die_id)

    def handle_extra(self, event):
        if event.kind == L.RETURN_DONE:
            self.complete_radio(event)
        elif event.kind in (L.WEIGHT_DONE, L.LOCAL_REDUCE_DONE, L.SOURCE_COMBINE_DONE):
            self.finish_reduction(event)
        elif event.kind == L.PARTIAL_READY:
            partial = self.partials[event.ref_id]
            source = self.tokens[partial.token_id].source_die
            if partial.die_id == source:
                self.deliver(partial, None)
            else:
                self.enqueue_flow(partial.token_id, partial.die_id, (source,), "return", partial_id=partial.partial_id)
        elif event.kind == L.SOURCE_TO_BASE_DONE:
            if event.token_id in self.completions:
                raise RuntimeError("duplicate Base completion")
            self.completions[event.token_id] = TokenCompletion(self.tokens[event.token_id], self.loop.now_s)
        else:
            super().handle_extra(event)

    def make_result(self):
        if (any(self.reduction_queues) or any(b is not None for b in self.reduction_busy)
                or self.raw_vectors or self.weighted_vectors or len(self.finished_jobs) != len(self.jobs)):
            raise RuntimeError("unfinished Combine work")
        front = self.front_result(self.experts_completions, radio_context=tuple(self.transmissions))
        return LayerResult(front, tuple(self.transmissions), tuple(self.flow_ready), tuple(self.jobs),
                           tuple(self.partials), tuple(self.deliveries[i] for i in sorted(self.deliveries)),
                           tuple(self.outputs[i] for i in sorted(self.outputs)),
                           tuple(self.completions[i] for i in sorted(self.completions)),
                           tuple(self.intervals), tuple(self.events))


def run_layer(config: SimulationConfig, *, trace_base_dir: str | Path = ".") -> LayerResult:
    return LayerSimulation(config, trace_base_dir=trace_base_dir).run()
