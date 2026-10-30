"""Shared event engine: Dispatch, expert batching, and one RF queue.

Stage 3 extends the hooks below so Return competes with future Dispatch
inside the same event loop, rather than being appended after Stage 2.
"""

from collections import deque
from pathlib import Path

from .batching import BatchExecution, ExpertBatcher, ExpertInput, advance_time
from .combine import FlowReady
from .compute import estimate_expert
from .config import SimulationConfig
from .dispatch import plan_dispatch
from .engine import EventLoop, PipelineEventType as E
from .moe import Flow, Interval, TokenCompletion, Transmission
from .pipeline_metrics import PipelineResult
from .placement import ExpertPlacement
from .thz_phy import THzLink
from .workload import build_workload


class PipelineSimulation:
    def __init__(self, config: SimulationConfig, *, trace_base_dir: str | Path = "."):
        self.config = config
        self.workload = build_workload(config, base_dir=trace_base_dir)
        self.tokens = {t.token_id: t for t in self.workload.tokens}
        self.placement = ExpertPlacement.contiguous(config.model.num_experts, config.topology.num_dies)
        self.plans = {t.token_id: plan_dispatch(t, self.placement) for t in self.workload.tokens}
        self.link = THzLink(config.phy)
        self.loop = EventLoop()
        self.batcher = ExpertBatcher(config.batching)
        self.events, self.intervals, self.inputs, self.executions, self.transmissions = [], [], [], [], []
        self.ignored_deadlines = []
        self.completions = {}
        self.completed_assignments = {t.token_id: set() for t in self.workload.tokens}
        self.ready_batches = [deque() for _ in range(config.topology.num_dies)]
        self.busy = [None] * config.topology.num_dies
        self.batch_executions = {}
        self.radio_queue = deque()
        self.active_tx = None
        self.flow_ready = []

    def start_compute(self, die):
        if self.busy[die] is not None or not self.ready_batches[die]:
            return
        batch = self.ready_batches[die].popleft()
        if batch.ready_s > self.loop.now_s:
            raise RuntimeError("compute cannot start before batch closure")
        estimate = estimate_expert(self.config.model, self.config.compute, len(batch.request_ids))
        end = advance_time(self.loop.now_s, estimate.duration_s)
        execution = BatchExecution(batch, self.loop.now_s, end, estimate.flops, estimate.weight_bytes,
                                   estimate.weight_reads, estimate.compute_s, estimate.memory_s)
        self.executions.append(execution)
        self.batch_executions[batch.batch_id] = execution
        self.busy[die] = batch.batch_id
        first_token = self.inputs[batch.request_ids[0]].token_id
        self.intervals.append(Interval(first_token, "expert_compute", f"Compute die {die}", self.loop.now_s, end, die))
        self.loop.schedule(end, E.EXPERT_BATCH_DONE, first_token, ref_id=batch.batch_id)

    def queue_closed(self, batches):
        for batch in batches:
            self.ready_batches[batch.die_id].append(batch)
            self.start_compute(batch.die_id)

    def admit(self, token_id, expert, local):
        request = ExpertInput(len(self.inputs), token_id, expert, self.placement.die_for(expert), self.loop.now_s, local)
        self.inputs.append(request)
        closed, deadlines = self.batcher.admit(request)
        for time_s, batch_id in deadlines:
            self.loop.schedule(time_s, E.BATCH_DEADLINE, token_id, ref_id=batch_id)
        self.queue_closed(closed)

    def enqueue_flow(self, token_id, src, dsts, kind, *, partial_id=None):
        flow = Flow(len(self.flow_ready), token_id, src, dsts, self.tokens[token_id].activation_bytes, kind)
        self.flow_ready.append(FlowReady(flow.flow_id, self.loop.now_s, partial_id))
        self.radio_queue.append(flow)
        self.start_radio()
        return flow.flow_id

    def radio_completion_kind(self, flow):
        if flow.flow_type != "dispatch":
            raise ValueError("Stage 2 accepts Dispatch only")
        return E.DISPATCH_DONE

    def start_radio(self):
        if self.active_tx is not None or not self.radio_queue:
            return
        flow = self.radio_queue.popleft()
        estimate = self.link.transfer(flow.payload_bytes, self.config.phy.num_subcarriers)
        payload_start = advance_time(self.loop.now_s, estimate.setup_s)
        end = advance_time(payload_start, estimate.payload_s)
        self.active_tx = Transmission(flow, self.loop.now_s, payload_start, end,
                                      tuple(range(self.config.phy.num_subcarriers)), estimate.rate_bps)
        self.transmissions.append(self.active_tx)
        self.intervals.append(Interval(flow.token_id, flow.flow_type, "THz shared spectrum", self.loop.now_s, end))
        self.loop.schedule(end, self.radio_completion_kind(flow), flow.token_id, ref_id=flow.flow_id)

    def complete_radio(self, event):
        if self.active_tx is None or self.active_tx.flow.flow_id != event.ref_id:
            raise RuntimeError("stale or duplicate RF completion")
        self.on_flow_done(self.active_tx)
        self.active_tx = None
        self.start_radio()

    def on_flow_done(self, transmission):
        for _, experts in self.plans[transmission.flow.token_id].remote_groups:
            for expert in experts:
                self.admit(transmission.flow.token_id, expert, False)

    def on_expert_done(self, request):
        """Extension hook; Stage 2 has no output/reduction path."""

    def on_experts_done(self, token_id):
        if token_id in self.completions:
            raise RuntimeError("duplicate experts-done token")
        self.completions[token_id] = TokenCompletion(self.tokens[token_id], self.loop.now_s)

    def handle_extra(self, event):
        raise RuntimeError(f"unhandled pipeline event: {event.kind}")

    def handle(self, event):
        self.events.append(event)
        token = self.tokens[event.token_id]
        if event.kind == E.TOKEN_ARRIVAL:
            end = advance_time(self.loop.now_s, self.config.timing.base_to_source_s)
            self.intervals.append(Interval(token.token_id, "base_to_source", "Base to Source", self.loop.now_s, end))
            self.loop.schedule(end, E.BASE_TO_SOURCE_DONE, token.token_id)
        elif event.kind == E.BASE_TO_SOURCE_DONE:
            end = advance_time(self.loop.now_s, self.config.timing.gate_s)
            self.intervals.append(Interval(token.token_id, "gate", f"Gate die {token.source_die}", self.loop.now_s, end,
                                           token.source_die))
            self.loop.schedule(end, E.GATE_DONE, token.token_id)
        elif event.kind == E.GATE_DONE:
            plan = self.plans[token.token_id]
            for expert in plan.local_experts:
                self.admit(token.token_id, expert, True)
            if plan.remote_dies:
                self.enqueue_flow(token.token_id, token.source_die, plan.remote_dies, "dispatch")
        elif event.kind == E.DISPATCH_DONE:
            self.complete_radio(event)
        elif event.kind == E.BATCH_DEADLINE:
            batch = self.batcher.expire(event.ref_id, self.loop.now_s)
            if batch is None:
                self.ignored_deadlines.append(event.ref_id)
            else:
                self.queue_closed((batch,))
        elif event.kind == E.EXPERT_BATCH_DONE:
            execution = self.batch_executions[event.ref_id]
            batch = execution.batch
            if self.busy[batch.die_id] != batch.batch_id:
                raise RuntimeError("stale or duplicate compute completion")
            for request_id in batch.request_ids:
                request = self.inputs[request_id]
                done = self.completed_assignments[request.token_id]
                if request.expert_id in done:
                    raise RuntimeError("duplicate expert computation")
                done.add(request.expert_id)
                self.on_expert_done(request)
                if len(done) == self.config.model.top_k:
                    self.loop.schedule(self.loop.now_s, E.TOKEN_EXPERTS_DONE, request.token_id)
            self.busy[batch.die_id] = None
            self.start_compute(batch.die_id)
        elif event.kind == E.TOKEN_EXPERTS_DONE:
            self.on_experts_done(token.token_id)
        else:
            self.handle_extra(event)

    def front_result(self, completions, *, radio_context=()):
        return PipelineResult(self.config, self.workload, tuple(self.inputs), tuple(self.executions),
                              tuple(t for t in self.transmissions if t.flow.flow_type == "dispatch"),
                              tuple(completions[i] for i in sorted(completions)),
                              tuple(s for s in self.intervals if s.kind in ("base_to_source", "gate", "dispatch", "expert_compute")),
                              tuple(e for e in self.events if isinstance(e.kind, E)), tuple(self.ignored_deadlines),
                              radio_context=radio_context)

    def make_result(self):
        return self.front_result(self.completions)

    def run(self):
        for token in self.workload.tokens:
            self.loop.schedule(token.arrival_s, E.TOKEN_ARRIVAL, token.token_id)
        limit = max(100_000, len(self.tokens) * (60 + 30 * self.config.model.top_k))
        self.loop.run(self.handle, max_events=limit)
        if (len(self.completions) != len(self.tokens) or self.batcher.pending or any(self.ready_batches)
                or any(b is not None for b in self.busy) or self.radio_queue or self.active_tx is not None):
            raise RuntimeError("unfinished pipeline work")
        result = self.make_result()
        result.audit()
        return result


def run_dispatch_compute(config: SimulationConfig, *, trace_base_dir: str | Path = ".") -> PipelineResult:
    return PipelineSimulation(config, trace_base_dir=trace_base_dir).run()
