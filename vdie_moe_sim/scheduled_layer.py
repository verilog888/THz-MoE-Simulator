"""Five controlled dataflow/scheduler variants on the same routed-expert model."""

from dataclasses import replace
from pathlib import Path

from .engine import PipelineEventType as E, RadioEventType as R
from .config import uses_broadcast
from .layer import LayerSimulation
from .radio import SharedRadio


SCHEMES = ("A", "B", "C", "D", "Proposed")


def effective_config(config, scheme=None):
    selected = scheme or config.scheduling.scheme
    return replace(config, scheduling=replace(config.scheduling, scheme=selected),
                   combine=replace(config.combine, local_combine=config.ablation.local_combine if config.ablation else selected not in ("A", "B")))


class ScheduledLayerSimulation(LayerSimulation):
    def __init__(self, config, *, trace_base_dir="."):
        super().__init__(effective_config(config), trace_base_dir=trace_base_dir)
        self.base_id = config.topology.num_dies
        self.radio = SharedRadio(self.loop, config.phy, config.scheduling, self.radio_done, self.idle_fraction)
        self.done_flows = set()

    def idle_fraction(self, state):
        if state.flow.flow_type != "dispatch":
            return 0.0
        dies = state.flow.dsts
        return sum(self.busy[d] is None and not self.ready_batches[d] for d in dies) / len(dies)

    def radio_done(self, state):
        self.loop.schedule(self.loop.now_s, R.DONE, state.flow.token_id, ref_id=state.flow.flow_id)

    def enqueue_flow(self, token_id, src, dsts, kind, *, partial_id=None):
        if self.config.scheduling.scheme == "A" and kind == "return":
            dsts, kind = (self.base_id,), "return_to_base"
        return self.radio.enqueue(token_id, src, dsts, self.tokens[token_id].activation_bytes,
                                  kind, partial_id=partial_id)

    def handle(self, event):
        if isinstance(event.kind, R):
            self.events.append(event)
            if event.kind != R.DONE:
                self.radio.handle(event)
                return
            state = self.radio.states[event.ref_id]
            if event.ref_id in self.done_flows or state.completed_s != self.loop.now_s:
                raise RuntimeError("stale or duplicate logical flow completion")
            self.done_flows.add(event.ref_id)
            flow = state.flow
            token = self.tokens[flow.token_id]
            if flow.flow_type == "dispatch_to_base":
                self.radio.enqueue(token.token_id, self.base_id, (self.placement.die_for(state.expert_ids[0]),),
                    flow.payload_bytes, "dispatch", expert_ids=state.expert_ids, parent_id=flow.flow_id)
            elif flow.flow_type == "return_to_base":
                self.radio.enqueue(token.token_id, self.base_id, (token.source_die,), flow.payload_bytes,
                    "return", partial_id=state.partial_id, parent_id=flow.flow_id)
            elif flow.flow_type == "dispatch":
                for expert in state.expert_ids:
                    self.admit(token.token_id, expert, False)
            elif flow.flow_type == "return":
                self.deliver(self.partials[state.partial_id], flow.flow_id)
            else:
                raise RuntimeError("unknown logical flow")
        elif event.kind == E.GATE_DONE:
            self.events.append(event)
            token, plan = self.tokens[event.token_id], self.plans[event.token_id]
            for expert in plan.local_experts:
                self.admit(token.token_id, expert, True)
            if not uses_broadcast(self.config):
                for die, experts in plan.remote_groups:
                    for expert in experts:
                        relay = self.config.scheduling.scheme == "A"
                        self.radio.enqueue(token.token_id, token.source_die, (self.base_id if relay else die,),
                            token.activation_bytes, "dispatch_to_base" if relay else "dispatch", expert_ids=(expert,))
            elif plan.remote_dies:
                self.radio.enqueue(token.token_id, token.source_die, plan.remote_dies, token.activation_bytes,
                    "dispatch", expert_ids=tuple(sorted(e for _, ids in plan.remote_groups for e in ids)))
        else:
            super().handle(event)

    def run(self):
        for token in self.workload.tokens:
            self.loop.schedule(token.arrival_s, E.TOKEN_ARRIVAL, token.token_id)
        self.loop.run(self.handle, max_events=self.config.scheduling.max_events)
        if (len(self.completions) != len(self.tokens) or self.batcher.pending or any(self.ready_batches)
                or any(b is not None for b in self.busy) or self.radio.unfinished
                or len(self.done_flows) != len(self.radio.states) or self.raw_vectors or self.weighted_vectors
                or any(self.reduction_queues) or any(b is not None for b in self.reduction_busy)
                or len(self.finished_jobs) != len(self.jobs)):
            raise RuntimeError("unfinished scheduled layer work")
        from .scheduled_metrics import ScheduledResult
        result = ScheduledResult.from_simulation(self)
        result.audit()
        return result


def run_scheduled(config, *, trace_base_dir: str | Path = "."):
    return ScheduledLayerSimulation(config, trace_base_dir=trace_base_dir).run()
