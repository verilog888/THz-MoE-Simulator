from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import numpy as np
from .config import SimulationConfig, integer, load_json, number, stable_digest
from .moe import Token
from .placement import ExpertPlacement
from .routing import uniform_route, zipf_route


def make_reference_token(config: SimulationConfig, placement: ExpertPlacement) -> Token:
    if config.workload.num_tokens != 1:
        raise ValueError("stage 1 reference supports exactly one token; multi-token is stage 2")
    if config.model.top_k != 1:
        raise ValueError("stage 1 reference supports top_k=1; multi-expert is stage 2")
    workload = config.workload
    if workload.routing not in ("fixed", "uniform") or workload.source_policy != "fixed" or config.batching.window_s:
        raise ValueError("stage 1 does not support trace/Zipf/source policies or batch windows; select stage2")
    if workload.routing == "uniform":
        ids, weights = uniform_route(config.model.num_experts, config.model.top_k,
                                     np.random.default_rng(workload.seed))
    else:
        ids = workload.expert_ids
        weights = workload.gate_weights or tuple(1.0 / len(ids) for _ in ids)
    if placement.die_for(ids[0]) == workload.source_die:
        raise ValueError("stage 1 requires a remote expert; local dispatch bypass is stage 2")
    return Token(0, workload.source_die, workload.arrival_s,
                 config.model.activation_bytes, ids, weights)


@dataclass(frozen=True)
class Workload:
    tokens: tuple[Token, ...]
    trace_sha256: str | None = None

    @property
    def digest(self) -> str:
        return stable_digest([asdict(t) for t in sorted(self.tokens, key=lambda t: t.token_id)])

    def to_trace(self) -> dict:
        return {"schema_version": 1, "tokens": [
            {key: value for key, value in asdict(t).items() if key != "activation_bytes"}
            for t in sorted(self.tokens, key=lambda t: (t.arrival_s, t.token_id))]}


def validate_workload(workload: Workload, config: SimulationConfig) -> None:
    if len(workload.tokens) != config.workload.num_tokens:
        raise ValueError("trace token count differs from num_tokens")
    seen = set()
    for token in workload.tokens:
        integer("token_id", token.token_id, 0)
        integer("source_die", token.source_die, 0)
        number("arrival_s", token.arrival_s, zero=True)
        if token.token_id in seen:
            raise ValueError("duplicate token_id")
        seen.add(token.token_id)
        if token.source_die >= config.topology.num_dies:
            raise ValueError("trace source_die outside topology")
        if token.activation_bytes != config.model.activation_bytes:
            raise ValueError("activation size differs from model")
        if len(token.expert_ids) != config.model.top_k or len(token.gate_weights) != config.model.top_k:
            raise ValueError("trace assignments/weights must have top_k entries")
        for expert in token.expert_ids:
            integer("expert_id", expert, 0)
            if expert >= config.model.num_experts:
                raise ValueError("trace expert_id outside placement")
        if len(set(token.expert_ids)) != len(token.expert_ids):
            raise ValueError("duplicate expert in Top-k")
        for weight in token.gate_weights:
            number("gate weight", weight, zero=True)
        # Real trace weights are preserved, including non-normalized weights.


def build_workload(config: SimulationConfig, *, base_dir: str | Path = ".") -> Workload:
    w = config.workload
    if w.routing == "trace":
        path = Path(base_dir) / w.trace_file
        raw = load_json(path)
        if (not isinstance(raw, dict) or set(raw) != {"schema_version", "tokens"}
                or type(raw["schema_version"]) is not int or raw["schema_version"] != 1
                or not isinstance(raw["tokens"], list)):
            raise ValueError("trace must contain schema_version=1 and a tokens array")
        tokens = []
        for row in raw["tokens"]:
            if not isinstance(row, dict) or set(row) != {"token_id", "source_die", "arrival_s", "expert_ids", "gate_weights"}:
                raise ValueError("trace token fields must be token_id, source_die, arrival_s, expert_ids, gate_weights")
            if not isinstance(row["expert_ids"], list) or not isinstance(row["gate_weights"], list):
                raise ValueError("trace assignments/weights must be arrays")
            tokens.append(Token(row["token_id"], row["source_die"], row["arrival_s"],
                                config.model.activation_bytes, tuple(row["expert_ids"]), tuple(row["gate_weights"])))
        workload = Workload(tuple(tokens), hashlib.sha256(path.read_bytes()).hexdigest())
    else:
        rng = np.random.default_rng(w.seed)
        tokens = []
        for i in range(w.num_tokens):
            if w.routing == "uniform":
                ids, weights = uniform_route(config.model.num_experts, config.model.top_k, rng)
            elif w.routing == "zipf":
                ids, weights = zipf_route(config.model.num_experts, config.model.top_k, w.zipf_skew, rng)
            else:
                ids, weights = w.expert_ids, w.gate_weights or tuple(1 / config.model.top_k for _ in w.expert_ids)
            source = w.source_die if w.source_policy == "fixed" else (w.source_die + i) % config.topology.num_dies
            arrival = w.arrival_s + i * w.interarrival_s
            if not math.isfinite(arrival) or (i > 0 and w.interarrival_s > 0 and arrival <= tokens[-1].arrival_s):
                raise ValueError("synthetic arrivals exceed time precision/range")
            tokens.append(Token(i, source, arrival, config.model.activation_bytes, ids, weights))
        workload = Workload(tuple(tokens))
    validate_workload(workload, config)
    return Workload(tuple(sorted(workload.tokens, key=lambda t: (t.arrival_s, t.token_id))), workload.trace_sha256)
