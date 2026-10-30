from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    token_id: int
    source_die: int
    arrival_s: float
    activation_bytes: int
    expert_ids: tuple[int, ...]
    gate_weights: tuple[float, ...]


@dataclass(frozen=True)
class Flow:
    flow_id: int
    token_id: int
    src: int
    dsts: tuple[int, ...]
    payload_bytes: int
    flow_type: str


@dataclass(frozen=True)
class Interval:
    token_id: int
    kind: str
    resource: str
    start_s: float
    end_s: float
    die_id: int = -1


@dataclass(frozen=True)
class Transmission:
    flow: Flow
    start_s: float
    payload_start_s: float
    end_s: float
    subcarriers: tuple[int, ...]
    rate_bps: float


@dataclass(frozen=True)
class TokenCompletion:
    token: Token
    completed_s: float
