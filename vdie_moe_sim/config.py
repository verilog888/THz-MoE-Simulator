"""Explicit SI units and strict input validation; no implicit hardware defaults."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path


def integer(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def number(name: str, value: float, *, zero: bool = False) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value < 0 if zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def stable_digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 7168
    d_ff: int = 2048
    num_experts: int = 256
    top_k: int = 8
    activation_precision: str = "bf16"
    weight_precision: str = "fp8"

    def __post_init__(self):
        for name in ("d_model", "d_ff", "num_experts", "top_k"):
            integer(name, getattr(self, name))
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.activation_precision not in ("bf16", "fp8"):
            raise ValueError("activation_precision must be bf16 or fp8")
        if self.weight_precision != "fp8":
            raise ValueError("this specification uses FP8 expert weights")

    @property
    def activation_bytes(self) -> int:
        return self.d_model * (2 if self.activation_precision == "bf16" else 1)

    @property
    def weight_bytes(self) -> int:
        return 3 * self.d_model * self.d_ff


@dataclass(frozen=True)
class TopologyConfig:
    num_dies: int = 8

    def __post_init__(self):
        integer("num_dies", self.num_dies)


@dataclass(frozen=True)
class ComputeConfig:
    compute_flops_per_s: float
    memory_bytes_per_s: float
    weight_reuse_tokens: int | None = None

    def __post_init__(self):
        number("compute_flops_per_s", self.compute_flops_per_s)
        number("memory_bytes_per_s", self.memory_bytes_per_s)
        if self.weight_reuse_tokens is not None:
            integer("weight_reuse_tokens", self.weight_reuse_tokens)


@dataclass(frozen=True)
class PHYConfig:
    total_bandwidth_hz: float
    num_subcarriers: int
    spectral_efficiency: float
    phy_efficiency: float
    setup_s: float

    def __post_init__(self):
        for name in ("total_bandwidth_hz", "spectral_efficiency", "phy_efficiency"):
            number(name, getattr(self, name))
        integer("num_subcarriers", self.num_subcarriers)
        number("setup_s", self.setup_s, zero=True)
        if self.phy_efficiency > 1:
            raise ValueError("phy_efficiency cannot exceed 1")


@dataclass(frozen=True)
class TimingConfig:
    base_to_source_s: float
    gate_s: float
    source_combine_s: float
    source_to_base_s: float

    def __post_init__(self):
        for name, value in asdict(self).items():
            number(name, value, zero=True)


@dataclass(frozen=True)
class BatchingConfig:
    window_s: float = 0.0
    max_tokens: int = 16

    def __post_init__(self):
        number("batch window_s", self.window_s, zero=True)
        integer("batch max_tokens", self.max_tokens)


@dataclass(frozen=True)
class CombineConfig:
    local_combine: bool = True
    weight_s: float = 0.0
    local_reduce_s: float = 0.0
    resource_model: str = "per_die_fifo"
    verification_seed: int = 0

    def __post_init__(self):
        if type(self.local_combine) is not bool:
            raise ValueError("local_combine must be a boolean")
        number("weight_s", self.weight_s, zero=True)
        number("local_reduce_s", self.local_reduce_s, zero=True)
        integer("verification_seed", self.verification_seed, 0)
        if self.resource_model not in ("per_die_fifo", "independent"):
            raise ValueError("combine resource_model must be per_die_fifo or independent")


@dataclass(frozen=True)
class AblationConfig:
    broadcast: bool = True
    local_combine: bool = True

    def __post_init__(self):
        if type(self.broadcast) is not bool or type(self.local_combine) is not bool:
            raise ValueError("ablation broadcast/local_combine must be booleans")


@dataclass(frozen=True)
class SchedulingConfig:
    scheme: str = "C"
    epoch_s: float = 1e-5
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    q_ref_bytes: float = 14336.0
    w_ref_s: float = 1e-5
    max_events: int = 1000000

    def __post_init__(self):
        if self.scheme not in ("A", "B", "C", "D", "Proposed"):
            raise ValueError("scheme must be A, B, C, D or Proposed")
        for name in ("epoch_s", "q_ref_bytes", "w_ref_s"):
            number(name, getattr(self, name))
        for name in ("alpha", "beta", "gamma", "delta"):
            number(name, getattr(self, name), zero=True)
        integer("max_events", self.max_events)

    @property
    def policy(self):
        return "tdma" if self.scheme in ("A", "B", "C") else "static" if self.scheme == "D" else "ca"


@dataclass(frozen=True)
class WorkloadConfig:
    num_tokens: int = 1
    seed: int = 0
    arrival_s: float = 0.0
    source_die: int = 0
    routing: str = "fixed"
    expert_ids: tuple[int, ...] = ()
    gate_weights: tuple[float, ...] = ()
    source_policy: str = "fixed"
    interarrival_s: float = 0.0
    zipf_skew: float = 0.0
    trace_file: str | None = None

    def __post_init__(self):
        integer("num_tokens", self.num_tokens)
        integer("seed", self.seed, 0)
        integer("source_die", self.source_die, 0)
        number("arrival_s", self.arrival_s, zero=True)
        number("interarrival_s", self.interarrival_s, zero=True)
        number("zipf_skew", self.zipf_skew, zero=True)
        if self.routing not in ("fixed", "uniform", "zipf", "trace"):
            raise ValueError("routing must be fixed, uniform, zipf or trace")
        if self.source_policy not in ("fixed", "round_robin", "trace"):
            raise ValueError("source_policy must be fixed, round_robin or trace")
        for name in ("expert_ids", "gate_weights"):
            if not isinstance(getattr(self, name), (list, tuple)):
                raise ValueError(f"{name} must be an array")
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for expert in self.expert_ids:
            integer("expert_id", expert, 0)
        if len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("Top-k assignments must be unique")
        if self.routing != "fixed" and (self.expert_ids or self.gate_weights):
            raise ValueError("non-fixed routing cannot also specify fixed assignments")
        if self.routing == "trace":
            if not isinstance(self.trace_file, str) or not self.trace_file.strip():
                raise ValueError("trace routing requires trace_file")
            if self.source_policy != "trace" or self.arrival_s != 0 or self.interarrival_s != 0:
                raise ValueError("trace supplies sources and arrivals; use source_policy=trace and zero synthetic arrivals")
        elif self.trace_file is not None or self.source_policy == "trace":
            raise ValueError("trace fields require routing=trace")
        if self.routing != "zipf" and self.zipf_skew != 0:
            raise ValueError("zipf_skew applies only to zipf routing")
        for weight in self.gate_weights:
            number("gate_weight", weight, zero=True)
        if self.gate_weights and not math.isclose(math.fsum(self.gate_weights), 1.0,
                                                   rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("reference gate_weights must sum to 1")


@dataclass(frozen=True)
class SimulationConfig:
    name: str
    model: ModelConfig
    topology: TopologyConfig
    compute: ComputeConfig
    phy: PHYConfig
    timing: TimingConfig
    workload: WorkloadConfig
    schema_version: int = 1
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    combine: CombineConfig = field(default_factory=CombineConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    ablation: AblationConfig | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty description")
        integer("schema_version", self.schema_version)
        if self.schema_version != 1:
            raise ValueError("unsupported schema_version")
        if self.ablation is not None and self.scheduling.scheme in ("A", "B"):
            raise ValueError("ablation must use C/D/Proposed as PHY selectors; A/B remain canonical")
        if self.model.num_experts % self.topology.num_dies:
            raise ValueError("contiguous balanced placement requires E divisible by ND")
        if self.workload.source_die >= self.topology.num_dies:
            raise ValueError("source_die is outside the topology")
        if self.workload.routing == "fixed":
            if len(self.workload.expert_ids) != self.model.top_k:
                raise ValueError("fixed expert_ids must have top_k entries")
            if any(e >= self.model.num_experts for e in self.workload.expert_ids):
                raise ValueError("expert_id is outside the placement")
            if self.workload.gate_weights and len(self.workload.gate_weights) != self.model.top_k:
                raise ValueError("gate_weights must match expert_ids")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())


def config_from_dict(raw: dict) -> SimulationConfig:
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a JSON object")
    fields = dict(raw)
    sections = {"model": ModelConfig, "topology": TopologyConfig,
                "compute": ComputeConfig, "phy": PHYConfig,
                "timing": TimingConfig, "workload": WorkloadConfig}
    try:
        for name, cls in sections.items():
            fields[name] = cls(**fields[name])
        if "batching" in fields:
            fields["batching"] = BatchingConfig(**fields["batching"])
        if "combine" in fields:
            fields["combine"] = CombineConfig(**fields["combine"])
        if "scheduling" in fields:
            fields["scheduling"] = SchedulingConfig(**fields["scheduling"])
        if fields.get("ablation") is not None:
            fields["ablation"] = AblationConfig(**fields["ablation"])
        return SimulationConfig(**fields)
    except (TypeError, KeyError) as exc:
        raise ValueError(f"invalid or missing configuration field: {exc}") from exc


def load_json(path: str | Path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_pairs)


def load_config(path: str | Path) -> SimulationConfig:
    return config_from_dict(load_json(path))


def uses_broadcast(config):
    return config.ablation.broadcast if config.ablation else config.scheduling.scheme not in ("A", "B")


def scheme_label(config):
    if config.ablation is None:
        return config.scheduling.scheme
    a = config.ablation
    return f"Ablation-{config.scheduling.policy}-b{int(a.broadcast)}-l{int(a.local_combine)}"
