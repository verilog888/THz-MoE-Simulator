"""Strict, preflighted experiment plans; deterministic one-factor or grid points."""

from dataclasses import asdict, dataclass, field, replace
import itertools
from pathlib import Path
import re

from .config import SimulationConfig, config_from_dict, integer, load_config, load_json, number, stable_digest


AXES = frozenset((
    "workload.num_tokens", "topology.num_dies", "model.top_k", "phy.num_subcarriers",
    "phy.total_bandwidth_hz", "phy.spectral_efficiency", "workload.zipf_skew", "model.activation_precision",
    "ablation.broadcast", "ablation.local_combine", "scheduling.epoch_s", "scheduling.alpha",
    "scheduling.beta", "scheduling.gamma", "scheduling.delta", "compute.compute_flops_per_s",
    "compute.memory_bytes_per_s", "compute.weight_reuse_tokens", "batching.window_s", "batching.max_tokens",
    "combine.weight_s", "combine.local_reduce_s"))


@dataclass(frozen=True)
class StatisticsConfig:
    bootstrap_resamples: int = 2000
    confidence: float = .95
    seed: int = 1729

    def __post_init__(self):
        integer("bootstrap_resamples", self.bootstrap_resamples, 0)
        if self.bootstrap_resamples and self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be zero (disabled) or at least 100")
        number("confidence", self.confidence)
        if self.confidence >= 1:
            raise ValueError("confidence must be between zero and one")
        integer("statistics seed", self.seed, 0)


@dataclass(frozen=True)
class SweepSpec:
    name: str
    base_config: str
    axes: dict
    seeds: tuple[int, ...]
    mode: str = "one_factor"
    study: str = "five_scheme"
    timelines: str = "representative"
    max_runs: int = 10000
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("sweep name must be nonempty")
        if not isinstance(self.base_config, str) or not self.base_config.strip():
            raise ValueError("sweep requires base_config")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported sweep schema_version")
        if self.mode not in ("one_factor", "grid") or self.study not in ("five_scheme", "ablation"):
            raise ValueError("mode must be one_factor/grid; study must be five_scheme/ablation")
        if self.timelines not in ("none", "representative", "all"):
            raise ValueError("timelines must be none, representative or all")
        integer("max_runs", self.max_runs)
        if not isinstance(self.seeds, (list, tuple)) or not self.seeds:
            raise ValueError("seeds must be a nonempty array")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        for seed in self.seeds:
            integer("routing seed", seed, 0)
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate routing seed is not an independent repetition")
        if not isinstance(self.axes, dict) or not self.axes:
            raise ValueError("axes must be a nonempty object")
        cleaned = {}
        for path, values in sorted(self.axes.items()):
            if path not in AXES:
                raise ValueError(f"unsupported scan axis: {path}")
            if path.startswith("ablation.") and self.study != "ablation":
                raise ValueError("broadcast/Local Combine scans require study=ablation")
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"axis {path} requires nonempty values")
            if any(not isinstance(v, (str, int, float, bool, type(None))) for v in values):
                raise ValueError("axis values must be JSON scalars")
            if any(v in values[:i] for i, v in enumerate(values)):
                raise ValueError(f"duplicate axis value: {path}")
            cleaned[path] = tuple(values)
        object.__setattr__(self, "axes", cleaned)

    @property
    def series(self):
        return ("C", "D", "Proposed") if self.study == "ablation" else ("A", "B", "C", "D", "Proposed")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SweepPoint:
    point_id: str
    config: SimulationConfig
    coordinates: dict


@dataclass(frozen=True)
class SweepPlan:
    spec: SweepSpec
    base: SimulationConfig
    points: tuple[SweepPoint, ...]
    views: tuple[dict, ...]
    trace_base_dir: Path

    @property
    def run_count(self):
        return len(self.points) * len(self.spec.seeds) * len(self.spec.series)

    def preview(self):
        return {"name": self.spec.name, "mode": self.spec.mode, "study": self.spec.study,
                "unique_parameter_points": len(self.points), "plot_coordinates": len(self.views),
                "routing_seeds": list(self.spec.seeds), "series": list(self.spec.series),
                "simulation_runs": self.run_count, "timelines": self.spec.timelines,
                "axes": self.spec.axes, "statistics": asdict(self.spec.statistics)}


def spec_from_dict(raw):
    if not isinstance(raw, dict):
        raise ValueError("sweep specification must be an object")
    fields = dict(raw)
    try:
        fields["statistics"] = StatisticsConfig(**fields.get("statistics", {}))
        return SweepSpec(**fields)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid sweep field: {exc}") from exc


def set_axes(base, changes, study):
    raw = base.to_dict()
    # This field is an explicit Stage 5 ablation, never a canonical baseline.
    if study == "ablation" and not raw.get("ablation"):
        raw["ablation"] = {"broadcast": True, "local_combine": True}
    for path, value in changes.items():
        section, name = path.split(".")
        raw[section][name] = value
    return config_from_dict(raw)


def compile_plan(spec, base, *, trace_base_dir="."):
    if spec.study == "five_scheme" and getattr(base, "ablation", None) is not None:
        raise ValueError("five_scheme base must not have ablation settings")
    if spec.study == "ablation" and base.scheduling.scheme in ("A", "B"):
        raise ValueError("ablation uses direct C/D/Proposed PHY policies, not relay/P2P baseline labels")
    if base.workload.routing in ("fixed", "trace") and len(spec.seeds) > 1:
        raise ValueError("fixed/trace routing ignores seed; use one seed, not pseudo-replication")
    if "workload.zipf_skew" in spec.axes and base.workload.routing != "zipf":
        raise ValueError("Zipf skew scans require Zipf routing in base_config")
    axes = list(spec.axes)
    if spec.mode == "one_factor":
        candidates = (({path: value}, path, value) for path in axes for value in spec.axes[path])
    else:
        size = 1
        for values in spec.axes.values():
            size *= len(values)
        if size * len(spec.seeds) * len(spec.series) > spec.max_runs:
            raise ValueError("grid exceeds explicit max_runs; inspect or raise the configured limit")
        candidates = ((dict(zip(axes, values)), "grid", i) for i, values in enumerate(itertools.product(*(spec.axes[p] for p in axes))))
    points, views = {}, []
    for changes, axis, value in candidates:
        config = set_axes(base, changes, spec.study)
        # All repetitions replace the base routing seed. Canonicalize it for point identity.
        config = replace(config, workload=replace(config.workload, seed=0))
        point_id = "p_" + stable_digest(numeric_identity(config.to_dict()))[:16]
        if point_id in points and points[point_id].config != config:
            raise ValueError("point ID hash collision")
        if point_id not in points:
            coords = {p: config.to_dict()[p.split('.')[0]][p.split('.')[1]] for p in axes}
            points[point_id] = SweepPoint(point_id, config, coords)
        views.append({"axis": axis, "value": value, "point_id": point_id,
                      "changes": changes, "view_order": len(views)})
    plan = SweepPlan(spec, base, tuple(points.values()), tuple(views), Path(trace_base_dir))
    if plan.run_count > spec.max_runs:
        raise ValueError("sweep exceeds explicit max_runs")
    return plan


def numeric_identity(value):
    """JSON 32e9 and 32000000000 denote the same experimental coordinate."""
    if type(value) is float and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k:numeric_identity(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [numeric_identity(v) for v in value]
    return value


def load_plan(path):
    path = Path(path).resolve()
    spec = spec_from_dict(load_json(path))
    base_path = (path.parent / spec.base_config).resolve()
    return compile_plan(spec, load_config(base_path), trace_base_dir=base_path.parent)
