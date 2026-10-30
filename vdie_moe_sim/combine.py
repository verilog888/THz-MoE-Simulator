"""Weighted-vector semantics and auditable Combine records, not neural FFNs."""

from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np
from .config import integer, number


def checked_vector(value, dimension: int) -> np.ndarray:
    integer("vector dimension", dimension)
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError("vector must be finite and have exactly d_model elements")
    return vector


def weighted_vector(value, weight: float, dimension: int) -> np.ndarray:
    number("gate weight", weight, zero=True)
    with np.errstate(over="raise", invalid="raise"):
        result = checked_vector(value, dimension) * weight
    return checked_vector(result, dimension)


def sum_vectors(values: Iterable[np.ndarray], dimension: int) -> np.ndarray:
    integer("vector dimension", dimension)
    result = np.zeros(dimension, dtype=np.float64)
    count = 0
    with np.errstate(over="raise", invalid="raise"):
        for value in values:
            result += checked_vector(value, dimension)
            count += 1
    if not count:
        raise ValueError("cannot combine an empty set of partials")
    return checked_vector(result, dimension)


def synthetic_expert_output(token_id: int, expert_id: int, dimension: int, seed: int) -> np.ndarray:
    """Assignment-local RNG: output is independent of routing/batching/scheduling order."""
    for name, value in (("token_id", token_id), ("expert_id", expert_id), ("seed", seed)):
        integer(name, value, 0)
    integer("dimension", dimension)
    return np.random.default_rng(np.random.SeedSequence([seed, token_id, expert_id])).uniform(-1, 1, dimension)


@dataclass(frozen=True)
class CombineJob:
    job_id: int
    token_id: int
    die_id: int
    kind: str  # weight, local_reduce, source_reduce
    expert_ids: tuple[int, ...]
    ready_s: float
    start_s: float
    end_s: float


@dataclass(frozen=True)
class PartialVector:
    partial_id: int
    token_id: int
    die_id: int
    expert_ids: tuple[int, ...]
    ready_s: float
    vector: np.ndarray

    def __post_init__(self):
        vector = np.asarray(self.vector, dtype=np.float64).copy()
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("partial vector must be one-dimensional and finite")
        vector.flags.writeable = False
        object.__setattr__(self, "vector", vector)


@dataclass(frozen=True)
class PartialDelivery:
    partial_id: int
    available_s: float
    flow_id: int | None  # None for a Source-local partial; no self-transmission.


@dataclass(frozen=True)
class FlowReady:
    flow_id: int
    ready_s: float
    partial_id: int | None  # None for Dispatch.


@dataclass(frozen=True)
class TokenOutput:
    token_id: int
    ready_s: float
    vector: np.ndarray

    def __post_init__(self):
        vector = np.asarray(self.vector, dtype=np.float64).copy()
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("output vector must be one-dimensional and finite")
        vector.flags.writeable = False
        object.__setattr__(self, "vector", vector)
