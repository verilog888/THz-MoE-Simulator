"""Synthetic assignments, not evaluation of a neural Gate."""

import numpy as np
from .config import integer, number


def uniform_route(num_experts: int, top_k: int,
                  rng: np.random.Generator) -> tuple[tuple[int, ...], tuple[float, ...]]:
    integer("num_experts", num_experts)
    integer("top_k", top_k)
    if top_k > num_experts:
        raise ValueError("top_k exceeds num_experts")
    ids = tuple(int(e) for e in rng.choice(num_experts, size=top_k, replace=False))
    return ids, tuple(1.0 / top_k for _ in ids)


def zipf_probabilities(num_experts: int, skew: float) -> np.ndarray:
    """PDF p.11 probabilities for the first draw; ranks start at 1."""
    integer("num_experts", num_experts)
    number("zipf skew", skew, zero=True)
    probabilities = np.exp(-skew * np.log(np.arange(1, num_experts + 1, dtype=float)))
    return probabilities / probabilities.sum()


def zipf_route(num_experts: int, top_k: int, skew: float,
               rng: np.random.Generator) -> tuple[tuple[int, ...], tuple[float, ...]]:
    integer("top_k", top_k)
    probabilities = zipf_probabilities(num_experts, skew)
    if top_k > num_experts or np.count_nonzero(probabilities) < top_k:
        raise ValueError("not enough representable positive Zipf probabilities for top_k")
    # Weighted sampling without replacement: inclusion probabilities are NOT k*p_i.
    ids = tuple(int(e) for e in rng.choice(num_experts, size=top_k, replace=False, p=probabilities))
    return ids, tuple(1.0 / top_k for _ in ids)
