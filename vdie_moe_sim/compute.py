"""PDF p.10 roofline equations, with FLOP/s and byte/s units."""

from dataclasses import dataclass
from .config import ComputeConfig, ModelConfig, integer


@dataclass(frozen=True)
class ComputeEstimate:
    flops: int
    weight_bytes: int
    weight_reads: int
    compute_s: float
    memory_s: float

    @property
    def duration_s(self) -> float:
        return max(self.compute_s, self.memory_s)


def estimate_expert(model: ModelConfig, compute: ComputeConfig, num_tokens: int) -> ComputeEstimate:
    integer("num_tokens", num_tokens, 0)
    flops = 6 * model.d_model * model.d_ff * num_tokens
    reuse = compute.weight_reuse_tokens
    reads = 0 if num_tokens == 0 else (1 if reuse is None else (num_tokens + reuse - 1) // reuse)
    return ComputeEstimate(flops, model.weight_bytes, reads,
                           flops / compute.compute_flops_per_s,
                           reads * model.weight_bytes / compute.memory_bytes_per_s)
