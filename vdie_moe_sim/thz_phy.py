"""Fixed-MCS packet model; setup reserves the allocated spectrum (A08)."""

from dataclasses import dataclass
from .config import PHYConfig, integer


@dataclass(frozen=True)
class TransferEstimate:
    payload_bits: int
    rate_bps: float
    setup_s: float
    payload_s: float

    @property
    def duration_s(self) -> float:
        return self.setup_s + self.payload_s


@dataclass(frozen=True)
class THzLink:
    config: PHYConfig

    @property
    def subcarrier_rate_bps(self) -> float:
        c = self.config
        return c.total_bandwidth_hz / c.num_subcarriers * c.spectral_efficiency * c.phy_efficiency

    def transfer(self, payload_bytes: int, num_subcarriers: int) -> TransferEstimate:
        integer("payload_bytes", payload_bytes, 0)
        integer("num_subcarriers", num_subcarriers)
        if num_subcarriers > self.config.num_subcarriers:
            raise ValueError("allocation exceeds total THz spectrum")
        rate = num_subcarriers * self.subcarrier_rate_bps
        # No bytes means no packet and no setup, rather than a phantom transfer.
        return TransferEstimate(payload_bytes * 8, rate,
                                self.config.setup_s if payload_bytes else 0.0,
                                payload_bytes * 8 / rate)
