"""One Block of Expert Dies; no uncalibrated geometry or SNR model."""

from dataclasses import dataclass
from .config import SimulationConfig, integer


@dataclass(frozen=True)
class Topology:
    num_dies: int
    base_die: int = -1

    def __post_init__(self):
        integer("num_dies", self.num_dies)

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "Topology":
        return cls(config.topology.num_dies)

    def check_die(self, die_id: int) -> None:
        integer("die_id", die_id, 0)
        if die_id >= self.num_dies:
            raise ValueError("die_id is outside topology")
