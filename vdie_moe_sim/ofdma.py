"""Allocation invariants shared with later OFDMA schedulers."""

from dataclasses import dataclass
from .config import integer


@dataclass(frozen=True)
class Allocation:
    flow_id: int
    subcarriers: tuple[int, ...]


def validate_allocations(allocations: tuple[Allocation, ...], total: int) -> None:
    integer("total", total)
    used: set[int] = set()
    flows: set[int] = set()
    for allocation in allocations:
        integer("flow_id", allocation.flow_id, 0)
        if allocation.flow_id in flows:
            raise ValueError("flow allocated more than once")
        flows.add(allocation.flow_id)
        if not allocation.subcarriers:
            raise ValueError("an admitted flow needs at least one subcarrier")
        for sc in allocation.subcarriers:
            integer("subcarrier", sc, 0)
            if sc >= total:
                raise ValueError("subcarrier outside total spectrum")
            if sc in used:
                raise ValueError("subcarrier collision / duplicate allocation")
            used.add(sc)
