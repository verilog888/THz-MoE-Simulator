"""Only the isolated reference allocator is implemented in stage 1."""

from .config import integer
from .moe import Flow
from .ofdma import Allocation, validate_allocations


class IsolatedFullBandAllocator:
    def __init__(self, num_subcarriers: int):
        integer("num_subcarriers", num_subcarriers)
        self.num_subcarriers = num_subcarriers

    def allocate(self, ready_flows: tuple[Flow, ...]) -> tuple[Allocation, ...]:
        if len(ready_flows) > 1:
            raise ValueError("isolated reference cannot schedule multiple competing flows")
        allocations = tuple(Allocation(f.flow_id, tuple(range(self.num_subcarriers)))
                            for f in ready_flows)
        validate_allocations(allocations, self.num_subcarriers)
        return allocations
