"""One activation per token broadcast; local assignments never enter RF."""

from dataclasses import dataclass
from .moe import Token
from .placement import ExpertPlacement


@dataclass(frozen=True)
class DispatchPlan:
    token_id: int
    source_die: int
    local_experts: tuple[int, ...]
    remote_groups: tuple[tuple[int, tuple[int, ...]], ...]

    @property
    def remote_dies(self) -> tuple[int, ...]:
        return tuple(die for die, _ in self.remote_groups)


def plan_dispatch(token: Token, placement: ExpertPlacement) -> DispatchPlan:
    groups = placement.group(token.expert_ids)
    return DispatchPlan(token.token_id, token.source_die, groups.get(token.source_die, ()),
                        tuple((die, ids) for die, ids in groups.items() if die != token.source_die))
