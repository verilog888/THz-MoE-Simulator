from dataclasses import dataclass
from .config import integer


@dataclass(frozen=True)
class ExpertPlacement:
    expert_to_die: tuple[int, ...]

    @classmethod
    def contiguous(cls, num_experts: int, num_dies: int) -> "ExpertPlacement":
        integer("num_experts", num_experts)
        integer("num_dies", num_dies)
        if num_experts % num_dies:
            raise ValueError("balanced placement requires E divisible by ND")
        per_die = num_experts // num_dies
        return cls(tuple(e // per_die for e in range(num_experts)))

    def die_for(self, expert_id: int) -> int:
        integer("expert_id", expert_id, 0)
        if expert_id >= len(self.expert_to_die):
            raise ValueError("unknown expert_id")
        return self.expert_to_die[expert_id]

    def group(self, expert_ids: tuple[int, ...]) -> dict[int, tuple[int, ...]]:
        groups: dict[int, list[int]] = {}
        for expert in expert_ids:
            groups.setdefault(self.die_for(expert), []).append(expert)
        return {die: tuple(experts) for die, experts in sorted(groups.items())}
