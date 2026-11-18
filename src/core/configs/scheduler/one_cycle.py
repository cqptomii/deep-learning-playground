from dataclasses import dataclass

from .base import SchedulerConfig


@dataclass
class OneCycleLRConfig(SchedulerConfig):
    name: str = "one_cycle"
    max_lr: float | list[float] = 1e-3
    total_steps: int | None = None
    epochs: int | None = None
    steps_per_epoch: int | None = None
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    three_phase: bool = False
    last_epoch: int = -1
