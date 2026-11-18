from dataclasses import dataclass

from .base import SchedulerConfig


@dataclass
class ReduceLROnPlateauConfig(SchedulerConfig):
    name: str = "reduce_on_plateau"
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float | list[float] = 0.0
    eps: float = 1e-8
