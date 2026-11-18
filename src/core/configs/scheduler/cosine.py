from dataclasses import dataclass

from .base import SchedulerConfig


@dataclass
class CosineAnnealingLRConfig(SchedulerConfig):
    name: str = "cosine"
    T_max: int = 1
    eta_min: float = 0.0
    last_epoch: int = -1
