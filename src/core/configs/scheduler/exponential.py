from dataclasses import dataclass

from .base import SchedulerConfig


@dataclass
class ExponentialLRConfig(SchedulerConfig):
    name: str = "exponential"
    gamma: float = 0.1
    last_epoch: int = -1
