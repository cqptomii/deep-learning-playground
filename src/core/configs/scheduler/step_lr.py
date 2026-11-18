from dataclasses import dataclass

from .base import SchedulerConfig


@dataclass
class StepLRConfig(SchedulerConfig):
    name: str = "step"
    step_size: int = 1
    gamma: float = 0.1
    last_epoch: int = -1