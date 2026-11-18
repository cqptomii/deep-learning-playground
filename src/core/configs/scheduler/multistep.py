from dataclasses import dataclass
from collections.abc import Sequence

from .base import SchedulerConfig


@dataclass
class MultiStepLRConfig(SchedulerConfig):
    name: str = "multistep"
    milestones: Sequence[int] = ()
    gamma: float = 0.1
    last_epoch: int = -1