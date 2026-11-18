from dataclasses import dataclass

from .base import MetricConfig


@dataclass
class DiceConfig(MetricConfig):
    type: str = "segmentation"
    name: str = "dice"
    average: str = "none"
    ignore_index: int | None = None
