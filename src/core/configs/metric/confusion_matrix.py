from dataclasses import dataclass

from .base import MetricConfig


@dataclass
class ConfusionMatrixConfig(MetricConfig):
    type: str = "classification"
    name: str = "confusion_matrix"
    ignore_index: int | None = None
