from dataclasses import dataclass

from .base import MetricConfig


@dataclass
class IoUConfig(MetricConfig):
    type: str = "segmentation"
    name: str = "iou"
    average: str = "none"
    ignore_index: int | None = None
