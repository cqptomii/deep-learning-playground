from dataclasses import dataclass

from .base import MetricConfig


@dataclass
class PixelAccuracyConfig(MetricConfig):
    type: str = "segmentation"
    name: str = "pixel_accuracy"
    ignore_index: int | None = None
