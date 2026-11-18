from dataclasses import dataclass

from .base import CallbackConfig


@dataclass
class MetricLoggerConfig(CallbackConfig):
    name: str = "metric_logger"
