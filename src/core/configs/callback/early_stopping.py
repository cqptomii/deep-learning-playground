from dataclasses import dataclass

from .base import CallbackConfig


@dataclass
class EarlyStoppingConfig(CallbackConfig):
    name: str = "early_stopping"
    mode: str = "min"
    monitor: str = "val_loss"
    patience: int = 5
    min_delta: float = 0.0
