from dataclasses import dataclass

from .base import OptimizerConfig


@dataclass
class RMSpropConfig(OptimizerConfig):
    name: str = "rmsprop"
    lr: float = 1e-3
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = False
    maximize: bool = False
    foreach: bool | None = None
    differentiable: bool = False