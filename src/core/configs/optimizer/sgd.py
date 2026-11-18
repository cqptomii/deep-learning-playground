from dataclasses import dataclass

from .base import OptimizerConfig


@dataclass
class SGDConfig(OptimizerConfig):
    name: str = "sgd"
    lr: float = 1e-3
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
    maximize: bool = False
    foreach: bool | None = None
    differentiable: bool = False
    fused: bool | None = None
