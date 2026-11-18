from dataclasses import dataclass

from .base import OptimizerConfig


@dataclass
class AdamConfig(OptimizerConfig):
    name: str = "adam"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    foreach: bool | None = None
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False
    fused: bool | None = None
