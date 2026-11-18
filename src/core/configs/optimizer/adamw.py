from dataclasses import dataclass

from .base import OptimizerConfig


@dataclass
class AdamWConfig(OptimizerConfig):
    name: str = "adamw"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    foreach: bool | None = None
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False
    fused: bool | None = None
