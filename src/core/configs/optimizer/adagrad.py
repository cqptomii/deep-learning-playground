from dataclasses import dataclass

from .base import OptimizerConfig


@dataclass
class AdagradConfig(OptimizerConfig):
    name: str = "adagrad"
    lr: float = 1e-2
    lr_decay: float = 0.0
    weight_decay: float = 0.0
    initial_accumulator_value: float = 0.0
    eps: float = 1e-10
    foreach: bool | None = None
    maximize: bool = False
    differentiable: bool = False
