from dataclasses import dataclass

from .base import OptimizerConfig


@dataclass
class LBFGSConfig(OptimizerConfig):
    name: str = "lbfgs"
    lr: float = 1.0
    max_iter: int = 20
    max_eval: int | None = None
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: str | None = None
