from dataclasses import dataclass

@dataclass
class OptimizerConfig:
    name: str
    lr: float = 1e-3
    weight_decay: float = 0.0