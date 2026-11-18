from torch.optim import (
    SGD, Adam, AdamW, RMSprop, Adagrad, LBFGS
)
from core.configs.optimizer import SGDConfig, AdamConfig, AdamWConfig, RMSpropConfig, AdagradConfig, LBFGSConfig
from .optimizer_registry import OptimizerRegistry

def __init_optimizers():
    """Initialize the optimizer registry with common optimizers."""
    from .optimizer_registry import OptimizerRegistry

    OptimizerRegistry.register("sgd", SGD, SGDConfig)
    OptimizerRegistry.register("adam", Adam, AdamConfig)
    OptimizerRegistry.register("adamw", AdamW, AdamWConfig)
    OptimizerRegistry.register("rmsprop", RMSprop, RMSpropConfig)
    OptimizerRegistry.register("adagrad", Adagrad, AdagradConfig)
    OptimizerRegistry.register("lbfgs", LBFGS, LBFGSConfig)
    
__init_optimizers()

__version__ = "0.1.0"

__all__ = [
    "OptimizerRegistry",
]