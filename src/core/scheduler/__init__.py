from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from ..configs.scheduler import SchedulerConfig, StepLRConfig, MultiStepLRConfig, ExponentialLRConfig, CosineAnnealingLRConfig, ReduceLROnPlateauConfig, OneCycleLRConfig
from .scheduler_registry import SchedulerRegistry

def __init_schedulers():
    """Initialize the scheduler registry with common schedulers."""

    SchedulerRegistry.register("steplr", StepLR, StepLRConfig)
    SchedulerRegistry.register("multisteplr", MultiStepLR, MultiStepLRConfig)
    SchedulerRegistry.register("exponentiallr", ExponentialLR, ExponentialLRConfig)
    SchedulerRegistry.register("cosineannealinglr", CosineAnnealingLR, CosineAnnealingLRConfig)
    SchedulerRegistry.register("reducelronplateau", ReduceLROnPlateau, ReduceLROnPlateauConfig)
    SchedulerRegistry.register("onecyclelr", OneCycleLR, OneCycleLRConfig)
    
__init_schedulers()

__version__ = "0.1.0"

__all__ = [
    "SchedulerRegistry",
]