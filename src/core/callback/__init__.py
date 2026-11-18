from ..configs import (
    EarlyStoppingConfig,
    MetricLoggerConfig,
    ModelCheckpointConfig,
    ProgressCallbackConfig,
    VisualizationCallbackConfig,
)
from .base_callback import BaseCallback
from .callback_registry import CallbackRegistry
from .early_stopping import EarlyStopping
from .metric_logger import MetricLogger
from .model_checkpoint import ModelCheckpoint
from .progress_callback import ProgressCallback
from .visualization_callback import VisualizationCallback


def __init_callbacks():
    """Initialize the callback registry with the available callbacks."""
    CallbackRegistry.register("early_stopping", EarlyStopping, EarlyStoppingConfig)
    CallbackRegistry.register("metric_logger", MetricLogger, MetricLoggerConfig)
    CallbackRegistry.register(
        "model_checkpoint", ModelCheckpoint, ModelCheckpointConfig
    )
    CallbackRegistry.register(
        "progress_callback", ProgressCallback, ProgressCallbackConfig
    )
    CallbackRegistry.register(
        "visualization_callback",
        VisualizationCallback,
        VisualizationCallbackConfig,
    )


__init_callbacks()

__version__ = "1.0.0"

__all__ = [
    "BaseCallback",
    "CallbackRegistry",
    "EarlyStopping",
    "MetricLogger",
    "ModelCheckpoint",
    "ProgressCallback",
    "VisualizationCallback"
]