from .base_callback import BaseCallback
from .early_stopping import EarlyStopping
from .metric_logger import MetricLogger
from .model_checkpoint import ModelCheckpoint
from .progress_callback import ProgressCallback
from .visualization_callback import VisualizationCallback


def get_callback(callback_name : str, **kwargs) -> BaseCallback:
    """Function to get the callback class based on the callback name.

    Args:
        callback_name (str): The name of the callback to retrieve.
        **kwargs: Additional keyword arguments to pass to the callback constructor.

    Returns:
        BaseCallback: An instance of the requested callback class.
    """
    callbacks = {
        "EarlyStopping": EarlyStopping,
        "MetricLogger": MetricLogger,
        "ModelCheckpoint": ModelCheckpoint,
        "ProgressCallback": ProgressCallback,
        "VisualizationCallback": VisualizationCallback
    }

    if callback_name not in callbacks:
        raise ValueError(f"Callback '{callback_name}' is not registered. Available callbacks: {list(callbacks.keys())}")

    return callbacks[callback_name](**kwargs)

__version__ = "1.0.0"

__all__ = [
    "BaseCallback",
    "EarlyStopping",
    "MetricLogger",
    "ModelCheckpoint",
    "ProgressCallback",
    "VisualizationCallback"
]