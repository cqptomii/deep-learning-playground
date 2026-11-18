from .base_metric import BaseMetric
from ..configs import (
    ConfusionMatrixConfig,
    DiceConfig,
    IoUConfig,
    MetricConfig,
    PixelAccuracyConfig,
)
from .classification import AccuracyMetric
from .components import ConfusionMatrix
from .components.confusion_matrix import ConfusionMatrixMetric
from .metric_registry import MetricRegistry
from .segmentation import DiceMetric, IoUMetric, PixelAccuracyMetric


def __init_metrics():
    """Initialize the metric registry with the available metrics."""
    MetricRegistry.register("classification", "accuracy", AccuracyMetric, MetricConfig)
    MetricRegistry.register(
        "classification",
        "confusion_matrix",
        ConfusionMatrixMetric,
        ConfusionMatrixConfig,
    )
    MetricRegistry.register("segmentation", "iou", IoUMetric, IoUConfig)
    MetricRegistry.register("segmentation", "dice", DiceMetric, DiceConfig)
    MetricRegistry.register(
        "segmentation",
        "pixel_accuracy",
        PixelAccuracyMetric,
        PixelAccuracyConfig,
    )
    
## Initialize the metrics registry
__init_metrics()



def get_metric(metric_type: str, metric_name: str, config: dict | None = None):
    """Fonction to retrieve the metric from the metric registry 

    Args:
        type (str): The type of the metric (e.g., classification, segmentation, detection).
        metric_name (str): The name of the metric to retrieve.
        config (dict, optional): The configuration for the metric. Defaults to {}.
    """
    return MetricRegistry.get(
        {"type": metric_type, "name": metric_name, **(config or {})}
    )


def get_metrics(metrics: list[dict]):
    """
    Function to retrieve a list of metrics from the metric registry.
    
    Args:
        metrics (list[Tuple[str, str]]): A list of tuples containing the type and name of each metric to retrieve.

    Returns:
        list[BaseMetric]: A list of the retrieved metrics.
    """
    built = []
    for metric in metrics or []:
        built.append(get_metric(**metric))
    return built

__version__ = "1.0.0"
__all__ = [
    "AccuracyMetric",
    "BaseMetric",
    "ConfusionMatrix",
    "ConfusionMatrixMetric",
    "DiceMetric",
    "IoUMetric",
    "MetricRegistry",
    "PixelAccuracyMetric",
    "get_metric",
    "get_metrics",
]