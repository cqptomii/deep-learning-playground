from .base_metric import BaseMetric
from classification import AccuracyMetric
from segmentation import IoUMetric, DiceMetric, PixelAccuracyMetric
from .confusion_matrix import ConfusionMatrixMetric
from .metric_registry import MetricRegistry
from typing import Tuple

def __init_metrics():
    MetricRegistry.register("accuracy", AccuracyMetric)
    MetricRegistry.register("confusion_matrix", ConfusionMatrixMetric)
    MetricRegistry.register("iou", IoUMetric)
    MetricRegistry.register("dice", DiceMetric)
    MetricRegistry.register("pixel_accuracy", PixelAccuracyMetric)
    
## Initialize the metrics registry
__init_metrics()



def get_metric(type : str, metric_name : str, config : dict = {}):
    """Fonction to retrieve the metric from the metric registry 

    Args:
        type (str): The type of the metric (e.g., classification, segmentation, detection).
        metric_name (str): The name of the metric to retrieve.
        config (dict, optional): The configuration for the metric. Defaults to {}.
    """
    return MetricRegistry.get(type, metric_name, config)

def get_metrics(metrics : list[Tuple[str, str]]):
    """
    Function to retrieve a list of metrics from the metric registry.
    
    Args:
        metrics (list[Tuple[str, str]]): A list of tuples containing the type and name of each metric to retrieve.

    Returns:
        list[BaseMetric]: A list of the retrieved metrics.
    """
    built = []
    for metric in metrics or []:
        built.append(get_metric(metric[0], metric[1]))
    return built

__version__ = "1.0.0"
__all__ = ["BaseMetric", "AccuracyMetric", "MetricRegistry", "get_metric", "get_metrics"]