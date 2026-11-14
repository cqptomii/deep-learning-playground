from .base_metric import BaseMetric
from .accuracy import AccuracyMetric
from .confusion_matrix import ConfusionMatrixMetric
from .metric_registry import MetricRegistry

_METRICS = {
    "accuracy": AccuracyMetric,
    "confusion_matrix": ConfusionMatrixMetric,
}


def get_metric(metric_name : str):
    if metric_name not in _METRICS:
        available = ", ".join(sorted(_METRICS))
        raise ValueError(f"Unknown metric '{metric_name}'. Available metrics: {available}.")
    return _METRICS[metric_name]()

def get_metrics(metrics : list[str]):
    built = []
    for metric in metrics or []:
        built.append(get_metric(metric))
    return built

__version__ = "1.0.0"
__all__ = ["BaseMetric", "AccuracyMetric", "MetricRegistry", "get_metric", "get_metrics"]