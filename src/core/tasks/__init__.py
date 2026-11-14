from ..metrics import get_metrics
from .base_task import BaseTask
from .classification_task import ClassificationTask
from ..loss_registry import get_loss_fn

__version__ = "1.0.0"


def get_task(type: str, train_metrics, test_metrics, loss_fn):
    """
    Factory function to get the appropriate task class based on the task type.

    Args:
        task_type (str): The type of task (e.g., "classification").
        train_metrics (List[BaseMetric]): List of metrics for training.
        test_metrics (List[BaseMetric]): List of metrics for testing.
        loss_fn: Loss function to be used for the task.
    Returns:
        BaseTask: An instance of the appropriate task class.
    """
    if type == "classification":
        return ClassificationTask(
            get_metrics(train_metrics),
            get_metrics(test_metrics),
            get_loss_fn(loss_fn),
        )
    else:
        raise ValueError(f"Unknown task type: {type}")


__all__ = [
    "BaseTask",
    "ClassificationTask"
]