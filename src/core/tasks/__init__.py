from .base_task import BaseTask
from .classification_task import ClassificationTask

__version__ = "1.0.0"


def get_task(task_type: str, train_metrics, test_metrics, loss_fn):
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
    if task_type == "classification":
        return ClassificationTask(train_metrics, test_metrics, loss_fn)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


__all__ = [
    "BaseTask",
    "ClassificationTask"
]