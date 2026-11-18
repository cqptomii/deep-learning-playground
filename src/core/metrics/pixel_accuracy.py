
import torch

from .base_metric import BaseMetric
from .confusion_matrix import ConfusionMatrix


class PixelAccuracyMetric(BaseMetric):

    def __init__(self, ignore_index=None):
        self.confusion_matrix = ConfusionMatrix(ignore_index)

    @property
    def metric_name(self):
        return "pixel_accuracy"

    def reset(self):
        self.confusion_matrix.reset()

    def update(self, outputs, batch):
        predictions = outputs.detach().argmax(dim=1)
        targets = batch[1]

        self.confusion_matrix.update(predictions, targets)

    def compute(self):

        matrix = self.confusion_matrix.compute()

        if matrix is None:
            return 0.0

        correct = matrix.diag().sum()
        total = matrix.sum()

        return (correct.float() / total).item() if total > 0 else 0.0
