from .base_metric import BaseMetric
from .utils.confusion_matrix import ConfusionMatrix


class ConfusionMatrixMetric(BaseMetric):

    def __init__(self, ignore_index=None):
        self.confusion_matrix = ConfusionMatrix(ignore_index)

    @property
    def metric_name(self):
        return "confusion_matrix"

    def reset(self):
        self.confusion_matrix.reset()

    def update(self, outputs, batch):
        predictions = outputs.detach().argmax(dim=1)
        targets = batch[1]

        self.confusion_matrix.update(predictions, targets)

    def compute(self):
        matrix = self.confusion_matrix.compute()

        return matrix.tolist() if matrix is not None else []
