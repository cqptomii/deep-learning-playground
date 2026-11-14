import torch

from .base_metric import BaseMetric


class ConfusionMatrixMetric(BaseMetric):
    """Counts predictions against labels into a confusion matrix."""

    def __init__(self):
        self.reset()

    @property
    def metric_name(self):
        return "confusion_matrix"

    def reset(self):
        self._matrix = None

    def update(self, outputs, batch):
        predictions = outputs.detach().argmax(dim=1).cpu()
        targets = batch[1].detach().cpu().long()
        observed = torch.cat([predictions, targets])
        size = int(observed.max().item()) + 1 if observed.numel() else 1

        if self._matrix is None:
            self._matrix = torch.zeros((size, size), dtype=torch.int64)
        elif size > self._matrix.shape[0]:
            grown = torch.zeros((size, size), dtype=torch.int64)
            previous = self._matrix.shape[0]
            grown[:previous, :previous] = self._matrix
            self._matrix = grown

        for target, prediction in zip(targets.tolist(), predictions.tolist()):
            self._matrix[target, prediction] += 1

    def compute(self):
        if self._matrix is None:
            return []
        return self._matrix.tolist()
