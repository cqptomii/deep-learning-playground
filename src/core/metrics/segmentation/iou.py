
import torch

from ..base_metric import BaseMetric
from ..components import ConfusionMatrix


class IOUMetric(BaseMetric):

    def __init__(
        self,
        average="none",
        ignore_index=None
    ):
        if average not in ("none", "macro", "weighted"):
            raise ValueError(f"Invalid average: {average}")

        self.average = average
        self.confusion_matrix = ConfusionMatrix(ignore_index)

    @property
    def metric_name(self):
        return "iou"

    def reset(self):
        self.confusion_matrix.reset()

    def update(self, outputs, batch):
        predictions = outputs.detach().argmax(dim=1)
        targets = batch[1]

        self.confusion_matrix.update(predictions, targets)

    def compute(self):

        matrix = self.confusion_matrix.compute()

        if matrix is None:
            return torch.tensor([])

        matrix = matrix.float()

        tp = matrix.diag()
        fp = matrix.sum(dim=0) - tp
        fn = matrix.sum(dim=1) - tp

        union = tp + fp + fn

        valid = union > 0

        iou = tp / union.clamp(min=1)

        if self.average == "none":
            return iou

        if not valid.any():
            return torch.tensor(0.)

        if self.average == "macro":
            return iou[valid].mean()

        if self.average == "weighted":
            weights = matrix.sum(dim=1)
            weights = weights[valid]
            weights = weights / weights.sum()

            return (iou[valid] * weights).sum()
