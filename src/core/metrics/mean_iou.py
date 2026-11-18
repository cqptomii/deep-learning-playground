from .base_metric import BaseMetric
import torch

class MeanIoU(BaseMetric):
    @property
    def metric_name(self) -> str:
        return "mean_iou"

    def reset(self):
        self.intersection = 0
        self.union = 0

    def update(self, preds, target):
        # Convert logits [B, C, H, W] to class indices [B, H, W]
        if preds.ndim == target.ndim + 1:
            preds = preds.argmax(dim=1)

        # Exclude ignored pixels if necessary
        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid = target != self.ignore_index

        intersection = ((preds == target) & valid).sum().item()
        union = (valid.sum().item() + (preds != target).sum().item())

        self.intersection += intersection
        self.union += union

    def compute(self) -> float:
        if self.union == 0:
            return 0.0
        return self.intersection / self.union