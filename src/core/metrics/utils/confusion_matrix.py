
import torch


class ConfusionMatrix:

    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self._matrix = None

    @torch.no_grad()
    def update(self, predictions, targets):

        predictions = predictions.detach().long()
        targets = targets.detach().long()

        # Convert predictions and targets to flat CPU tensors
        predictions = predictions.reshape(-1).cpu()
        targets = targets.reshape(-1).cpu()

        if predictions.shape != targets.shape:
            raise ValueError(
                "Predictions and targets must have the same shape."
            )

        # Ignore selected target classes
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            predictions = predictions[valid]
            targets = targets[valid]

        if targets.numel() == 0:
            return

        if torch.any(predictions < 0) or torch.any(targets < 0):
            raise ValueError("Class indices must be non-negative.")

        # Determine the required matrix size
        observed = torch.cat([predictions, targets])
        required_size = int(observed.max().item()) + 1

        current_size = (
            self._matrix.shape[0]
            if self._matrix is not None
            else 0
        )

        num_classes = max(required_size, current_size)

        # Initialize or expand the matrix
        if self._matrix is None or num_classes > current_size:

            matrix = torch.zeros(
                (num_classes, num_classes),
                dtype=torch.int64
            )

            if self._matrix is not None:
                matrix[:current_size, :current_size] = self._matrix

            self._matrix = matrix

        # Accumulate confusion matrix
        indices = targets * num_classes + predictions

        batch_matrix = torch.bincount(
            indices,
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)

        self._matrix += batch_matrix

    def compute(self):
        return self._matrix.clone() if self._matrix is not None else None