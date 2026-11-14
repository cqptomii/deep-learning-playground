from __future__ import annotations

from typing import TYPE_CHECKING

from .base_callback import BaseCallback

if TYPE_CHECKING:
    from ..engine import Trainer

class MetricLogger(BaseCallback):

    """
    Callback class for logging metrics during training, validation and testing phases.
    """
    def on_epoch_end(self, trainer : Trainer, epoch : int, metrics : dict):
        """Method to save the epoch train metrics

        Args:
            trainer (Trainer): Trainer object
            epoch (int): Epoch number
            metrics (dict): Dictionary of metrics
        """
        trainer.history.log_epoch(
            epoch=epoch,
            stage="train",
            metrics=metrics
        )

    def on_validation_end(self, trainer : Trainer, metrics : dict):
        """Method to save the epoch validation metrics

        Args:
            trainer (Trainer): Trainer object
            metrics (dict): Dictionary of metrics
        """
        trainer.history.log_epoch(
            epoch=trainer.current_epoch,
            stage="validation",
            metrics=metrics
        )
        
    def on_test_end(self, trainer : Trainer, metrics : dict):
        """Method to save the epoch test metrics

        Args:
            trainer (Trainer): Trainer object
            metrics (dict): Dictionary of metrics
        """
        trainer.history.log_epoch(
            epoch=trainer.current_epoch,
            stage="test",
            metrics=metrics
        )
    