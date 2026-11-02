from .base_metric import BaseMetric
import torch


class AccuracyMetric(BaseMetric):
    """
    Class to compute the accuracy of predictions against the true targets.
    Inherits from BaseMetric.
    """
    
    def __init__(self):
        self.reset()
        
    @property
    def metric_name(self):
        return "accuracy"
    
    
    def reset(self):
        """
        Resets the internal state of the metric.
        """
        self.correct = 0
        self.total = 0

    def update(self, outputs, batch):
        """
        Computes the accuracy of predictions against the true targets.

        Args:
            outputs (torch.Tensor): The predicted outputs from the model.
            batch (tuple): A tuple containing the input data and true labels.

        Returns:
            float: The accuracy as a percentage.
        """
        ## Get the predicted class by taking the argmax
        predictions = outputs.detach().argmax(dim=1)
        targets = batch[1].detach()
        
        ## Calculate the number of correct predictions
        self.correct = (predictions == targets).sum().item()
        self.total += targets.numel()
    
    def compute(self):
        if self.total == 0:
            return 0.0

        return self.correct / self.total