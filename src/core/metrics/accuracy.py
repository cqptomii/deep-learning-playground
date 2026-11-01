from .base_metric import BaseMetric
import torch


class AccuracyMetric(BaseMetric):
    """
    Class to compute the accuracy of predictions against the true targets.
    Inherits from BaseMetric.
    """
    
    def compute(self, outputs, targets):
        """
        Computes the accuracy of predictions against the true targets.

        Args:
            outputs (torch.Tensor): The predicted outputs from the model.
            targets (torch.Tensor): The true labels.

        Returns:
            float: The accuracy as a percentage.
        """
        ## Get the predicted class by taking the argmax
        _, predicted_classes = torch.max(outputs, 1)
        
        ## Calculate the number of correct predictions
        correct_predictions = (predicted_classes == targets).sum().item()
        
        ## Calculate accuracy
        accuracy = correct_predictions / targets.size(0) * 100.0
        
        return accuracy