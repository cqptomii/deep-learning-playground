import torch

def compute_accuracy(predictions, targets):
    """
    Computes the accuracy of predictions against the true targets.

    Args:
        predictions (torch.Tensor): The predicted outputs from the model.
        targets (torch.Tensor): The true labels.

    Returns:
        float: The accuracy as a percentage.
    """
    ## Get the predicted class by taking the argmax
    _, predicted_classes = torch.max(predictions, 1)
    
    ## Calculate the number of correct predictions
    correct_predictions = (predicted_classes == targets).sum().item()
    
    ## Calculate accuracy
    accuracy = correct_predictions / targets.size(0) * 100.0
    
    return accuracy