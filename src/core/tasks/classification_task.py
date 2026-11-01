from tasks import BaseTask
from torch import nn

class ClassificationTask(BaseTask):
    
    def __init__(self, metrics, loss_fn = nn.CrossEntropyLoss()):
        self.metrics = metrics
        self.loss_fn = loss_fn
        
    def to_device(self, batch, device : str):
        """Method to move the batch data to the specified device

        Args:
            batch (_type_): Batch data containing inputs and labels
            device (str): device to which the batch data should be moved
        """
        inputs, labels = batch
        
        return inputs.to(device), labels.to(device)
    
    def forward(self, model, batch):
        """Method to perform a forward pass through the model

        Args:
            model (_type_): The model to perform the forward pass
            batch (_type_): Batch data containing inputs and labels

        Returns:
            _type_: Model outputs after the forward pass
        """
        inputs, _ = batch
        return model(inputs)
    
    def compute_loss(self, outputs, batch):
        """Method to compute the loss between model outputs and true labels

        Args:
            outputs (_type_): Model outputs after the forward pass
            batch (_type_): Batch data containing inputs and labels

        Returns:
            _type_: Computed loss value
        """
        _, labels = batch
        
        return self.loss_fn(outputs, labels)