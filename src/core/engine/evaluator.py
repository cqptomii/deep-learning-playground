from tasks import BaseTask

from typing import List

import torch


class Evaluator:
    def __init__(self, task : BaseTask, device : str):
        self.task = task
        self.device = device

    @torch.no_grad() ## Disable gradient computation for evaluation 
    def evaluate(self, model, dataloader) -> dict:
        """_summary_

        Args:
            model (_type_): _description_
            dataloader (_type_): _description_

        Returns:
            dict: _description_
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        for batch in dataloader:
            inputs, targets = batch
            
            ## Move the inputs and targets to the device
            inputs, targets = self.task.move_to_device(inputs, targets, self.device)
        
            ## Forward pass
            outputs = model(inputs)
            
            ## Compute the loss
            loss = self.task.loss_fn(outputs, targets)
            
            ## Update the total loss and total samples
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            ## Update the metrics for the evaluation
            for metric in self.task.test_metrics:
                metric.update(outputs, targets)

        if total_samples == 0:
            raise ValueError("No samples were processed.")

        results = {
            "test_loss" : total_loss / total_samples,
        }
        
        ## Compute the metrics for the evaluation
        for metric in self.task.test_metrics:
            metric.update(outputs, targets)
            results[f"test_{metric.metric_name}"] = metric.compute()
        
        ## Return the validation metrics
        return results