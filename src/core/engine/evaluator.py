from ..tasks import BaseTask

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
        model.eval()

        for metric in self.task.test_metrics:
            metric.reset()
        
        total_loss = 0.0
        total_samples = 0
        
        for batch in dataloader:
            ## Move the inputs and targets to the device
            batch = self.task.to_device(batch, self.device)

            ## Forward pass
            outputs = self.task.forward(model, batch)
            
            ## Compute the loss
            loss = self.task.compute_loss(outputs, batch)
            batch_size = self.task.get_batch_size(batch)
            
            ## Update the total loss and total samples
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            ## Update the metrics for the evaluation
            for metric in self.task.test_metrics:
                metric.update(outputs, batch)

        if total_samples == 0:
            raise ValueError("No samples were processed.")

        results = {
            "loss": total_loss / total_samples,
        }
        
        ## Compute the metrics for the evaluation
        for metric in self.task.test_metrics:
            results[metric.metric_name] = metric.compute()
        
        return results