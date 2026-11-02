from tasks import BaseTask
from engine import Evaluator, History
from callback import BaseCallback

import torch
from typing import List
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        model,
        task : BaseTask,
        optimizer : torch.optim.Optimizer,
        scheduler = None,
        callbacks : List[BaseCallback] = None,
        device : str ="cpu",
        grad_clip=None,
        epochs : int = 10
    ):
        ## DataLoaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        ## Create the Evaluator object
        self.evaluator = Evaluator(task=task)
        
        ## Model
        self.model = model.to(device)
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.callbacks = callbacks or []
        self.device = device
        self.grad_clip = grad_clip

        self.epochs = epochs
        self.current_epoch = 0
        
        ## Initialize the history where we will store the metrics for each epoch
        self.history = History()
      
    def _train_epoch(self, epoch : int):
        """_summary_

        Args:
            epoch (int): _description_

        Raises:
            ValueError: _description_
        """
        
        ## Set the model to training mode
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        
        ## Call the callbacks at the beginning of the training epoch
        for callback in self.callbacks:
            callback.on_train_batch_begin(
                trainer=self, 
                batch_idx=epoch
            )

        for batch in enumerate(self.train_loader):
            ## Move the batch data to the specified device
            batch = self.task.to_device(batch, self.device)
            
            ## Reset the gradients of the optimizer to zero before the backward pass
            self.optimizer.zero_grad(set_to_none=True)
            
            ## Forward pass
            outputs = self.task.forward(self.model, batch)
    
            ## Compute the loss
            loss = self.task.compute_loss(outputs, batch)        
            ## Backward pass
            loss.backward()
            
            if self.grad_clip is not None:
                clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip,
                )
            
            ## Update the model parameters
            self.optimizer.step()
            
            ## Update the total loss and total samples for the epoch
            batch_size = self.task.get_batch_size(batch)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            ## Update the metrics for the current batch
            for metric in self.task.train_metrics:
                metric.update(outputs, batch)
                
            ## Call the callbacks at the end of each training batch
            for callback in self.callbacks:
                callback.on_train_batch_end(
                    trainer=self, 
                    batch_idx=epoch
                )
            
        if total_samples == 0:
            raise ValueError("No samples were processed during the training epoch.")
        
        results = {
            "train_loss": total_loss / total_samples,
        }
        
        # Compute epoch metrics
        for metric in self.task.train_metrics:
            results[f"train_{metric.metric_name}"] = metric.compute()

        
        ## Call the callbacks at the end of the training epoch
        for callback in self.callbacks:
            callback.on_epoch_end(
                trainer=self,
                epoch=self.current_epoch,
                metrics=results
            )
        
    def _validate(self, epoch : int):
        """_summary_

        Args:
            epoch (int): _description_

        Returns:
            _type_: _description_
        """
        ## Call the callbacks at the beginning of the validation phase
        for callback in self.callbacks:
            callback.on_validation_begin(self)

        
        results = self.evaluator.evaluate(
            self.model,
            self.val_loader
        )

        results = {
            f"val_{key}": value
            for key, value in results.items()
        }

        for callback in self.callbacks:
            callback.on_validation_end(self, results)

        return results
    
    def fit(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        ## Process Callbacks at the beginning of training
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.epochs):

            self.current_epoch = epoch + 1

            ## Process Callbacks at the beginning of each epoch
            for callback in self.callbacks:
                callback.on_epoch_begin(
                    trainer=self,
                    epoch=self.current_epoch
                )

            ## Training
            self._train_epoch(epoch=self.current_epoch)

            ## Validation
            val_metrics = self._validate(epoch=self.current_epoch)

            ## Update the learning rate scheduler if it is provided
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

        for callback in self.callbacks:
            callback.on_train_end(self)

        return self.history
    
    def test(self):
        """Method to test the model on the test dataloader

        Returns:
            dict: _description_
        """
        ## Call the callbacks at the beginning of the test phase
        for callback in self.callbacks:
            callback.on_test_begin(self)

        results = self.evaluator.evaluate(
            self.model,
            self.test_loader
        )

        results = {
            f"test_{key}": value
            for key, value in results.items()
        }

        for callback in self.callbacks:
            callback.on_test_end(self, results)

        return results
    
    
    
    def get_history(self):
        """Method to get the training history

        Returns:
            History: The training history object.
        """
        return self.history
            
        
        
    