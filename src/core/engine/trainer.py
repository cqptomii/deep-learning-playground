from .history import History
from tasks import BaseTask
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(
        self,
        model,
        task : BaseTask,
        optimizer,
        evaluator,
        scheduler = None,
        callbacks = None,
        device : str ="cpu",
        grad_clip=None,
    ):
        self.model = model.to(device)
        self.task = task
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.scheduler = scheduler

        self.callbacks = callbacks or []
        self.device = device
        self.grad_clip = grad_clip

        self.current_epoch = 0
        
        ## Initialize the history where we will store the metrics for each epoch
        self.history = History()
      
    def train_epoch(self, train_loader):
        
        ## Set the model to training mode
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0        

        for batch in train_loader:
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
            
        if total_samples == 0:
            raise ValueError("No samples were processed during the training epoch.")

        results = {
            "train_loss": total_loss / total_samples,
        }
        
        ## Compute and insert metrics into the results dictionary (TO DO)
        
        return results
            
    
    def fit(self, train_loader, val_loader, epochs):

        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(epochs):

            self.current_epoch = epoch + 1

            for callback in self.callbacks:
                callback.on_epoch_begin(self)

            # Training
            train_metrics = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.evaluator.evaluate(
                self.model,
                val_loader,
            )

            ## Update the learning rate scheduler if it is provided
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            for callback in self.callbacks:
                callback.on_epoch_end(self)

        for callback in self.callbacks:
            callback.on_train_end(self)

        return self.history
            
            
        
        
    