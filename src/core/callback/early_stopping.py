from callback import BaseCallback

class EarlyStopping(BaseCallback):
    def __init__(self,
                 mode='min',
                 metric='val_loss',
                 patience=5, 
                 min_delta=0.0
    ):
        self.mode = mode
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        
        
        
        ## Check if the mode is valid
        if self.mode not in ['min', 'max']:
            raise ValueError("Mode must be either 'min' or 'max'.")
        
        ## Initialize the best loss
        if self.mode == 'min':
            self.best_loss = float('inf')
        else:
            self.best_loss = float('-inf')
            
        self.counter = 0
        
    def on_validation_end(self, trainer, metrics):
        
        current_metric = metrics.get(self.metric)
        
        ## Check if the current metric is None
        if current_metric is None:
            return
        
        ## Check if the current metric is better than the best metric
        
        ## For 'min' strategy, we want to minimize the metric
        if self.mode == 'min':
            if current_metric < self.best_loss - self.min_delta:
                self.best_loss = current_metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered. No improvement in {self.metric} for {self.patience} consecutive validation checks.")
                    trainer.stop_training = True
        
        ## For 'max' strategy, we want to maximize the metric
        else:
            if current_metric > self.best_loss + self.min_delta:
                self.best_loss = current_metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered. No improvement in {self.metric} for {self.patience} consecutive validation checks.")
                    trainer.stop_training = True