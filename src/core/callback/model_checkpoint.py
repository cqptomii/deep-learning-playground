from callback import BaseCallback


class ModelCheckpoint(BaseCallback):
    def __init__(self, save_dir : str, model_name : str = "best_model.pt", monitor='val_loss', mode='min'):
        self.save_path = save_dir + "/" + model_name
        self.monitor = monitor
        self.mode = mode
        
        ## Check if the mode is valid
        if self.mode not in ['min', 'max']:
            raise ValueError("Mode must be either 'min' or 'max'.")
        
        ## Initialize the best score based on the mode
        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')


    def on_validation_end(self, trainer, metrics):
        current_score = metrics.get(self.monitor)
        
        ## Check if the current score is None
        if current_score is None:
            return
        
        ## Check if the current score is better than the best score
        
        ## For 'min' mode, we want to minimize the score
        if self.mode == 'min':
            if current_score < self.best_score:
                last_best_score = self.best_score
                self.best_score = current_score
                trainer.save_model(self.save_path)
                print(f"{self.monitor} improved from {last_best_score} to {current_score}")
        else:
            if current_score > self.best_score:
                last_best_score = self.best_score
                self.best_score = current_score
                trainer.save_model(self.save_path)
                print(f"{self.monitor} improved from {last_best_score} to {current_score}")