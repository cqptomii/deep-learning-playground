from abc import ABC
from engine import Trainer

class BaseCallback(ABC):

    def on_train_begin(self, trainer : Trainer):
        """Method call at the beginning of the train cycle

        Args:
            trainer (Trainer): The trainer object that is calling the callback.
        """
        pass


    def on_epoch_begin(self, trainer : Trainer, epoch : int):
        """Method call at the beginning of each epoch

        Args:
            trainer (Trainer): The trainer object that is calling the callback.
            epoch (int): The current epoch.
        """
        pass

    def on_train_batch_begin(self, trainer : Trainer, batch_idx : int):
        """Method call at the beginning of each training batch

        Args:
            trainer (Trainer): The trainer object that is calling the callback.
            batch_idx (int): The index of the current batch.
        """
        pass
    
    def on_validation_begin(self, trainer : Trainer):
        """Method call at the beginning of the validation phase
    
        Args:
            trainer (Trainer): The trainer object that is calling the callback.
        """
        pass

    def on_train_end(self, trainer : Trainer):
        """Method call at the end of the train cycle
    
        Args:
            trainer (Trainer): The trainer object that is calling the callback.
        """
        pass
    
    def on_train_batch_end(self, trainer : Trainer, batch_idx : int):
        """Method call at the end of each training batch

        Args:
            trainer (Trainer): The trainer object that is calling the callback.
            batch_idx (int): The index of the current batch.
            metrics (dict): The metrics for the current batch.
        """
        pass

    def on_validation_end(self, trainer : Trainer, metrics : dict):
        """Method call at the end of the validation phase

        Args:
            trainer (Trainer): The trainer object that is calling the callback.
            metrics (dict): The metrics for the validation phase.
        """
        pass
    
    def on_epoch_end(self, trainer : Trainer, epoch : int, metrics : dict):
        """Method call at the end of each epoch
    
        Args:
            trainer (Trainer): The trainer object that is calling the callback.
            epoch (int): The current epoch.
            metrics (dict): The metrics for the current epoch.
        """
        pass

    def on_exception(self, trainer : Trainer, exception : Exception):
        """Method call when an exception is raised during training

        Args:
            trainer (Trainer): The trainer object that is calling the callback.
            exception (Exception): The exception that was raised.
        """
        pass