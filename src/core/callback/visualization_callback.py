from .base_callback import BaseCallback
from visualization import BaseVisualizer
from typing import List

class VisualizationCallback(BaseCallback):

    def __init__(self, visualizers : List[BaseVisualizer], frequency : int = 1, save_path : str = None):
        self.visualizers : List[BaseVisualizer] = visualizers
        self.frequency : int = frequency
        self.save_path : str = save_path

    def on_epoch_end(self, trainer, epoch, metrics):
        """Method to add the training metrics computed into the different visualizers

        Args:
            trainer (Trainer): The trainer instance.
            epoch (int): The current epoch.
            metrics (_type_): The training metrics.
        """
        if epoch % self.frequency != 0:
            return

        for visualizer in self.visualizers:
            ## Check if the visualizer is with this stage
            if visualizer.stage == "train":
                visualizer.update(trainer, metrics)        
    def on_validation_end(self, trainer, metrics):
        """Method to add the validation metrics computed into the different visualizers

        Args:
            trainer (Trainer): The trainer instance.
            metrics (_type_): The validation metrics.
        """
        if trainer.current_epoch % self.frequency != 0:
            return
        
        for visualizer in self.visualizers:
            ## Check if the visualizer is with this stage
            if visualizer.stage == "validation":
                visualizer.update(trainer, metrics)
    
    def on_train_end(self, trainer):
        """Method to render and save visualization related to stage "train" and "validation"

        Args:
            trainer (Trainer): The trainer instance.
        """
        ## Render and save the visualizations at the end of training for stage "train" and "validation"
        for visualizer in self.visualizers:
            if visualizer.stage in ["train", "validation"]:
                visualizer.render()
                visualizer.save(f"{self.save_path}/{visualizer.__class__.__name__}.png")
                visualizer.clear()  
    def on_test_end(self, trainer, metrics):
        """Method to render and save visualization related to stage "test"

        Args:
            trainer (Trainer): The trainer instance.
            metrics (_type_): The test metrics.
        """
        ## Render and save the visualizations at the end of testing for stage "test"
        for visualizer in self.visualizers:
            if visualizer.stage == "test":
                visualizer.update(trainer, metrics)
                visualizer.render()
                visualizer.save(f"{self.save_path}/{visualizer.__class__.__name__}.png")
                visualizer.clear()