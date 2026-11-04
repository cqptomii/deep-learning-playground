from .base_callback import BaseCallback
from visualization import BaseVisualizer
from typing import List

class VisualizationCallback(BaseCallback):

    def __init__(self, visualizers : List[BaseVisualizer], frequency : int = 1, save_path : str = None):
        self.visualizers : List[BaseVisualizer] = visualizers
        self.frequency : int = frequency
        self.save_path : str = save_path

    def on_epoch_end(self, trainer, epoch, metrics):

        if epoch % self.frequency != 0:
            return

        for visualizer in self.visualizers:
            visualizer.update(trainer, metrics)
            
    def on_train_end(self, trainer):
        
        ## Render and save the visualizations at the end of training
        for visualizer in self.visualizers:
            visualizer.render()
            visualizer.save(f"{self.save_path}/{visualizer.__class__.__name__}.png")
            visualizer.clear()