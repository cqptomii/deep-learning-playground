from abc import ABC, abstractmethod

class BaseVisualizer(ABC):
    """
    Base interface class for all Visualizers
    """
    
    def __init__(self, stage : str = "train"):
        self.stage = stage
    
    @abstractmethod
    def update(self, trainer, metrics):
        """
        Update the visualizer with new metrics.
        Args:
            trainer: The trainer object that is calling the visualizer.
            metrics (dict): A dictionary containing the latest metrics.
        """
        pass
    
    @abstractmethod
    def render(self):
        """
        Render the visualizer.
        """
        pass
        
    @abstractmethod
    def save(self, path: str):
        """
        Save the rendered visualizer to a file.
        Args:
            path (str): The file path where the visualizer should be saved.
        """
        pass
    
    @abstractmethod
    def clear(self):
        pass