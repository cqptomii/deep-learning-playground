from .base_visualizer import BaseVisualizer
from .training_curve_visualizer import TrainingCurveVisualizer
from .confusion_matrix_visualizer import ConfusionMatrixVisualizer
from .f1_score_visualizer import F1ScoreVisualizer
from .utils import create_default_visualizers
from core.tasks import BaseTask

from typing import Dict, Callable

VISUALIZERS_REGISTRY : Dict[str, Callable] = {
    "TrainingCurveVisualizer": TrainingCurveVisualizer,
    "ConfusionMatrixVisualizer": ConfusionMatrixVisualizer,
    "F1ScoreVisualizer": F1ScoreVisualizer
}

def process_visualizers(visualizers_config: list, task : BaseTask) -> list[BaseVisualizer]:
    """Method to create each visualizers provided by the user in config and default visualizers based on the metrics used in the task.

    Args:
        visualizers_config (list): A list of visualizer configurations specified by the user.
        task (BaseTask): The task instance containing the metrics used for training and validation.

    Returns:
        list[BaseVisualizer]: A list of visualizer instances, including both user-specified and default visualizers.
    """
    
    visualizers = []
    
    ## Create default visualizers
    default_visualizers = create_default_visualizers()
    
    ## Add default visualizers to the list
    visualizers.extend(default_visualizers)
    
    
    ##
    ## Process the visualizers provided by the user
    ##
    try:
        for visualizer_config in visualizers_config.keys():
            name = visualizer_config
            config = visualizers_config.get(name, {})
            
            if name not in VISUALIZERS_REGISTRY:
                raise ValueError(f"Visualizer '{name}' is not registered. Available visualizers: {list(VISUALIZERS_REGISTRY.keys())}")
            
            visualizer_class = VISUALIZERS_REGISTRY[name]
            visualizer_instance = visualizer_class(task=task, **config)
            visualizers.append(visualizer_instance)
    except Exception as e:
        raise ValueError(f"Error processing visualizers configuration: {e}")
    
    return visualizers


__version__ = "1.0.0"

__all__ = [
    "BaseVisualizer",
    "TrainingCurveVisualizer",
    "ConfusionMatrixVisualizer",
    "F1ScoreVisualizer"
]
