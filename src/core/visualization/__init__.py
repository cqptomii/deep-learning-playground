from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

from .base_visualizer import BaseVisualizer
from .confusion_matrix_visualizer import ConfusionMatrixVisualizer
from .f1_score_visualizer import F1ScoreVisualizer
from .training_curve_visualizer import TrainingCurveVisualizer
from .utils import create_default_visualizers

if TYPE_CHECKING:
    from ..tasks import BaseTask

VISUALIZERS_REGISTRY : Dict[str, Callable] = {
    "training_curve": TrainingCurveVisualizer,
    "confusion_matrix": ConfusionMatrixVisualizer,
    "f1_score": F1ScoreVisualizer
}

def process_visualizers(visualizers_config: dict) -> list[BaseVisualizer]:
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
        for visualizer_name, visualizer_config in visualizers_config.items():

            visualizer_name = visualizer_name.lower()

            if visualizer_name not in VISUALIZERS_REGISTRY:
                raise ValueError(f"Visualizer '{visualizer_name}' is not registered. Available visualizers: {list(VISUALIZERS_REGISTRY.keys())}")
            
            visualizer_class = VISUALIZERS_REGISTRY[visualizer_name]
            visualizer_instance = visualizer_class(**visualizer_config)
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
