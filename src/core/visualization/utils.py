from typing import List

from .base_visualizer import BaseVisualizer
from .training_curve_visualizer import TrainingCurveVisualizer

def create_default_visualizers() -> List[BaseVisualizer]:
    """
    Create default visualizers based on the metrics used in the task.
    
    Returns:
        List[BaseVisualizer]: A list of visualizer instances.
    """
    visualizers = []
    
    ##
    ## By default, Two Training curve are created
    ##  - One for the loss metrics
    ##  - One for the accuracy metrics
    ##
            
    training_curve_loss = TrainingCurveVisualizer(
        metrics=[
            "train_loss",
            "val_loss"
        ],
        stage="train",
        plot_title="Training and Validation Loss",
        plot_xlabel="Epoch",
        plot_ylabel="Loss"
    )
    training_curve_accuracy = TrainingCurveVisualizer(
        metrics=[
            "train_accuracy",
            "val_accuracy"
        ],
        stage="train",
        plot_title="Training and Validation Accuracy",
        plot_xlabel="Epoch",
        plot_ylabel="Accuracy"
    )
    
    visualizers.append(training_curve_loss)
    visualizers.append(training_curve_accuracy)
    
    return visualizers