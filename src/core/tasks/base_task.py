from abc import ABC, abstractmethod
from typing import List
from torch import nn

from ..callback import get_callback
from ..metrics import BaseMetric
from ..visualization import BaseVisualizer

class BaseTask(ABC):

    def __init__(self, 
                train_metrics : List[BaseMetric], 
                test_metrics : List[BaseMetric],
                loss_fn = nn.CrossEntropyLoss()
    ):
            self._train_metrics = train_metrics
            self._test_metrics = test_metrics
            self.loss_fn = loss_fn
            
    @property
    def train_metrics(self):
        return self._train_metrics

    @property
    def test_metrics(self):
        return self._test_metrics

    @abstractmethod
    def to_device(self, batch, device):
        pass

    @abstractmethod
    def forward(self, model, batch):
        pass

    @abstractmethod
    def compute_loss(self, outputs, batch):
        pass

    @abstractmethod
    def get_batch_size(self, batch):
        pass
    
    
    def get_callbacks(self, callbacks : list[dict], visualizers : List[BaseVisualizer] = [], save_path : str = "visualizations") -> list:
        """Method to create and return the list of callbacks specified by the user with logging callbacks added to the list.

        Args:
            callbacks (list[dict]): List of callback configurations specified by the user. 
        """
        callback_instances = []
        
        ## Add each callback specified into the list
        for callback_name, callback_config in callbacks.items():
            
            callback_instance = get_callback(callback_name, **callback_config)
            callback_instances.append(callback_instance)
            
        ## Create the metric Logger callback
        metric_logger = get_callback("metric_logger")
        ## Create the visualization callbacks correlated with the metrics used
        vis_callback = get_callback(
            "visualization_callback",
            visualizers=visualizers,
            save_path=save_path,
        )
        
        callback_instances.append(metric_logger)
        callback_instances.append(vis_callback)
        
        
        return callback_instances