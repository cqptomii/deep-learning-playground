from abc import ABC, abstractmethod
from typing import List
from metrics import BaseMetric
from torch import nn

class BaseTask(ABC):

    def __init__(self, 
                train_metrics : List[BaseMetric], 
                test_metrics : List[BaseMetric],
                loss_fn = nn.CrossEntropyLoss()
    ):
            self.train_metrics = train_metrics
            self.test_metrics = test_metrics
            self.loss_fn = loss_fn
            
    @property
    @abstractmethod
    def train_metrics(self):
        pass
    
    @property
    @abstractmethod
    def test_metrics(self):
        pass

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