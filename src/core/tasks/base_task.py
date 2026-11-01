from abc import ABC, abstractmethod


class BaseTask(ABC):

    @property
    @abstractmethod
    def metrics(self):
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