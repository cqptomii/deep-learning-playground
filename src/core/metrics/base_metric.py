from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Base class for all metrics. All metrics should inherit from this class and implement the compute method.
    """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, outputs, batch):
        pass

    @abstractmethod
    def compute(self) -> float:
        pass