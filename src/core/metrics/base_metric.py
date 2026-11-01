from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Base class for all metrics. All metrics should inherit from this class and implement the compute method.
    """
    @abstractmethod
    def compute(self, outputs, targets):
        pass