from abc import ABC, abstractmethod

class ModelBuilder(ABC):
    @abstractmethod
    def build(self, config : dict):
        pass