from abc import ABC, abstractmethod

class BaseSplitter(ABC):
    
    def __init__(self, random_seed: int = None):
        self.random_seed = random_seed
    
    @abstractmethod
    def split(self, dataset, split_ratio : tuple) -> list:
        pass