from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class DataModule(ABC):
    
    def __init__(self, batch_size=32, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.folds : list = None
        self.current_fold_index = 0

    @abstractmethod
    def setup(self):
        """Create train, validation and test datasets."""
        pass
    
    def next_fold(self):
        """
            Method to move to the next fold in the fold list.
        """
        current_fold = self.folds[self.current_fold_index]
        
        self.train_dataset, self.val_dataset, self.test_dataset = current_fold
        self.current_fold_index += 1
        
        ## If we have reached the end of the folds, reset the index to 0
        self.current_fold_index = self.current_fold_index % len(self.folds)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )