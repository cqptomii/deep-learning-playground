from base_splitter import BaseSplitter
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

class StratifiedSplitter(BaseSplitter):
    
    def __init__(self, random_seed: int = None):
        super().__init__(random_seed=random_seed)
    
    def split(self, dataset, split_ratio: tuple):
        
        train_ratio, val_ratio, test_ratio = split_ratio
        
        indices = list(range(len(dataset)))
        labels = dataset.targets
        
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=train_ratio + val_ratio,
            stratify=labels,
            random_state=self.random_seed
        )
        
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=[labels[i] for i in temp_idx],
            random_state=self.random_seed
        )
        
        return {
            "train_idx" : Subset(dataset, train_idx.tolist()),
            "val_idx" : Subset(dataset, val_idx.tolist()),
            "test_idx" : Subset(dataset, test_idx.tolist())
        }