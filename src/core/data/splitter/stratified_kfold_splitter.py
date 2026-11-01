from splitter import BaseSplitter
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Subset

class StratifiedKFoldSplitter(BaseSplitter):
    
    def __init__(self, n_splits: int = 10, shuffle: bool = True, random_seed: int = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.stratified_kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_seed)
    
    def split(self, dataset, split_ratio):
        folds = []
        indices = list(range(len(dataset)))
        labels = dataset.targets

        ## Split the dataset into train and test sets
        train_idx, test_idx = train_test_split(
            indices,
            test_size=split_ratio[2],
            stratify=labels,
            random_state=self.random_seed
        )
        
        ## Split the train set into k folds
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.stratified_kfold.split(indices, labels)
        ):
            folds.append({
                "train_idx": Subset(dataset, train_idx.tolist()),
                "val_idx": Subset(dataset, val_idx.tolist()),
                "test_idx": Subset(dataset, test_idx.tolist())
            })

        return folds
        
        