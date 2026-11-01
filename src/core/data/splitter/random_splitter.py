from torch.utils.data import random_split
from splitter import  BaseSplitter

class RandomSplitter(BaseSplitter):
    
    def split(self, dataset, split_ratio: tuple, random_seed: int = None):
        """
        Split the dataset into train, validation, and test sets based on the provided split ratio.

        Args:
            dataset (Dataset): The dataset to be split.
            split_ratio (tuple): A tuple containing the ratios for train, validation, and test splits.
        """
        total_length = len(dataset)
        
        
        ## Calculate the lengths for each split based on the provided ratios
        train_length = int(total_length * split_ratio[0])
        val_length = int(total_length * split_ratio[1])
        test_length = total_length - train_length - val_length

        ## Use random_split to split the dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_length, val_length, test_length]
        )

        return train_dataset, val_dataset, test_dataset
        