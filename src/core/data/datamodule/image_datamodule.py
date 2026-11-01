from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

from splitter import SplitterRegistry, BaseSplitter
from datamodule import DataModule

class ImageDataModule(DataModule):

    def __init__(
        self,
        root,
        batch_size=32,
        num_workers=4,
        train_transform=None,
        eval_transform=None,
        split_strategy='random',
        split_ratio=(0.8, 0.15, 0.15)
    ):
        """ DataModule initializer method

        Args:
            root (str): Path to the root directory of the dataset.
            batch_size (int, optional): The size of the batch to use for training. Defaults to 32.
            num_workers (int, optional): The number of workers to use for data loading. Defaults to 4.
            train_transform (_type_, optional): The transform to apply to the training data. Defaults to None.
            eval_transform (_type_, optional): The transform to apply to the evaluation data. Defaults to None.
            split_strategy (str, optional): The strategy to use for splitting the data. Defaults to 'random'.
            split_ratio (tuple, optional): The ratio of the data to use for each split. Defaults to (0.8, 0.15, 0.15).
        """
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        
        self.root = root
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.split_strategy = split_strategy
        self.split_ratio = split_ratio
        
        ## Get the splitter based on the strategy
        self.splitter : BaseSplitter = SplitterRegistry.get_splitter(split_strategy)

    def setup(self):
        """
        Create train, validation and test datasets.
        """
        ## Load the dataset
        dataset = ImageFolder(
            root=self.root,
            transform=self.train_transform
        )
        
        ## Split the dataset (TO ADD)
        self.train_dataset, self.val_dataset, self.test_dataset = self.splitter.split(dataset, self.split_ratio)