from torchvision import datasets
from torch.utils.data import DataLoader

class DatasetZoo:
    """A class that provides access to pre-trained datasets from torchvision."""

    @staticmethod
    def get_dataset(dataset_name: str,
                    batch_size: int = 32, 
                    shuffle: bool = True,
                    num_workers: int = 0,
                    **kwargs):
        """
        Get a pre-trained dataset from torchvision.

        Args:
            dataset_name (str): The name of the dataset to retrieve.
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data in the DataLoader.
            num_workers (int): The number of worker processes to use for data loading.
            **kwargs: Additional keyword arguments to pass to the dataset constructor.
        Returns:
            DataLoader: A DataLoader for the requested dataset.
        """
        dataset = datasets.__dict__[dataset_name](
            **kwargs
            )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )