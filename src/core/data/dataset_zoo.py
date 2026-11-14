from torchvision import datasets
from torch.utils.data import DataLoader
import torch
from typing import Any, Callable, Tuple
from .splitter import SplitterRegistry


class DatasetZoo:
    """A class that provides access to pre-trained datasets from torchvision."""

    _DATASET_LIBRARIES: dict[str, Any] = {
        "torchvision": datasets,
    }
    
    @classmethod
    def get_dataset_class(
        cls,
        dataset_name: str,
        library: str = "torchvision",
    ) -> Callable[..., Any]:
        """ Class Method to get the constructor of the dataset class

        Args:
            dataset_name (str): The name of the dataset to retrieve.
            library (str, optional): The library from which to retrieve the dataset. Defaults to "torchvision".

        Raises:
            ValueError: If the dataset is not found in the specified library.
            ValueError: If the library is not supported.

        Returns:
            Callable[..., Any]: The constructor of the dataset class.
        """
        try:
            dataset_library = cls._DATASET_LIBRARIES[library]
            
        ## Handle the case where the library is not supported
        except KeyError as error:
            supported = ", ".join(sorted(cls._DATASET_LIBRARIES.keys()))
            raise ValueError(
                f"Unsupported dataset library: {library!r}. "
                f"Supported libraries: {supported}."
            ) from error

        ## Get the dataset class from the specified library
        dataset_class = getattr(dataset_library, dataset_name, None)

        ## Handle the case where the dataset class is not found or is not callable
        if dataset_class is None or not callable(dataset_class):
            raise ValueError(
                f"Dataset {dataset_name!r} was not found in {library!r}."
            )

        return dataset_class
    
    
    @classmethod
    def get_dataloaders(
        self,
        dataset_name: str,
        library: str = "torchvision",
        root: str = "./data",
        batch_size: int = 32,
        num_workers: int = 0,
        download: bool = False,
        validation_ratio: Tuple[float, float] = (0.7, 0.15),
        split_strategy: str = "random",
        **dataset_kwargs: Any,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """ Get DataLoaders for the specified dataset, splitting it into training, validation, and test sets.

        Args:
            dataset_name (str): _description_
            library (str, optional): _description_. Defaults to "torchvision".
            root (str, optional): _description_. Defaults to "./data".
            batch_size (int, optional): _description_. Defaults to 32.
            num_workers (int, optional): _description_. Defaults to 0.
            validation_ratio (float, optional): _description_. Defaults to 0.15.
            split_strategy (str, optional): _description_. Defaults to "random".
            download (bool, optional): _description_. Defaults to False.
            random_seed (int | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: _description_
        """
        if not 0 < validation_ratio < 1:
            raise ValueError("validation_ratio must be between 0 and 1.")

        dataset_class = self.get_dataset_class(dataset_name, library)
        
        ## Get the transforms from the dataset_kwargs if they exist
        train_transform = dataset_kwargs.pop("train_transform", None)
        eval_transform = dataset_kwargs.pop("eval_transform", None)

        ## Apply the transforms to the datasets
        train_dataset = dataset_class(
            root=root,
            train=True,
            download=download,
            **dataset_kwargs,
        )

        test_dataset = dataset_class(
            root=root,
            train=False,
            download=download,
            **dataset_kwargs,
        )
        
        ## Split the train dataset into train and validation subsets
        splitter = SplitterRegistry.get_splitter(split_strategy)
        validation_ratio = (validation_ratio, 0.0)
        
        train_subset, validation_subset = splitter.split(
            train_dataset,
            ratios=validation_ratio
        )

        if train_transform is not None:
            train_subset.dataset.transform = train_transform
        if eval_transform is not None:
            validation_subset.dataset.transform = eval_transform
            test_dataset.transform = eval_transform
                        
        return (
            DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            ),
            DataLoader(
                validation_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        )
    
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