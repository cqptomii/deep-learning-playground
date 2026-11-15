from .dataset_zoo import DatasetZoo
from .vision import *
from .datamodule import get_data_module

def get_dataset(dataset_config: dict):
    """Function which Try to load the dataset using built-in datasets first, 
        if not found,
        then try to load it from the provided data_path

    Args:
        dataset_config (dict): Configuration for the dataset to be loaded.
    """
    dataset_name = dataset_config.pop("name")
    if dataset_name is None:
        raise ValueError("dataset_name must be provided in the dataset configuration.")

    dataset_path = dataset_config.pop("data_path", None)
    if dataset_path is None:
        raise ValueError("data_path must be provided in the dataset configuration.")

    dataset_type = dataset_config.pop("type", None)
    if dataset_type is None:
        raise ValueError("type must be provided in the dataset configuration.")

    if dataset_type.startswith("torch_"):
        batch_size = dataset_config.pop("batch_size", 32)
        num_workers = dataset_config.pop("num_workers", 0)
        download = dataset_config.pop("download", True)
        train_transform = dataset_config.pop("train_transform", None)
        eval_transform = dataset_config.pop("eval_transform", None)
        
        return DatasetZoo.get_dataloaders(
            dataset_name=dataset_name,
            library=dataset_type.replace("torch_", ""),
            root=dataset_path,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            train_transform=train_transform,
            eval_transform=eval_transform,
            **dataset_config
        )

    if dataset_type == "image":
        data_module = get_data_module(
            dataset_config=dataset_config,
            batch_size=dataset_config.get("batch_size", 32),
            num_workers=dataset_config.get("num_workers", 0),
        )

        data_module.setup()
        data_module.next_fold()

        return (
            data_module.train_dataloader(),
            data_module.val_dataloader(),
            data_module.test_dataloader(),
        )

    raise ValueError(f"Unsupported dataset type: {dataset_type}")


__version__ = "1.0.0"

__all__ = [
    "DatasetZoo",
    "get_dataset",
    "ApplyTransform",
    "TransformsWrapper",
]
