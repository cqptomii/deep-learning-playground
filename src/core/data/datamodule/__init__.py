from .datamodule import DataModule
from .image_datamodule import ImageDataModule

def get_data_module(
    dataset_config: dict,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataModule:
    dataset_type = dataset_config.get("type")
    splitting = dataset_config.get("splitting", {})

    if dataset_type == "image":
        return ImageDataModule(
            root=dataset_config["data_path"],
            batch_size=batch_size,
            num_workers=num_workers,
            train_transform=dataset_config.get("train_transform"),
            eval_transform=dataset_config.get("eval_transform"),
            split_strategy=splitting.get("strategy", "random"),
            split_ratio=tuple(splitting.get("ratio", [0.70, 0.15, 0.15])),
        )
    

    raise ValueError(f"Unsupported dataset type: {dataset_type}")

__version__ = "1.0.0"

__all__ = [
    "DataModule", 
    "ImageDataModule"
]