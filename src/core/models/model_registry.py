from typing import Callable, Tuple
from .builder.base_builder import BaseBuilder
from torch import nn


MODEL_REGISTRY: dict[str, Tuple[Callable, dict]] = {
    "resnet18": (None, {})
}
class ModelRegistry:
    """ Registry which contained differents DL models organized with the following structure
    
       - model_type: (e.g. segmentation, classification, detection)
            - model_name: (e.g. unet, resnet, yolov5)

    Raises:
        ValueError: If the model architecture is not registered in the registry.

    """

    ## The registry dictionary to store model builders
    _builders = {}

    @classmethod
    def register(cls, type : str,name: str, builder : BaseBuilder, overwrite: bool = False):
        """Method to register a new model with its corresponding builder in the registry.

        Args:
            type (str): The type of the model architecture (e.g., segmentation, classification, detection).
            name (str): The name of the model to register.
            builder (BaseBuilder): The builder class responsible for constructing the model.
            overwrite (bool, optional): Specify wether to overwrite the model if it is an existing model in the registry. Defaults to False.
        """

        ## Check if the model type already exists in the registry
        if type not in cls._builders:
            cls._builders[type] = {}
        
        if not overwrite and name in cls._builders[type]:
            print(f"Model architecture '{name}' already exists in the registry for type '{type}'.")
            return

        cls._builders[type][name] = builder

    @classmethod
    def build(cls, type : str, name: str, config: dict):
        """Method which build the right model with the configuration dict given

        Args:
            type (str): The type of the model architecture (e.g., segmentation, classification, detection).
            name (str): The name of the model to register.
            config (dict): The configuration dictionary for the model.

        Raises:
            ValueError: If the model type or architecture is not registered in the registry.

        Returns:
            _type_: The built model instance.
        """

        if type not in cls._builders:
            raise ValueError(f"Unknown model type: {type}")

        if name not in cls._builders[type]:
            raise ValueError(f"Unknown model architecture: {name}")

        builder = cls._builders[type][name]()

        return builder.build(config)