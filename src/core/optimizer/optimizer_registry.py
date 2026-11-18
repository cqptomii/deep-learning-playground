from dataclasses import fields

from ..configs import OptimizerConfig

class OptimizerRegistry:
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, optimizer_class, config_class : OptimizerConfig):
        """Class method to register an optimizer class with a given name.

        Args:
            name (str): The name of the optimizer.
            optimizer_class: The optimizer class to register.
        """
        cls._registry[name.lower()] = {
            "optimizer": optimizer_class,
            "config": config_class
        }
    
    
    @classmethod
    def get(cls, config : dict, model_parameters):
        """Class method to build an optimizer class given its name.

        Args:
            parameters (dict): The config dictionary containing the optimizer name and its parameters.

        Returns:
            The optimizer class associated with the given name.
        """
        config = config.copy()
        name = config.pop("name", None)

        if name is None:
            raise ValueError("Missing optimizer name.")

        entry = cls._registry.get(name.lower())

        if entry is None:
            raise ValueError(f"Unknown optimizer: {name}")

        # Convert YAML dictionary into the corresponding dataclass
        optimizer_config = entry["config"](**config)

        # Extract optimizer parameters
        parameters = {
            field.name: getattr(optimizer_config, field.name)
            for field in fields(optimizer_config)
        }
    
        # Instantiate PyTorch optimizer
        return entry["optimizer"](model_parameters, **parameters)