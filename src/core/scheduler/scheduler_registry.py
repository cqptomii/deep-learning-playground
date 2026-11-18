from dataclasses import fields
from ..configs.scheduler.base import SchedulerConfig


class SchedulerRegistry:
    
    _registry = {}
        
    @classmethod
    def register(cls, name: str, scheduler_class, config_class : SchedulerConfig):
        """Class method to register an scheduler class with a given name.

        Args:
            name (str): The name of the scheduler.
            scheduler_class: The scheduler class to register.
            config_class: The config class to register.
        """
        cls._registry[name.lower()] = cls._registry[name.lower()] = {
            "scheduler": scheduler_class,
            "config": config_class
        }
    
        
    @classmethod
    def get(cls, config: dict, optimizer):
        """Method which instanciate an scheduler based on it's configuration

        Args:
            config (dict): The configuration dictionary for the scheduler.
            optimizer (_type_): The optimizer to use with the scheduler.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        config = config.copy()
        name = config.pop("name", None)

        if name is None:
            raise ValueError("Missing scheduler name.")

        entry = cls._registry.get(name.lower())

        if entry is None:
            raise ValueError(f"Unknown scheduler: {name}")

        # Convert YAML dictionary into the corresponding dataclass
        scheduler_config = entry["config"](**config)

        # Extract scheduler parameters
        parameters = {
            field.name: getattr(scheduler_config, field.name)
            for field in fields(scheduler_config)
        }

        # Instantiate PyTorch scheduler
        return entry["scheduler"](optimizer, **parameters)
    