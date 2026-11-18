from dataclasses import fields
from ..configs import CallbackConfig


class CallbackRegistry:

    _registry = {}

    @classmethod
    def register(cls, name: str, callback_class, config_class: CallbackConfig):
        """Register a callback class with its corresponding config dataclass."""
        cls._registry[name.lower()] = {
            "callback": callback_class,
            "config": config_class,
        }

    @classmethod
    def get(cls, config: dict):
        """Build a callback from a configuration dictionary."""
        config = config.copy()
        name = config.pop("name", None)

        if name is None:
            raise ValueError("Missing callback name.")

        entry = cls._registry.get(name.lower())

        if entry is None:
            raise ValueError(f"Unknown callback: {name}")

        callback_config = entry["config"](**config)

        parameters = {
            field.name: getattr(callback_config, field.name)
            for field in fields(callback_config)
            if field.name != "name"
        }

        return entry["callback"](**parameters)
