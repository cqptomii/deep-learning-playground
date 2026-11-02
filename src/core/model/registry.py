from typing import Callable
from torch import nn


class ModelRegistry:

    def __init__(self):
        self._models: dict[str, Callable] = {}

    def register(self, name: str):
        def decorator(factory):
            if name in self._models:
                raise ValueError(
                    f"Model '{name}' already registered"
                )

            self._models[name] = factory
            return factory

        return decorator

    def build(self, name: str, **kwargs) -> nn.Module:
        if name not in self._models:
            raise ValueError(
                f"Unknown model: {name}"
            )

        model = self._models[name](**kwargs)

        if not isinstance(model, nn.Module):
            raise TypeError("Factory must return nn.Module")

        return model

    def list_models(self):
        return list(self._models.keys())
    

registry = ModelRegistry()