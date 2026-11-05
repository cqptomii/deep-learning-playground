from typing import Callable, Tuple
from torch import nn


MODEL_REGISTRY: dict[str, Tuple[Callable, dict]] = {
    "resnet18": (None, {})
}