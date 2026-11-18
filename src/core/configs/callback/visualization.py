from dataclasses import dataclass, field
from typing import Any

from .base import CallbackConfig


@dataclass
class VisualizationCallbackConfig(CallbackConfig):
    name: str = "visualization_callback"
    visualizers: list[Any] = field(default_factory=list)
    frequency: int = 1
    save_path: str | None = None
