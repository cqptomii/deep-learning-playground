from dataclasses import dataclass

from .base import CallbackConfig


@dataclass
class ProgressCallbackConfig(CallbackConfig):
    name: str = "progress_callback"
