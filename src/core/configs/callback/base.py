from dataclasses import dataclass


@dataclass
class CallbackConfig:
    """Common configuration shared by callbacks."""

    name: str
