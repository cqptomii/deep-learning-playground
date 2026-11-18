from dataclasses import dataclass, field
from typing import Any

@dataclass
class MetricConfig:
    """Class to hold the configuration for a metric."""
    type: str
    name: str
    config: dict[str, Any] = field(default_factory=dict)