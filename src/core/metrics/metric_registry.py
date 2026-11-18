from dataclasses import fields

from ..configs import MetricConfig


class MetricRegistry:

    _registry = {}

    @classmethod
    def register(
        cls,
        metric_type: str,
        name: str,
        metric_class,
        config_class: MetricConfig,
    ):
        """Register a metric class with its corresponding config dataclass."""
        metric_type = metric_type.lower()
        cls._registry.setdefault(metric_type, {})[name.lower()] = {
            "metric": metric_class,
            "config": config_class,
        }

    @classmethod
    def get(cls, config: dict):
        """Build a metric from a configuration dictionary."""
        config = config.copy()
        metric_type = config.pop("type", None)
        name = config.pop("name", None)

        if metric_type is None:
            raise ValueError("Missing metric type.")
        if name is None:
            raise ValueError("Missing metric name.")

        entry = cls._registry.get(metric_type.lower(), {}).get(name.lower())
        if entry is None:
            raise ValueError(f"Unknown metric: {metric_type}/{name}")

        metric_config = entry["config"](
            type=metric_type,
            name=name,
            **config,
        )
        parameters = {
            field.name: getattr(metric_config, field.name)
            for field in fields(metric_config)
            if field.name not in {"type", "name", "config"}
        }

        parameters.update(metric_config.config)
        return entry["metric"](**parameters)
