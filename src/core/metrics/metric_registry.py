from .base_metric import BaseMetric

def MetricRegistry():
    """ 
    A registry for metrics. This class allows you to register and retrieve metric classes based on their type and name.
    """

    ## Metric registry
    _registry = {}
    
    @classmethod
    def register(cls, type : str, name : str, class_instance : BaseMetric):
        """ Method which register a new metric in the registry 

        Args:
            type (str): The type of the metric (e.g., classification, segmentation, detection).
            name (str): The name of the metric to register.
            class_instance (BaseMetric): The instance of the metric class to register.
        """
        
        if type not in cls._registry:
            cls._registry[type] = {}
        
        if name in cls._registry[type]:
            print(f"Metric '{name}' already exists in the registry for type '{type}'.")
            return
        
        cls._registry[type][name] = class_instance
        
    @classmethod
    def get(cls, type : str, name : str, config : dict = {}):
        """ Method which return the metric class instance from the registry

        Args:
            type (str): The type of the metric (e.g., classification, segmentation, detection).
            name (str): The name of the metric to retrieve.
            config (dict, optional): The configuration dictionary for the metric. Defaults to {}.
        Raises:
            ValueError: If the metric type or name is not registered in the registry.

        Returns:
            BaseMetric: The metric object.
        """
        
        if type not in cls._registry:
            raise ValueError(f"Unknown metric type: {type}")
        
        if name not in cls._registry[type]:
            raise ValueError(f"Unknown metric name: {name}")
        
        return cls._registry[type][name](**config)
        