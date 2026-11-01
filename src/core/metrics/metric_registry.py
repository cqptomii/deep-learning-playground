from metrics import BaseMetric

def MetricRegistry():
    """
        Class to register and manage metrics for evaluation during training and validation.
    """
    
    def __init__(self):
        self.train_metrics = {}
        self.val_metrics = {}
        
    
    def register_metric(self,
        metric_name : str,
        metric_class : BaseMetric,
        metric_type : str = "train"
    ):
        """
            Method to register a metric function for either training or validation.

            Args:
                metric_name (str): Name of the metric to be registered.
                metric_class (BaseMetric): Class of the metric to be registered.
                metric_type (str): Type of the metric, either "train" or "val".
        """
        if metric_type == "train":
            self.train_metrics[metric_name] = metric_class()
        elif metric_type == "val":
            self.val_metrics[metric_name] = metric_class()
        else:
            raise ValueError("metric_type must be either 'train' or 'val'")
    
    def get_metrics(self, metric_type : str = "train"):
        """
            Method to retrieve the registered metrics for either training or validation.

            Args:
                metric_type (str): Type of the metrics to retrieve, either "train" or "val".

            Returns:
                dict: Dictionary of registered metrics.
        """
        if metric_type == "train":
            return self.train_metrics
        elif metric_type == "val":
            return self.val_metrics
        else:
            raise ValueError("metric_type must be either 'train' or 'val'")