from torch.optim import SGD, Adam, AdamW, RMSprop, Adagrad, LBFGS


class OptimizerRegistry:
    
    @staticmethod
    def get_optimizer(optimizer_strategy: str, model_parameters, **kwargs):
        """Static method to get the optimizer given by the strategy

        Args:
            optimizer_strategy (str): Optimizer strategy to use.
            model_parameters: The parameters of the model to optimize.
            **kwargs: Additional keyword arguments to pass to the optimizer.

        Raises:
            ValueError: If the provided optimizer is not supported.

        Returns:
            torch.optim.Optimizer: The initialized optimizer.
        """
        if optimizer_strategy == "sgd":
            return SGD(model_parameters, **kwargs)
        elif optimizer_strategy == "adam":
            return Adam(model_parameters, **kwargs)
        elif optimizer_strategy == "adamw":
            return AdamW(model_parameters, **kwargs)
        elif optimizer_strategy == "rmsprop":
            return RMSprop(model_parameters, **kwargs)
        elif optimizer_strategy == "adagrad":
            return Adagrad(model_parameters, **kwargs)
        elif optimizer_strategy == "lbfgs":
            return LBFGS(model_parameters, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer strategy: {optimizer_strategy}")