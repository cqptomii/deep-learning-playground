from torch.optim import SGD, Adam, AdamW, RMSprop, Adagrad, LBFGS


class OptimizerRegistry:
    
    @staticmethod
    def get_optimizer(name: str, model_parameters, **kwargs):
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
        if name == "sgd":
            return SGD(model_parameters, **kwargs)
        elif name == "adam":
            return Adam(model_parameters, **kwargs)
        elif name == "adamw":
            return AdamW(model_parameters, **kwargs)
        elif name == "rmsprop":
            return RMSprop(model_parameters, **kwargs)
        elif name == "adagrad":
            return Adagrad(model_parameters, **kwargs)
        elif name == "lbfgs":
            return LBFGS(model_parameters, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer strategy: {name}")