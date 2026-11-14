from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR


class SchedulerRegistry:
    """
    A registry for learning rate schedulers.
    """
    @staticmethod
    def get_scheduler(name: str, optimizer, **kwargs):
        """Static method to get the scheduler given by the strategy

        Args:
            scheduler_strategy (str): Scheduler strategy to use.
            optimizer (torch.optim.Optimizer): The optimizer for which the scheduler is to be applied.
            **kwargs: Additional keyword arguments to pass to the scheduler.
            
        Raises:
            ValueError: If the provided scheduler is not supported.

        Returns:
            torch.optim.lr_scheduler.LRScheduler: The initialized scheduler.
        """
        if name == "step":
            return StepLR(optimizer, **kwargs)
        elif name == "multistep":
            return MultiStepLR(optimizer, **kwargs)
        elif name == "exponential":
            return ExponentialLR(optimizer, **kwargs)
        elif name == "cosine":
            return CosineAnnealingLR(optimizer, **kwargs)
        elif name == "reduce_on_plateau":
            return ReduceLROnPlateau(optimizer, **kwargs)
        elif name == "one_cycle":
            return OneCycleLR(optimizer, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler strategy: {name}")