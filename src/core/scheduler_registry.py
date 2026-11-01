from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR


class SchedulerRegistry:
    """
    A registry for learning rate schedulers.
    """
    @staticmethod
    def get_scheduler(scheduler_strategy: str, optimizer, **kwargs):
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
        if scheduler_strategy == "step":
            return StepLR(optimizer, **kwargs)
        elif scheduler_strategy == "multistep":
            return MultiStepLR(optimizer, **kwargs)
        elif scheduler_strategy == "exponential":
            return ExponentialLR(optimizer, **kwargs)
        elif scheduler_strategy == "cosine":
            return CosineAnnealingLR(optimizer, **kwargs)
        elif scheduler_strategy == "reduce_on_plateau":
            return ReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler_strategy == "one_cycle":
            return OneCycleLR(optimizer, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler strategy: {scheduler_strategy}")