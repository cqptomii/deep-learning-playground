from splitter import RandomSplitter, StratifiedSplitter, BaseSplitter, StratifiedKFoldSplitter

class SplitterRegistry:
    """Registry class to manage different data splitting strategies."""
    
    @staticmethod
    def get_splitter(splitter_strategy : str, random_seed: int = None, **kwargs) -> BaseSplitter:
        """Static method to get the splitter given by the strategy

        Args:
            splitter_strategy (str): Splitter strategy to use. 
                Currently, can be one of "stratified", "random", "kfold", or "group".
            random_seed (int, optional): Random seed to use for reproducibility.
            **kwargs: Additional keyword arguments to pass to the splitter.

        Raises:
            ValueError: If the provided splitter is not supported.

        Returns:
            BaseSplitter: Splitter Object corresponding to the provided strategy.
        """
        if splitter_strategy == "stratified":
            return StratifiedSplitter(random_seed=random_seed)
        elif splitter_strategy == "random":
            return RandomSplitter(random_seed=random_seed)
        elif splitter_strategy == "kfold":
            return StratifiedKFoldSplitter(random_seed=random_seed, **kwargs)
        elif splitter_strategy == "group":
            raise NotImplementedError("KFoldSplitter is not implemented yet.")
        else:
            raise ValueError(f"Unsupported splitter strategy: {splitter_strategy}")        