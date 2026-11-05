from torchvision import models

class ModelZoo:
    """A class that provides access to pre-trained models from torchvision."""

    @staticmethod
    def get_model(model_name: str, pretrained: bool = True, weights : str = None, **kwargs):
        """
        Load a torchvision model dynamically.

        Args:
            model_name: Name of the torchvision model.
            pretrained: Whether to load pretrained weights.
            weights: Name of the weights enum member (e.g. IMAGENET1K_V1).
            **kwargs: Additional arguments for the model constructor.

        Returns:
            An initialized torchvision model.
        """

        ## Retrieve model constructor
        try:
            model_fn = models.get_model_builder(model_name)
        except Exception:
            raise ValueError(f"Unknown torchvision model: '{model_name}'")
        
        ## Retrieve weights enum
        weights_enum = models.get_model_weights(model_name)

        ## Determine which weights to load
        selected_weights = None

        if pretrained:
            if weights is not None:
                try:
                    selected_weights = weights_enum[weights]
                except KeyError:
                    raise ValueError(
                        f"Invalid weights '{weights}' for model '{model_name}'. "
                        f"Available weights: {list(weights_enum.__members__.keys())}"
                    )
            else:
                selected_weights = weights_enum.DEFAULT

        ## Instantiate model
        model = model_fn(weights=selected_weights, **kwargs)

        return model