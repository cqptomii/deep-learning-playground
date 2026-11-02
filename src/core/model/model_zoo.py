from torchvision import models


class ModelZoo:
    """A class that provides access to pre-trained models from torchvision."""

    @staticmethod
    def get_model(model_name: str, pretrained: bool = True, **kwargs):
        """
        Get a pre-trained model from torchvision.

        Args:
            model_name (str): The name of the model to retrieve.
            pretrained (bool): Whether to load the model with pre-trained weights.
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            The requested pre-trained model.
        """
        return models.__dict__[model_name](pretrained=pretrained, **kwargs)