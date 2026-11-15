from abc import ABC, abstractmethod
from torch import nn


class BaseEncoder(nn.Module, ABC):

    @property
    @abstractmethod
    def feature_channels(self) -> list[int]:
        """Abstract Method which return the number of channels produced by the encoder at each feature level.

        Returns:
            list[int]: list of number of channels produced by the encoder at each feature level.
        """
        pass

    @property
    @abstractmethod
    def feature_strides(self) -> list[int]:
        """
        Abstract Method which return the spatial downsampling factor at each feature level.
        Returns:
            list[int]: list of spatial downsampling factor at each feature level.
        """
        pass

    @abstractmethod
    def forward(self, x):
        """Abstract Method which make the forward pass onto the encoder 

        Args:
            x (list[Tensor]): List of feature maps at each level of the encoder. 
        """
        pass