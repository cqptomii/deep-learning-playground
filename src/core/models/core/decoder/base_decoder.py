from abc import ABC, abstractmethod
from torch import nn

class BaseDecoder(nn.Module, ABC):

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Abstract Method which give the number of output channels

        Returns:
            int: number of output channels
        """
        pass

    @abstractmethod
    def forward(self, features, skip_connections=True):
        """ Abstract Method which make the forward pass onto the decoder.
        Args:
            features: Feature maps produced by the encoder.
            skip_connections: Whether to use skip connections.

        Returns:
            Decoded feature map.
        """
        pass