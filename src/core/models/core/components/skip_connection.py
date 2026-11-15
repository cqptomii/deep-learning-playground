import torch
from torch import nn


class SkipConnection(nn.Module):
    def __init__(self, mode="concat"):
        super().__init__()
        self.mode = mode

    def forward(self, decoder_features, encoder_features):
        """Method 

        Args:
            decoder_features (_type_): _description_
            encoder_features (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        ## Check if the encoder features are None, if so return the decoder features
        if encoder_features is None:
            return decoder_features
        
        ## Check if the two features maps have the same dimensions 
        if decoder_features.shape[-2:] != encoder_features.shape[-2:]:
            raise ValueError(f"Decoder features and encoder features must have the same spatial dimensions. Got {decoder_features.shape[-2:]} and {encoder_features.shape[-2:]}.")
        
        ## Concat the encoder features with the decoder features along the channel dimension
        if self.mode == "concat":
            return torch.cat(
                [decoder_features, encoder_features],
                dim=1
            )

        elif self.mode == "add":
            return decoder_features + encoder_features

        raise ValueError(f"Unknown skip mode: {self.mode}")