from torch import nn
from .conv import ConvBlock2D
from .conv_transpose import ConvTransposeBlock2D
from ..components.skip_connection import SkipConnection

class DecoderBlock(nn.Module):
    
    
    def __init__(self,
            in_channels : int,
            skip_channels : int,
            out_channels : int,
            kernel_size : int = 3,
            stride : int = 1,
            padding : int = 1,
            activation : nn.Module = nn.ReLU(inplace=True),
            upsampling : str = "conv_transpose"
    ):  
        super().__init__()
        
        if upsampling == "conv_transpose":
            self.upsample = ConvTransposeBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation
            )
            
        else:
            raise ValueError(f"Upsampling method {upsampling} not supported. Please choose 'conv_transpose'.")
        
        
        self.skip = SkipConnection(mode="concat")
        
    
    def forward(self, x, skip_features):
        """Forward method to get the output of the decoder block

        Args:
            x (Tensor): Input tensor to the decoder block
            skip_features (Tensor): Skip connection features from the encoder

        Returns:
            Tensor : Output tensor from the decoder block
        """
        
        x = self.upsample(x)
        x = self.skip(x, skip_features)
        
        return x
        
        
        
        
        
        
        
        
        
    
        
        