from torch import nn


class ConvTransposeBlock2D(nn.Module):
    """A transposed convolutional block that consists of a transposed convolutional layer, followed by batch normalization and an activation function."""
    
    def __init__(self, 
                in_channels : int,
                out_channels : int,
                kernel_size : int = 3,
                stride : int = 1,
                padding : int = 1,
                activation : nn.Module = nn.ReLU(inplace=True),
                use_batchnorm : bool = True
    ):
        super().__init__()
        
        
        block_list = []
        
        
        for i in range(2):
            conv_layer = nn.ConvTranspose2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size,
                stride,
                padding
            )
            block_list.append(conv_layer)
            
            if use_batchnorm:
                block_list.append(nn.BatchNorm2d(out_channels))
            
            block_list.append(activation)
        
        self.block = nn.Sequential(*block_list)
        
    def forward(self, x):
        """Forward method to get the output of the transposed convolutional block

        Args:
            x (Tensor): Input tensor to the transposed convolutional block

        Returns:
            Tensor : Output tensor from the transposed convolutional block
        """
        return self.block(x)