from torch import nn
from .encoder import BaseEncoder
from .decoder import BaseDecoder

class UNet(nn.Module):

    def __init__(
        self,
        encoder : BaseEncoder,
        decoder : BaseDecoder,
        num_classes : int,
        use_skip_connections : bool =True
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.use_skip_connections = use_skip_connections

        self.segmentation_head = nn.Conv2d(
            decoder.out_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        input_size = x.shape[-2:]

        features = self.encoder(x)
                
        decoded = self.decoder(
            features,
            skip_connections=self.use_skip_connections
        )

        output = self.segmentation_head(decoded)

        return output