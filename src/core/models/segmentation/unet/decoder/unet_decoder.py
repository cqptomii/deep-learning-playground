from torch import nn
from core.models.core.blocks.decoder import DecoderBlock
from ....core.decoder.base_decoder import BaseDecoder



class UNetDecoder(BaseDecoder):
    
    def __init__(self):
        super().__init__()
        
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(512, 256, 256),
            DecoderBlock(256, 128, 128),
            DecoderBlock(128, 64, 64),
        ])
        
        self.conv_blocks = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        ])
        
    
    def forward(self, features, skip_connections=True):
        """Forward method to get the output of the decoder

        Args:
            features (list[Tensor]): List of feature maps at each level of the encoder. 
            skip_connections (bool): Whether to use skip connections or not.

        Returns:
            Tensor : Output tensor from the decoder
        """
        
        x = features[-1]
        skips = features[:-1][::-1]
        
        for decoder_block, conv, skip in zip(self.decoder_blocks, self.conv_blocks, skips):
            x = decoder_block(x, skip if skip_connections else None)
            x = conv(x)