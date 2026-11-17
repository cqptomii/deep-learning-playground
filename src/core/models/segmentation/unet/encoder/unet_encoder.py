from torch import nn
from ....core.encoder.base_encoder import BaseEncoder



class UNetEncoder(BaseEncoder):
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ),
        ])
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Forward method to get the output of the encoder

        Args:
            x (Tensor): Input tensor to the encoder

        Returns:
            list[Tensor] : List of feature maps at each level of the encoder
        """
        features = []
        
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        
        return features
        
        