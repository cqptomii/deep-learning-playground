from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from .base_encoder import BaseEncoder


class PretrainedEncoder(BaseEncoder):

    def __init__(
        self,
        backbone: nn.Module,
        return_nodes: dict[str, str],
        feature_channels: list[int],
        feature_strides: list[int],
    ):
        super().__init__()

        self.backbone = create_feature_extractor(
            backbone,
            return_nodes=return_nodes
        )

        self._feature_channels = feature_channels
        self._feature_strides = feature_strides

    @property
    def feature_channels(self):
        return self._feature_channels

    @property
    def feature_strides(self):
        return self._feature_strides

    def forward(self, x):
        features = self.backbone(x)

        return list(features.values())