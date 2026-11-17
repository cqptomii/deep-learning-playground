from .base_encoder import BaseEncoder


class CustomEncoder(BaseEncoder):

    def __init__(self, feature_channels: list[int], feature_strides: list[int]):
        super().__init__()

        self._feature_channels = feature_channels
        self._feature_strides = feature_strides

    @property
    def feature_channels(self):
        return self._feature_channels

    @property
    def feature_strides(self):
        return self._feature_strides