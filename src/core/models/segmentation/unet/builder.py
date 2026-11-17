from ...builder.base_builder import BaseBuilder
from ...builder.encoder_builder import EncoderBuilder
from ...builder.encoder_builder import DecoderBuilder
from .unet import UNet

class UNetBuilder(BaseBuilder):

    def build(self, config: dict):

        encoder = EncoderBuilder.build(
            config["encoder"]
        )

        decoder = DecoderBuilder.build(
            config["decoder"],
            encoder=encoder
        )

        return UNet(
            encoder=encoder,
            decoder=decoder,
            num_classes=config["num_classes"],
            use_skip_connections=config.get(
                "use_skip_connections", True
            )
        )