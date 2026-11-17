from .base_builder import BaseBuilder
from ..model_zoo import ModelZoo
from ..core.encoder import PretrainedEncoder
from ..core.encoder import CustomEncoder


class EncoderBuilder(BaseBuilder):
    
    @staticmethod
    def build(config: dict):

        encoder_type = config["type"]

        if encoder_type == "pretrained":

            backbone = ModelZoo.get_model(
                model_name=config["model"],
                use_pretrained=config.get("pretrained", True),
                weights=config.get("weights")
            )

            return PretrainedEncoder(
                backbone=backbone,
                **get_encoder_metadata(config["name"])
            )

        elif encoder_type == "custom":
            return CustomEncoder(**config["params"])

        raise ValueError(
            f"Unknown encoder type: {encoder_type}"
        )