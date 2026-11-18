from dataclasses import dataclass

from .base import CallbackConfig


@dataclass
class ModelCheckpointConfig(CallbackConfig):
    name: str = "model_checkpoint"
    save_dir: str = "./checkpoints"
    model_name: str = "best_model.pt"
    monitor: str = "val_loss"
    mode: str = "min"
