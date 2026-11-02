
from callback import BaseCallback
from engine import Trainer
from tqdm.auto import tqdm


class ProgressCallback(BaseCallback):

    def __init__(self):
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, trainer : Trainer):
        self.epoch_bar = tqdm(
            total=trainer.epochs,
            desc="Training",
            unit="epoch",
            position=0,
        )

    def on_epoch_begin(self, trainer : Trainer, epoch : int):
        self.batch_bar = tqdm(
            total=len(trainer.train_loader),
            desc=f"Epoch {epoch + 1}/{trainer.epochs}",
            unit="batch",
            position=1,
            leave=False,
        )

    def on_train_batch_end(self, trainer : Trainer, batch_idx : int, metrics : dict):
        self.batch_bar.update(1)

        self.batch_bar.set_postfix(
            train_loss=f"{metrics.get('train_loss', 0):.4f}",
            train_acc=f"{metrics.get('train_accuracy', 0):.4f}"
        )

    def on_validation_end(self, trainer : Trainer, metrics : dict):
        self.epoch_bar.set_postfix(
            val_loss=f"{metrics['val_loss']:.4f}",
            val_acc=f"{metrics.get('val_accuracy', 0):.4f}",
        )

    def on_epoch_end(self, trainer : Trainer, epoch : int, metrics : dict):
        self.epoch_bar.update(1)

        self.epoch_bar.set_postfix(
            train_loss=f"{metrics['train_loss']:.4f}",
            train_acc=f"{metrics['train_accuracy']:.4f}",
            val_loss=f"{metrics['val_loss']:.4f}",
            val_acc=f"{metrics['val_accuracy']:.4f}",
        )

        self.batch_bar.close()
        self.batch_bar = None

    def on_train_end(self, trainer : Trainer):
        if self.batch_bar is not None:
            self.batch_bar.close()

        if self.epoch_bar is not None:
            self.epoch_bar.close()
