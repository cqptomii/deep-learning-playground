import pandas as pd


class History:

    def __init__(self):
        self.batch_records = []
        self.epoch_records = []

    def log_batch(
        self,
        epoch: int,
        step: int,
        stage: str,
        metrics: dict,
        **kwargs
    ):
        self.batch_records.append({
            "epoch": epoch,
            "step": step,
            "stage": stage,
            **metrics,
            **kwargs
        })

    def log_epoch(
        self,
        epoch: int,
        stage: str,
        metrics: dict,
        **kwargs
    ):
        self.epoch_records.append({
            "epoch": epoch,
            "stage": stage,
            **metrics,
            **kwargs
        })

    def batch_dataframe(self):
        return pd.DataFrame(self.batch_records)

    def epoch_dataframe(self):
        return pd.DataFrame(self.epoch_records)