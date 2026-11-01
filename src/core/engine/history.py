import pandas as pd


class History:

    def __init__(self):
        self.records = []

    def log(self, epoch: int, metrics: dict):
        record = {"epoch": epoch, **metrics}
        self.records.append(record)

    def to_dataframe(self):
        return pd.DataFrame(self.records)

    def save(self, path: str):
        self.to_dataframe().to_csv(path, index=False)