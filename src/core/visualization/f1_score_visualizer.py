from .base_visualizer import BaseVisualizer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class F1ScoreVisualizer(BaseVisualizer):

    def __init__(self):
        self.confusion_matrix = np.zeros((2, 2), dtype=int)
        self.fig = None
        self.ax = None

    def update(self, trainer, metrics):

        tp = metrics.get("tp", 0)
        fp = metrics.get("fp", 0)
        tn = metrics.get("tn", 0)
        fn = metrics.get("fn", 0)

        self.confusion_matrix = np.array([
            [tn, fp],
            [fn, tp]
        ])

    def render(self):

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 5))

        self.ax.clear()

        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=self.ax
        )

        self.ax.set_title("Confusion Matrix")
        self.ax.set_xlabel("Predicted Label")
        self.ax.set_ylabel("True Label")

        self.fig.tight_layout()
        plt.show(block=False)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, path):

        if self.fig is not None:
            self.fig.savefig(path, bbox_inches="tight")

    def clear(self):

        self.confusion_matrix = np.zeros((2, 2), dtype=int)

        if self.ax is not None:
            self.ax.clear()