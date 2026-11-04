from .base_visualizer import BaseVisualizer
import matplotlib.pyplot as plt
from typing import List, Optional


class TrainingCurveVisualizer(BaseVisualizer):

    def __init__(
        self,
        metrics_to_plot: Optional[List[str]] = None,
        plot_title: str = "Training Curves",
        plot_xlabel: str = "Epoch",
        plot_ylabel: str = "Value"
    ):
        ## Metric history to store the metrics over epochs
        self.metrics_history = []

        ## Hyperparameters for the plot
        self.metrics_to_plot = metrics_to_plot or [
            'train_loss',
            'val_loss',
            'train_accuracy',
            'val_accuracy'
        ]

        self.plot_title = plot_title
        self.plot_xlabel = plot_xlabel
        self.plot_ylabel = plot_ylabel

        ## Variable to store the plot figure
        self._plot = None
        self._ax = None

    def update(self, trainer, metrics: dict):
        self.metrics_history.append(metrics.copy())

    def render(self):

        ## Render the training curve using matplotlib
        if not self.metrics_history:
            raise ValueError(
                "No metrics to render. Please update the visualizer with metrics first."
            )

        ## Create or clear the plot figure
        if self._plot is None:
            self._plot, self._ax = plt.subplots(figsize=(10, 6))
        else:
            self._ax.clear()

        epochs = list(range(1, len(self.metrics_history) + 1))

        ## Extract metrics for plotting from the history
        metrics_to_plot = {
            metric: [] for metric in self.metrics_to_plot
        }

        ## Populate the metrics_to_plot dictionary with values from metrics_history
        for metrics in self.metrics_history:
            for metric in self.metrics_to_plot:
                metrics_to_plot[metric].append(metrics.get(metric, None))

        ## Plot the training curves
        plotted_metrics = False

        for metric, values in metrics_to_plot.items():

            if any(v is not None for v in values):
                self._ax.plot(
                    epochs,
                    values,
                    label=metric
                )
                plotted_metrics = True

        self._ax.set_xlabel(self.plot_xlabel)
        self._ax.set_ylabel(self.plot_ylabel)
        self._ax.set_title(self.plot_title)
        self._ax.grid(True, alpha=0.3)

        if plotted_metrics:
            self._ax.legend()

        ## Show the training curves plot
        self._plot.tight_layout()
        plt.show(block=False)
        self._plot.canvas.draw_idle()
        self._plot.canvas.flush_events()

    def save(self, path: str):

        ## Save the rendered plot to the specified path
        if self._plot is None:
            raise ValueError(
                "No plot to save. Please render the visualizer first."
            )

        self._plot.savefig(path, bbox_inches='tight')

    def clear(self):

        self.metrics_history.clear()

        if self._plot is not None:
            plt.close(self._plot)

        self._plot = None
        self._ax = None