from .base_visualizer import BaseVisualizer
import matplotlib.pyplot as plt
import numpy as np


class ConfusionMatrixVisualizer(BaseVisualizer):

    def __init__(self, classes: list = None):
        self.confusion_matrix = None
        self.classes = classes

        self._plot = None
        self._ax = None
        self._image = None

    def update(self, trainer, metrics: dict):
        # Update the confusion matrix based on the provided metrics
        self.confusion_matrix = metrics.get('confusion_matrix', None)

    def render(self):

        if self.confusion_matrix is None:
            raise ValueError("No confusion matrix to render. Please update the visualizer with metrics first.")

        matrix = np.asarray(self.confusion_matrix)

        ## Render the confusion matrix using matplotlib
        if self._plot is None:
            self._plot, self._ax = plt.subplots(figsize=(7, 6))

            self._image = self._ax.imshow(
                matrix,
                interpolation='nearest',
                cmap=plt.cm.Blues
            )

            self._plot.colorbar(self._image, ax=self._ax)

        else:
            self._image.set_data(matrix)
            self._image.set_clim(vmin=0, vmax=max(matrix.max(), 1))

        self._ax.set_title('Confusion Matrix')
        self._ax.set_xlabel('Predicted label')
        self._ax.set_ylabel('True label')

        if self.classes:
            self._ax.set_xticks(range(len(self.classes)))
            self._ax.set_yticks(range(len(self.classes)))

            self._ax.set_xticklabels(self.classes, rotation=45)
            self._ax.set_yticklabels(self.classes)

        # Display values inside the matrix
        for text in self._ax.texts:
            text.remove()

        threshold = matrix.max() / 2

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self._ax.text(
                    j, i, format(matrix[i, j], 'd'),
                    ha='center',
                    va='center',
                    color='white' if matrix[i, j] > threshold else 'black'
                )

        ## Show the confusion matrix plot
        self._plot.tight_layout()
        plt.show(block=False)
        self._plot.canvas.draw_idle()
        self._plot.canvas.flush_events()

    ## Save the plot
    def save(self, path):

        if self._plot is None:
            raise ValueError("No plot to save. Please render the visualizer first.")

        self._plot.savefig(path, bbox_inches='tight')

    def clear(self):
        self.confusion_matrix = None

        if self._plot is not None:
            plt.close(self._plot)

        self._plot = None
        self._ax = None
        self._image = None