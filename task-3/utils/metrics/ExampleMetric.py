from utils.Metric import Metric
import numpy as np


class ExampleMetric(Metric):
    def get(self, values: np.ndarray) -> float:
        return 0.0
