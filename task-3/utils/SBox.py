import numpy as np
from utils.Metric import Metric


class SBox:
    """
    S-Box representation class (Possible over-engineering but looks cool at least)
    """

    def __init__(
        self, values: np.ndarray, eager_metric_calculations: bool, metric_functions
    ):
        self.values = values
        self.eager_metric_calculations = eager_metric_calculations
        self.metric_functions = metric_functions
        for metric_name, metric_func_class in self.metric_functions:
            if not isinstance(metric_func_class, Metric):
                raise TypeError("Passed metrc is not an isntance of the Metric class")

        if self.eager_metric_calculations:
            self.metrics = self.calculate_metrics()
        else:
            self.metrics = None

        if not (
            (self.values.size == 256)
            and np.array_equal(np.sort(self.values.ravel()), np.arange(256))
        ):
            raise TypeError("Invalid SBox created.")

    def set_values(self, values: np.ndarray):
        self.values = values

    def calculate_metrics(self):
        metric_values = {}
        for metric_name, metric_func_obj in self.metric_functions:
            metric_values[metric_name] = metric_func_obj.get(self.values)
        return metric_values
