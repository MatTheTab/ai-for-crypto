from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    """
    An abstract class for SBox metric
    """

    @abstractmethod
    def get(self, values: np.ndarray) -> float:
        raise NotImplementedError("Must be implemented")
