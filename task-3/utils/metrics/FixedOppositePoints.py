from utils.Metric import Metric
import numpy as np


class FixedOppositePoints(Metric):
    def get(self, values: np.ndarray) -> float:
        sbox = values.flatten()
        indices = np.arange(len(sbox))
        is_fixed = sbox == indices
        fixed_count = np.sum(is_fixed)
        opposite_target = indices ^ 0xFF
        is_opposite = sbox == opposite_target
        opposite_count = np.sum(is_opposite)
        total_count = fixed_count + opposite_count
        return float(total_count)
