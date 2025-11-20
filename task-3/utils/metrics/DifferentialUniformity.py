from utils.Metric import Metric
import numpy as np


class DifferentialUniformity(Metric):
    def get(self, values: np.ndarray) -> float:
        sbox = values.flatten()
        max_count = 0
        x_values = np.arange(
            len(sbox),
            dtype=sbox.dtype if sbox.dtype in (np.uint8, np.int16) else np.uint8,
        )
        for delta_x in range(1, len(sbox)):
            x_shifted = x_values ^ delta_x
            S_x = sbox[x_values]
            S_x_shifted = sbox[x_shifted]
            delta_y_array = S_x ^ S_x_shifted
            counts = np.bincount(delta_y_array, minlength=len(sbox))
            current_max_count = np.max(counts)
            if current_max_count > max_count:
                max_count = current_max_count
        return float(max_count)
