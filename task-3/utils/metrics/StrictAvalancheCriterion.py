from utils.Metric import Metric
import numpy as np


class StrictAvalancheCriterion(Metric):
    def get(self, values: np.ndarray) -> float:
        sbox = values.flatten()
        S_box_size = sbox.size
        N = int(np.log2(S_box_size))
        avalanche_matrix = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            e_i = 1 << i
            input_indices = np.arange(S_box_size)
            flipped_indices = input_indices ^ e_i
            output_difference = sbox[input_indices] ^ sbox[flipped_indices]
            for j in range(N):
                e_j = 1 << j
                flip_count = np.count_nonzero((output_difference & e_j) > 0)
                probability = flip_count / S_box_size
                avalanche_matrix[i, j] = probability
        deviation_matrix = np.abs(avalanche_matrix - 0.5)
        max_deviation = np.max(deviation_matrix)
        return float(max_deviation)
