from utils.Metric import Metric
import numpy as np
from scipy.stats import pearsonr


class BitIndependeceCriterion(Metric):
    def get(self, value: np.ndarray) -> float:
        sbox = value.flatten()
        N = len(sbox)
        if N == 0 or N & (N - 1) != 0:
            raise ValueError("S-box size must be a power of 2.")
        n = int(np.log2(N))
        m = sbox.dtype.itemsize * 8
        if N > 0:
            m = int(np.ceil(np.log2(np.max(sbox) + 1)))
            m = max(m, 1)
        if m == 0:
            return 0.0
        max_abs_correlation = 0.0
        for i in range(n):
            E_i = 1 << i
            diff_vector_Y = sbox ^ sbox[np.bitwise_xor(np.arange(N), E_i)]
            output_flip_sequences = []
            for j in range(m):
                sequence_j = (diff_vector_Y >> j) & 1
                output_flip_sequences.append(sequence_j)
            for j in range(m):
                for k in range(j + 1, m):
                    seq_j = output_flip_sequences[j]
                    seq_k = output_flip_sequences[k]
                    try:
                        correlation, _ = pearsonr(seq_j, seq_k)
                    except ValueError:
                        correlation = 0.0
                    abs_correlation = abs(correlation)
                    if abs_correlation > max_abs_correlation:
                        max_abs_correlation = abs_correlation
        return float(max_abs_correlation)
