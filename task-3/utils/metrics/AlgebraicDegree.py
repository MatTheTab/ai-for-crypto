from utils.Metric import Metric
import numpy as np


class AlgebraicDegree(Metric):
    def _compute_anf_coeffs(self, truth_table: np.ndarray) -> np.ndarray:
        coeffs = truth_table.copy().astype(int)
        N = len(coeffs)
        n = int(np.log2(N))
        for i in range(n):
            step = 1 << i
            for j in range(N):
                if (j & step) == 0:
                    coeffs[j + step] = (coeffs[j + step] + coeffs[j]) % 2
        return coeffs

    def get(self, values: np.ndarray) -> float:
        N, m = 16, 16
        sbox = values.reshape((N, m))
        n = int(np.log2(N))
        if 2**n != N:
            raise ValueError(f"Input size N ({N}) must be a power of 2 (2^n).")
        max_degree = 0
        for i in range(m):
            f_i_truth_table = sbox[:, i]
            coeffs = self._compute_anf_coeffs(f_i_truth_table)
            current_degree = 0
            for k in range(N - 1, -1, -1):
                if coeffs[k] == 1:
                    degree_k = bin(k).count("1")
                    current_degree = degree_k
                    break
            if current_degree > max_degree:
                max_degree = current_degree
        return float(max_degree)
