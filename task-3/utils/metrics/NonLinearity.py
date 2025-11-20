from utils.Metric import Metric
import numpy as np


class NonLinearity(Metric):
    def get(self, values: np.ndarray) -> float:
        return calculate_nonlinearity(values)


def fwht(a):
    a = np.array(a, dtype=int)
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a


def get_component_function(sbox, mask):
    n = int(np.log2(len(sbox)))
    truth_table = []
    for x in range(len(sbox)):
        y = sbox[x]
        bit = bin(y & mask).count("1") % 2
        truth_table.append(1 if bit == 0 else -1)
    return truth_table


def calculate_nonlinearity(values):
    if len(values.shape) >= 2:
        sbox = values.flatten()
    else:
        sbox = values
    n = int(np.log2(len(sbox)))
    size = len(sbox)
    min_nl = size  # Start high
    for mask in range(1, size):
        f = get_component_function(sbox, mask)
        spectrum = fwht(f)
        max_abs_wht = np.max(np.abs(spectrum))
        nl = (2 ** (n - 1)) - (max_abs_wht / 2)
        if nl < min_nl:
            min_nl = nl
    return int(min_nl)
