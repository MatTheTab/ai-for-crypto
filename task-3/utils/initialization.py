import numpy as np
from utils.SBox import SBox

def initialize_random(
    num_individuals: int, eager_metric_calculations: bool = False, metric_functions=None
):
    sboxes = []

    for _ in range(num_individuals):
        values = np.random.permutation(256).astype(np.uint8)
        sbox = SBox(
            values=values,
            eager_metric_calculations=eager_metric_calculations,
            metric_functions=metric_functions,
        )
        sboxes.append(sbox)

    return sboxes

def initialize_fy_random(
    num_individuals: int, eager_metric_calculations: bool = False, metric_functions=None
):
    """
    Fisher-Yates shuffle-based initialization
    """
    def yf_shuffle(arr, n=256):
        for i in range(n-1,0,-1):
            j = np.random.randint(0,i+1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    
    sboxes = []
    values = np.random.permutation(256).astype(np.uint8)
    for _ in range(num_individuals):
        values = yf_shuffle(values)
        sbox = SBox(
            values=values.copy(),
            eager_metric_calculations=eager_metric_calculations,
            metric_functions=metric_functions,
        )
        sboxes.append(sbox)

    return sboxes