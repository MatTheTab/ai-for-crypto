import numpy as np
from utils.SBox import SBox


def initialize_random(
    num_individuals: int,
    *,
    eager_metric_calculations: bool = False,
    metric_functions=()
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
