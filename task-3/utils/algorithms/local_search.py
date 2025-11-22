from typing import Iterable
import numpy as np


def node_swap_move(arr: np.ndarray) -> Iterable[np.ndarray]:
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            new_arr = arr.copy()
            new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
            yield new_arr


def edge_swap_move(arr: np.ndarray) -> Iterable[np.ndarray]:
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            new_arr = arr.copy()
            new_arr[i : j + 1] = new_arr[i : j + 1][::-1]
            yield new_arr
