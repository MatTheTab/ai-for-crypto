from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Iterable, Iterator
import os
import math
import numpy as np
import sys


def produce_data(n_examples: int, message_size: int) -> List[bytes]:
    """Generates a list of random byte strings."""
    data = []
    for i in range(n_examples):
        message = os.urandom(message_size)
        data.append(message)
    return data


def get_elementary_bit_vector(i: int, n: int) -> bytes:
    """Return a bytes object representing an n-bit vector with bit i=1 (LSB = bit 0)."""
    n_bytes = math.ceil(n / 8)
    vector = bytearray(n_bytes)
    byte_index = i // 8
    bit_index = i % 8  # LSB-first order
    vector[n_bytes - byte_index - 1] = 1 << bit_index  # fill from right
    return bytes(vector)


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """Bitwise XOR of two byte strings of equal length."""
    return bytes(x ^ y for x, y in zip(a, b))


def bytes_to_bitstring(b: bytes) -> str:
    """Convert bytes to bitstring representation."""
    return "".join(f"{byte:08b}" for byte in b)


def results_to_arr(
    results: List[Tuple[str, str, str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts a list of (original, changed, delta) bitstring tuples to numpy arrays."""
    arr_changed = []
    arr_original = []
    arr_delta = []

    for x, y, z in results:
        temp_original = []
        temp_changed = []
        temp_delta = []

        for i in range(len(x)):
            temp_original.append(int(x[i]))
            temp_changed.append(int(y[i]))
            temp_delta.append(int(z[i]))

        arr_original.append(temp_original)
        arr_changed.append(temp_changed)
        arr_delta.append(temp_delta)

    arr_changed = np.array(arr_changed, dtype=int)
    arr_original = np.array(arr_original, dtype=int)
    arr_delta = np.array(arr_delta, dtype=int)
    return arr_original, arr_changed, arr_delta


def change_bit(data: bytes, bit_index: int) -> bytes:
    """Flips the bit at the specified index in the byte string."""
    num_bits = len(data) * 8
    if not (0 <= bit_index < num_bits):
        raise ValueError(f"bit_index {bit_index} out of range (0–{num_bits - 1})")

    data_mut = bytearray(data)
    byte_index = bit_index // 8
    bit_pos = 7 - (bit_index % 8)
    data_mut[byte_index] ^= 1 << bit_pos
    return bytes(data_mut)


def bytes_to_int(b: bytes) -> int:
    """Converts a byte string to an integer (big-endian)."""
    return int.from_bytes(b, byteorder="big")


def hamming_distance_int(a: int, b: int) -> int:
    """Calculates the Hamming distance between two integers."""
    return (a ^ b).bit_count()


def generate_bytes(byte_range: int, num_bytes: int) -> Iterator[bytes]:
    """Generates sequential byte strings up to byte_range."""
    for i in range(byte_range):
        yield i.to_bytes(num_bytes, byteorder=sys.byteorder)


def generate_bytes_random_colision(num_bytes: int) -> Iterator[bytes]:
    """Generates infinite random byte strings."""
    while True:
        yield os.urandom(num_bytes)


@dataclass
class AlgorithmArgs:
    byte_range: int
    num_bytes: int
    hashing_function: Callable
    mode: Optional[str] = None
    pop_size: Optional[int] = None
    num_generations: Optional[int] = None
    prob_mutation: Optional[float] = None
    prob_crossover: Optional[float] = None
    tournament_size: Optional[int] = None
    pretty_print: bool = True
