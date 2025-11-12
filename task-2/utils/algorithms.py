import hashlib
import os
import struct
import sys
from typing import List


def run_md5(data: bytes) -> bytes:
    """Computes the MD5 hash of the input data."""
    return hashlib.md5(data).digest()


def run_sha(data: bytes) -> bytes:
    """Computes the SHA-256 hash of the input data."""
    return hashlib.sha256(data).digest()


def run_random_hash(data: bytes) -> bytes:
    """Generates random bytes of the same length as the input data."""
    return os.urandom(len(data))


KEY_XOR = os.urandom(64)
KEY_MUL = os.urandom(64)


def run_MatTheHash(data: bytes) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    if len(data) != 64:
        raise ValueError("run_MatTheHash expects exactly 32 bytes of input")

    x = int.from_bytes(data, sys.byteorder)
    xor_key = int.from_bytes(KEY_XOR, sys.byteorder)
    mul_key = int.from_bytes(KEY_MUL, sys.byteorder)
    x ^= xor_key
    x = (x * mul_key) % (1 << 512)
    return x.to_bytes(64, sys.byteorder)
