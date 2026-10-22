"""Difference hash (dHash) for finding near-duplicate images across splits.

dHash shrinks an image to 9x8 grayscale and records, for each row, whether each pixel
is brighter than its right neighbour: 64 bits that barely change under resizing,
compression or small brightness shifts. Two images whose hashes differ in only a few
bits (Hamming distance) are almost certainly the same shot. That catches video frames
or public-dataset images that slipped into two splits.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

HASH_SIZE = 8


def dhash(image: np.ndarray) -> int:
    """64-bit dHash of a BGR or grayscale image (as returned by cv2.imread)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (HASH_SIZE + 1, HASH_SIZE), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def to_hex(value: int) -> str:
    return f"{value:016x}"


def near_duplicates(
    a: Sequence[str], b: Sequence[str], max_distance: int, block: int = 1024
) -> list[tuple[int, int, int]]:
    """(index in a, index in b, distance) for every pair of hex hashes within max_distance."""
    hashes_a = np.array([int(h, 16) for h in a], dtype=np.uint64)
    hashes_b = np.array([int(h, 16) for h in b], dtype=np.uint64)
    pairs = []
    # Compare in blocks so memory stays at block * len(b) instead of len(a) * len(b).
    for start in range(0, len(hashes_a), block):
        distances = np.bitwise_count(hashes_a[start : start + block, None] ^ hashes_b[None, :])
        for i, j in zip(*np.nonzero(distances <= max_distance), strict=True):
            pairs.append((start + int(i), int(j), int(distances[i, j])))
    return pairs
