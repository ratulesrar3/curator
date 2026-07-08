"""
Camelot wheel math: Spotify (key, mode) -> Camelot code, and a precomputed
24x24 compatibility matrix for vectorized harmonic scoring.

Camelot codes are indexed 0..23 as (number-1)*2 + (0 if 'A' else 1),
so 1A=0, 1B=1, 2A=2, ... 12B=23.
"""

import numpy as np

from . import config

# Spotify pitch class + mode -> Camelot code (ported from genre-crossover)
CAMELOT_FROM_KEY_MODE = {
    (0, 1): "8B", (1, 1): "3B", (2, 1): "10B", (3, 1): "5B", (4, 1): "12B",
    (5, 1): "7B", (6, 1): "2B", (7, 1): "9B", (8, 1): "4B", (9, 1): "11B",
    (10, 1): "6B", (11, 1): "1B",
    (0, 0): "5A", (1, 0): "12A", (2, 0): "7A", (3, 0): "2A", (4, 0): "9A",
    (5, 0): "4A", (6, 0): "11A", (7, 0): "6A", (8, 0): "1A", (9, 0): "8A",
    (10, 0): "3A", (11, 0): "10A",
}


def parse(code: str) -> tuple[int, str]:
    """'8A' -> (8, 'A'). Raises ValueError on malformed codes."""
    number, letter = int(code[:-1]), code[-1].upper()
    if not (1 <= number <= 12) or letter not in ("A", "B"):
        raise ValueError(f"bad camelot code: {code!r}")
    return number, letter


def to_index(code: str) -> int:
    number, letter = parse(code)
    return (number - 1) * 2 + (0 if letter == "A" else 1)


def ring_distance(n1: int, n2: int) -> int:
    """Shortest distance around the 12-position wheel."""
    d = abs(n1 - n2) % 12
    return min(d, 12 - d)


def compatibility(code_a: str, code_b: str) -> float:
    """Harmonic mixing score in [0, 1] between two Camelot keys."""
    s = config.CAMELOT_SCORES
    n1, l1 = parse(code_a)
    n2, l2 = parse(code_b)
    dist = ring_distance(n1, n2)
    same_letter = l1 == l2

    if dist == 0:
        return s["same"] if same_letter else s["relative"]
    if dist == 1:
        return s["neighbor"] if same_letter else s["diagonal"]
    if dist == 2 and same_letter:
        return s["energy_boost"]
    if dist == 3 and same_letter:
        return s["step3"]
    return s["clash"]


def build_matrix() -> np.ndarray:
    """24x24 compatibility matrix indexed by to_index()."""
    codes = [f"{n}{l}" for n in range(1, 13) for l in ("A", "B")]
    m = np.zeros((24, 24))
    for a in codes:
        for b in codes:
            m[to_index(a), to_index(b)] = compatibility(a, b)
    return m


MATRIX = build_matrix()
