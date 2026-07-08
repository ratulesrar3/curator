import numpy as np
import pytest

from src import camelot


def test_key_mode_mapping_known_values():
    # C major = 8B, A minor = 8A (relative pair)
    assert camelot.CAMELOT_FROM_KEY_MODE[(0, 1)] == "8B"
    assert camelot.CAMELOT_FROM_KEY_MODE[(9, 0)] == "8A"


def test_parse_and_index_roundtrip():
    assert camelot.parse("8A") == (8, "A")
    assert camelot.parse("12B") == (12, "B")
    assert camelot.to_index("1A") == 0
    assert camelot.to_index("12B") == 23
    with pytest.raises(ValueError):
        camelot.parse("13A")


def test_ring_distance_wraps():
    assert camelot.ring_distance(1, 12) == 1
    assert camelot.ring_distance(1, 7) == 6
    assert camelot.ring_distance(5, 5) == 0


def test_compatibility_truths():
    assert camelot.compatibility("8A", "8A") == 1.0          # same key
    assert camelot.compatibility("8A", "8B") == 0.95         # relative major/minor
    assert camelot.compatibility("8A", "9A") == 0.85         # neighbor
    assert camelot.compatibility("8A", "7A") == 0.85
    assert camelot.compatibility("8A", "9B") == 0.65         # diagonal
    assert camelot.compatibility("8A", "10A") == 0.55        # +2 energy boost
    assert camelot.compatibility("8A", "3B") == pytest.approx(0.10)  # clash


def test_matrix_is_symmetric_and_bounded():
    m = camelot.MATRIX
    assert m.shape == (24, 24)
    assert np.allclose(m, m.T)
    assert m.min() > 0 and m.max() == 1.0
    assert np.allclose(np.diag(m), 1.0)
