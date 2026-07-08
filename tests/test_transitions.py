import numpy as np
import pandas as pd
import pytest

from src import config
from src.transitions import TransitionScorer


def _row(track_id, genre, bpm, camelot="8A", energy=0.7, artists="X"):
    return {
        "track_id": track_id, "track_name": track_id, "artists": artists,
        "genre": genre, "bpm": bpm, "effective_bpm": bpm, "halftime": False,
        "energy": energy, "camelot": camelot, "duration_ms": 240_000,
        "danceability": 0.7, "valence": 0.5, "acousticness": 0.1,
        "instrumentalness": 0.5, "popularity": 50,
        "bridge_to": "house", "bridge_dist": 0.9, "is_bridge": False,
    }


@pytest.fixture()
def scorer(mini_adjacency):
    rows = [
        _row("a", "drum-and-bass", 174.0),          # 0
        _row("b", "hip hop", 87.0),                  # 1: exact half-time of 0
        _row("c", "techno", 130.0),                  # 2
        _row("d", "techno", 131.0),                  # 3: within techno tolerance
        _row("e", "techno", 145.0),                  # 4: way outside tolerance
        _row("f", "hip hop", 95.0),                  # 5
    ]
    return TransitionScorer(pd.DataFrame(rows), mini_adjacency)


def test_halftime_double_time_is_compatible(scorer):
    scores, allowed = scorer.tempo_scores(0)   # 174 BPM dnb
    assert allowed[1]                           # 87 BPM hip hop reachable at 2x
    # perfect ratio match, discounted only by the blend penalty
    assert scores[1] == pytest.approx(config.HALFTIME_BLEND_PENALTY)


def test_close_tempo_scores_higher_than_far(scorer):
    scores, allowed = scorer.tempo_scores(2)   # 130 techno
    assert scores[3] > 0.9                      # 131 easy blend
    assert not allowed[4]                       # 145 beyond the hard cap for techno


def test_genre_tolerance_blend(scorer):
    # the same absolute pct delta is scored more leniently for hip hop pairs
    # (10% tol) than techno pairs (4% tol)
    pct = 0.03
    tol_techno = config.GENRE_BPM_TOLERANCE["techno"]
    tol_hiphop = config.GENRE_BPM_TOLERANCE["hip hop"]
    assert np.exp(-((pct / tol_hiphop) ** 2)) > np.exp(-((pct / tol_techno) ** 2))


def test_score_candidates_forbids_hard_cap_only(scorer):
    scores, allowed = scorer.score_candidates(2, target_energy=0.7)
    assert not allowed[4]
    # key clashes stay allowed (soft penalty), same-key same-bpm ranks top
    assert allowed[3]
    assert scores[3] == max(scores[i] for i in range(scorer.n) if allowed[i] and i != 2)


def test_pair_components_breakdown(scorer):
    comp = scorer.pair_components(0, 1, target_energy=0.7)
    assert comp["bpm_ratio"] == 0.5              # 174 -> 87 half-time
    assert comp["crossing"] is True
    assert 0 <= comp["composite"] <= 1.2
    for key in ("harmonic", "tempo", "energy", "genre", "bridge"):
        assert key in comp


def test_relaxation_widens_reach(scorer):
    _, allowed_strict = scorer.score_candidates(2, 0.7)
    _, allowed_relaxed = scorer.score_candidates(2, 0.7, tol_mult=2.5)
    assert allowed_relaxed.sum() >= allowed_strict.sum()
    assert allowed_relaxed[4]                    # 145 reachable once relaxed
