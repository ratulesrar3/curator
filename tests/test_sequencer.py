import numpy as np
import pandas as pd
import pytest

from src import arcs, config
from src.sequencer import build_set, playtimes


@pytest.fixture(scope="module")
def result(mini_library, mini_adjacency):
    return build_set(
        mini_library, mini_adjacency, arcs.TEMPLATES["full_journey"],
        hours=1.0, vibe="test set", seed=11,
    )


def test_reaches_target_duration(result):
    df = result.tracklist
    total = df["start_s"].iloc[-1] + df["playtime_s"].iloc[-1]
    assert total >= 3600


def test_no_track_repeats(result):
    df = result.tracklist
    assert df[["track_name", "artists"]].duplicated().sum() == 0


def test_artist_cooldown(result):
    artists = result.tracklist["artists"].tolist()
    for i, artist in enumerate(artists):
        window = artists[max(0, i - config.ARTIST_COOLDOWN):i]
        assert artist not in window


def test_tempo_hard_cap_respected(result):
    df = result.tracklist.iloc[1:]
    unrelaxed = df[df["relaxed"] == ""]
    # worst-case blended tolerance is the loosest genre pairing
    max_tol = max(config.GENRE_BPM_TOLERANCE.values())
    ladder_max = max(m for m, _, _ in config.RELAX_LADDER)
    cap_pct = (np.exp(config.TEMPO_HARD_CAP_MULT * max_tol) - 1) * 100
    assert (unrelaxed["bpm_delta_pct"].abs() <= cap_pct + 1e-6).all()
    # even relaxed transitions stay inside the widest ladder rung
    relaxed_cap = (np.exp(config.TEMPO_HARD_CAP_MULT * max_tol * ladder_max) - 1) * 100
    assert (df["bpm_delta_pct"].abs() <= relaxed_cap + 1e-6).all()


def test_deterministic_with_seed(mini_library, mini_adjacency):
    a = build_set(mini_library, mini_adjacency, arcs.TEMPLATES["warmup"], 0.5, seed=3)
    b = build_set(mini_library, mini_adjacency, arcs.TEMPLATES["warmup"], 0.5, seed=3)
    pd.testing.assert_frame_equal(a.tracklist, b.tracklist)


def test_different_seed_changes_set(mini_library, mini_adjacency):
    a = build_set(mini_library, mini_adjacency, arcs.TEMPLATES["warmup"], 0.5, seed=3)
    b = build_set(mini_library, mini_adjacency, arcs.TEMPLATES["warmup"], 0.5, seed=4)
    assert not a.tracklist["track_name"].equals(b.tracklist["track_name"])


def test_playtimes_clipped(mini_library):
    p = playtimes(mini_library)
    assert (p >= config.PLAY_MIN_S).all() and (p <= config.PLAY_MAX_S).all()


def test_transition_columns_present(result):
    df = result.tracklist
    for col in ("t_harmonic", "t_tempo", "t_energy", "t_genre", "t_score",
                "bpm_delta_pct", "crossing", "target_energy"):
        assert col in df.columns
    assert df.iloc[1:]["t_score"].notna().all()
