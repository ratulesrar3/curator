import numpy as np
import pandas as pd
import pytest

from src import config

GENRE_BPM = {
    "hip hop": 92, "k-pop": 124, "house": 124, "techno": 130, "drum-and-bass": 174,
}


@pytest.fixture(scope="session")
def mini_library() -> pd.DataFrame:
    """Synthetic 100-track library with realistic clusters per genre."""
    rng = np.random.default_rng(7)
    rows = []
    keys = [f"{n}{l}" for n in range(1, 13) for l in ("A", "B")]
    for i in range(100):
        genre = config.GENRES[i % len(config.GENRES)]
        bpm = GENRE_BPM[genre] + rng.normal(0, 2.5)
        rows.append(
            {
                "track_id": f"t{i}",
                "track_name": f"Track {i}",
                "artists": f"Artist {i % 40}",   # some artist repeats
                "genre": genre,
                "bpm": round(bpm, 1),
                "effective_bpm": bpm * (2 if genre in config.HALFTIME_GENRES and bpm < 120 else 1),
                "halftime": genre in config.HALFTIME_GENRES and bpm < 120,
                "energy": float(np.clip(rng.uniform(0.3, 0.95), 0, 1)),
                "camelot": keys[rng.integers(0, 24)],
                "duration_ms": int(rng.uniform(180_000, 320_000)),
                "danceability": 0.7,
                "valence": 0.5,
                "acousticness": 0.1,
                "instrumentalness": 0.5,
                "popularity": 50,
                "bridge_to": config.GENRES[(i + 1) % len(config.GENRES)],
                "bridge_dist": 0.8,
                "is_bridge": i % 7 == 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def mini_adjacency() -> pd.DataFrame:
    base = {
        "hip hop":        [0.0, 0.9, 1.1, 1.3, 1.2],
        "k-pop":          [0.9, 0.0, 0.4, 0.5, 1.0],
        "house":          [1.1, 0.4, 0.0, 0.25, 0.9],
        "techno":         [1.3, 0.5, 0.25, 0.0, 0.8],
        "drum-and-bass":  [1.2, 1.0, 0.9, 0.8, 0.0],
    }
    return pd.DataFrame(base, index=config.GENRES)
