"""
Track library builder.

Reads the cached HuggingFace dump (7,000 rows across 7 raw genre labels),
cleans and dedupes it, computes Camelot keys, effective BPM, genre centroids,
the genre adjacency matrix, and bridge scores, then writes data/library.csv.
"""

import os

import numpy as np
import pandas as pd

from . import camelot, config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "spotify_raw.csv")
LIBRARY_PATH = os.path.join(ROOT, "data", "library.csv")
ADJACENCY_PATH = os.path.join(ROOT, "output", "genre_adjacency.csv")

KEEP_COLS = [
    "track_id", "track_name", "artists", "genre", "bpm", "effective_bpm",
    "halftime", "energy", "camelot", "duration_ms", "danceability",
    "valence", "acousticness", "instrumentalness", "popularity",
    "bridge_to", "bridge_dist", "is_bridge",
]


def genre_space(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Z-standardized (effective_bpm, energy) space + per-genre centroids.

    Genres absent from df get a centroid at the global mean so every
    downstream distance stays finite (an MSD-derived library has ~no k-pop).
    """
    feats = df[["effective_bpm", "energy"]].to_numpy(dtype=float)
    mu, sd = feats.mean(axis=0), feats.std(axis=0)
    z = (feats - mu) / sd
    genres = df["genre"].to_numpy()
    centroids = {}
    for g in config.GENRES:
        pts = z[genres == g]
        centroids[g] = pts.mean(axis=0) if len(pts) else np.zeros(2)
    return z, mu, sd, centroids


def _adjacency(centroids: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {a: {b: float(np.linalg.norm(centroids[a] - centroids[b])) for b in config.GENRES}
         for a in config.GENRES}
    ).round(3)


def adjacency_from(df: pd.DataFrame) -> pd.DataFrame:
    """Genre adjacency matrix recomputed from a loaded library frame."""
    return _adjacency(genre_space(df)[3])


def annotate_bridges(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add bridge_to / bridge_dist / is_bridge; return (df, adjacency).

    Bridge scoring: distance to the nearest foreign centroid, plus a
    pitch-fader tempo check against that centroid's BPM. Genres with no
    tracks in df are never bridge targets.
    """
    z, mu, sd, centroids = genre_space(df)
    genres = df["genre"].to_numpy()

    centroid_arr = np.stack([centroids[g] for g in config.GENRES])
    dists = np.linalg.norm(z[:, None, :] - centroid_arr[None, :, :], axis=2)
    own_idx = np.array([config.GENRES.index(g) for g in genres])
    dists[np.arange(len(df)), own_idx] = np.inf
    present = set(genres)
    for j, g in enumerate(config.GENRES):
        if g not in present:
            dists[:, j] = np.inf
    nearest = dists.argmin(axis=1)

    df["bridge_to"] = [config.GENRES[i] for i in nearest]
    df["bridge_dist"] = dists[np.arange(len(df)), nearest]
    centroid_bpm = centroid_arr[:, 0] * sd[0] + mu[0]
    tempo_gap = np.abs(df["effective_bpm"].to_numpy() - centroid_bpm[nearest])
    tempo_ok = tempo_gap / np.maximum(df["effective_bpm"].to_numpy(), 1) <= config.BRIDGE_TEMPO_TOL
    df["is_bridge"] = (df["bridge_dist"] < config.BRIDGE_DIST_MAX) & tempo_ok
    return df, _adjacency(centroids)


def build(raw_path: str = RAW_PATH, library_path: str = LIBRARY_PATH,
          adjacency_path: str = ADJACENCY_PATH) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)

    df = raw[raw["track_genre"].isin(config.GENRE_MAP)].copy()
    df["genre"] = df["track_genre"].map(config.GENRE_MAP)
    df = df.rename(columns={"tempo": "bpm"})

    # Clean: playable tempo, DJ-set-appropriate duration, valid key
    df = df[(df["bpm"] > 0) & (df["key"] >= 0)]
    df = df[df["duration_ms"].between(config.MIN_DURATION_MS, config.MAX_DURATION_MS)]

    # Dedupe: same Spotify ID, then same (title, artists) across releases
    df = df.drop_duplicates(subset="track_id")
    name_key = df["track_name"].str.lower().str.strip() + "|" + df["artists"].str.lower().str.strip()
    df = df[~name_key.duplicated()]

    df["camelot"] = df.apply(
        lambda r: camelot.CAMELOT_FROM_KEY_MODE[(int(r["key"]), int(r["mode"]))], axis=1
    )

    # Effective BPM for pocket/arc analysis (pairwise scoring handles
    # half/double-time natively; see transitions.py)
    df["halftime"] = df["genre"].isin(config.HALFTIME_GENRES) & (df["bpm"] < 120)
    df["effective_bpm"] = np.where(df["halftime"], df["bpm"] * 2, df["bpm"])

    df = df.reset_index(drop=True)
    df, adjacency = annotate_bridges(df)

    df = df[KEEP_COLS]
    os.makedirs(os.path.dirname(library_path), exist_ok=True)
    os.makedirs(os.path.dirname(adjacency_path), exist_ok=True)
    df.to_csv(library_path, index=False)
    adjacency.to_csv(adjacency_path)
    return df


def load(library_path: str = LIBRARY_PATH) -> pd.DataFrame:
    if not os.path.exists(library_path):
        return build()
    return pd.read_csv(library_path)


def load_adjacency(adjacency_path: str = ADJACENCY_PATH) -> pd.DataFrame:
    if not os.path.exists(adjacency_path):
        build()
    return pd.read_csv(adjacency_path, index_col=0)


if __name__ == "__main__":
    lib = build()
    print(f"library: {len(lib)} tracks")
    print(lib["genre"].value_counts().to_string())
    print(f"bridges: {int(lib['is_bridge'].sum())}")
