"""
Million Song Dataset -> curator library adapter.

MSD is not a drop-in base dataset: its `energy`/`danceability` fields are
0.0 for every track and it carries no genre labels (see
docs/msd_feasibility.md). This adapter makes it usable as a *supplementary*
source by joining two companion files:

  data/msd/msd_summary_file.h5  - per-track audio features (tempo, key/mode
                                  + confidences, loudness, duration)
  data/msd/lastfm_tags.db       - Last.fm tags, the only MSD companion with
                                  granular electronic subgenres

Genres come from a curated tag->genre map; energy comes from a linear
loudness->energy proxy fit on the Spotify library (both columns exist in
data/spotify_raw.csv). Spotify-only columns (valence etc.) are NaN - they
are pass-through metadata, never scored. Output is data/library_msd.csv in
the exact library schema, plus output/msd_coverage.md with the funnel.
"""

import os
import sqlite3

import h5py
import numpy as np
import pandas as pd

from . import camelot, config, library

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(ROOT, "data", "msd", "msd_summary_file.h5")
TAGS_PATH = os.path.join(ROOT, "data", "msd", "lastfm_tags.db")
MSD_LIBRARY_PATH = os.path.join(ROOT, "data", "library_msd.csv")
COVERAGE_PATH = os.path.join(ROOT, "output", "msd_coverage.md")

# Lowercase Last.fm tag -> curator genre. Chosen from an empirical survey of
# tag frequencies at val >= MSD_TAG_VAL_MIN; generic tags ("electronic",
# "electro") and ambiguous hybrids ("techno house") are deliberately absent.
TAG_MAP = {
    "hip-hop": "hip hop", "hip hop": "hip hop", "hiphop": "hip hop",
    "rap": "hip hop", "underground hip-hop": "hip hop",
    "underground hip hop": "hip hop", "instrumental hip-hop": "hip hop",
    "true hip hop": "hip hop",

    "k-pop": "k-pop", "kpop": "k-pop", "korean pop": "k-pop",

    "house": "house", "deep house": "house", "progressive house": "house",
    "tech house": "house", "tech-house": "house", "electro house": "house",
    "vocal house": "house", "soulful house": "house", "funky house": "house",
    "minimal house": "house", "acid house": "house", "chicago house": "house",

    "techno": "techno", "minimal techno": "techno", "minimal-techno": "techno",
    "detroit techno": "techno", "deep techno": "techno", "dub techno": "techno",
    "hard techno": "techno", "techno minimal": "techno",

    "drum and bass": "drum-and-bass", "drum n bass": "drum-and-bass",
    "drum'n'bass": "drum-and-bass", "drum & bass": "drum-and-bass",
    "drum 'n' bass": "drum-and-bass", "drumnbass": "drum-and-bass",
    "dnb": "drum-and-bass", "jungle": "drum-and-bass",
    "liquid funk": "drum-and-bass", "atmospheric drum and bass": "drum-and-bass",
}

ANALYSIS_FIELDS = [
    "track_id", "tempo", "key", "key_confidence", "mode", "mode_confidence",
    "loudness", "duration",
]
SLAB = 200_000


def tag_rows(tags_path: str = TAGS_PATH) -> pd.DataFrame:
    """(track_id, tag, val) for every mapped tag at/above the weight floor."""
    con = sqlite3.connect(tags_path)
    placeholders = ",".join("?" * len(TAG_MAP))
    q = f"""
        SELECT tids.tid AS track_id, lower(tags.tag) AS tag, MAX(tid_tag.val) AS val
        FROM tid_tag
        JOIN tags ON tags.ROWID = tid_tag.tag
        JOIN tids ON tids.ROWID = tid_tag.tid
        WHERE lower(tags.tag) IN ({placeholders}) AND tid_tag.val >= ?
        GROUP BY tids.tid, lower(tags.tag)
    """
    try:
        return pd.read_sql_query(q, con, params=(*TAG_MAP, config.MSD_TAG_VAL_MIN))
    finally:
        con.close()


def resolve_genres(rows: pd.DataFrame) -> pd.DataFrame:
    """Winner genre per track by strongest tag weight; exact ties are
    ambiguous and dropped. Returns (track_id, genre, tag_val)."""
    df = rows.copy()
    df["genre"] = df["tag"].map(TAG_MAP)
    per = df.groupby(["track_id", "genre"], as_index=False)["val"].max()
    top = per[per["val"] == per.groupby("track_id")["val"].transform("max")]
    top = top[~top["track_id"].duplicated(keep=False)]
    return top.rename(columns={"val": "tag_val"}).reset_index(drop=True)


def read_summary(track_ids: pd.Series, summary_path: str = SUMMARY_PATH) -> pd.DataFrame:
    """One slab-scan of the 1M-row summary file, keeping only wanted tracks."""
    wanted = np.sort(track_ids.to_numpy(dtype="S"))
    chunks = []
    with h5py.File(summary_path, "r") as f:
        analysis = f["analysis/songs"]
        meta = f["metadata/songs"]
        n = analysis.shape[0]
        for i in range(0, n, SLAB):
            a = analysis.fields(ANALYSIS_FIELDS)[i:i + SLAB]
            mask = np.isin(a["track_id"], wanted)
            if not mask.any():
                continue
            a = a[mask]
            m = meta.fields(["title", "artist_name"])[i:i + SLAB][mask]
            chunk = {fld: a[fld] for fld in ANALYSIS_FIELDS}
            chunk["track_id"] = chunk["track_id"].astype("U")  # IDs are ASCII
            chunk["track_name"] = np.char.decode(m["title"], "utf-8", "replace")
            chunk["artists"] = np.char.decode(m["artist_name"], "utf-8", "replace")
            chunks.append(pd.DataFrame(chunk))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=ANALYSIS_FIELDS + ["track_name", "artists"]
    )


def fit_energy_proxy(raw_path: str = library.RAW_PATH) -> dict:
    """Linear energy ~ loudness fit on the Spotify dump (all rows are the
    same five genres the MSD tracks are mapped onto)."""
    df = pd.read_csv(raw_path, usecols=["energy", "loudness"]).dropna()
    slope, intercept = np.polyfit(df["loudness"], df["energy"], 1)
    pred = slope * df["loudness"] + intercept
    ss_res = float(((df["energy"] - pred) ** 2).sum())
    ss_tot = float(((df["energy"] - df["energy"].mean()) ** 2).sum())
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot,
        "n": len(df),
    }


def apply_energy_proxy(loudness: np.ndarray, proxy: dict) -> np.ndarray:
    return np.clip(proxy["slope"] * np.asarray(loudness, dtype=float)
                   + proxy["intercept"], 0.0, 1.0)


def assemble(feats: pd.DataFrame, proxy: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Filter + transform a merged (features x genre) frame into the exact
    library schema. Returns (library_df, adjacency, funnel_counts)."""
    funnel = {"tagged & matched": len(feats)}
    df = feats.copy()
    df = df.rename(columns={"tempo": "bpm"})
    df["duration_ms"] = (df["duration"] * 1000).round().astype(int)

    df = df[(df["bpm"] > 0) & df["key"].between(0, 11) & df["mode"].isin((0, 1))]
    df = df[df["duration_ms"].between(config.MIN_DURATION_MS, config.MAX_DURATION_MS)]
    funnel["playable tempo/key & DJ-set duration"] = len(df)

    df = df[df["key_confidence"] >= config.MSD_KEY_CONF_MIN]
    funnel[f"key confidence >= {config.MSD_KEY_CONF_MIN}"] = len(df)

    df = df.drop_duplicates(subset="track_id")
    name_key = (df["track_name"].str.lower().str.strip() + "|"
                + df["artists"].str.lower().str.strip())
    df = df[~name_key.duplicated()]
    funnel["deduped (title, artist)"] = len(df)

    df["energy"] = apply_energy_proxy(df["loudness"].to_numpy(), proxy)
    df["camelot"] = df.apply(
        lambda r: camelot.CAMELOT_FROM_KEY_MODE[(int(r["key"]), int(r["mode"]))], axis=1
    )
    df["halftime"] = df["genre"].isin(config.HALFTIME_GENRES) & (df["bpm"] < 120)
    df["effective_bpm"] = np.where(df["halftime"], df["bpm"] * 2, df["bpm"])
    for col in ("danceability", "valence", "acousticness", "instrumentalness",
                "popularity"):
        df[col] = np.nan

    df = df.reset_index(drop=True)
    df, adjacency = library.annotate_bridges(df)
    return df[library.KEEP_COLS], adjacency, funnel


def coverage_report(df: pd.DataFrame, adjacency: pd.DataFrame, proxy: dict,
                    funnel: dict, tag_funnel: dict) -> str:
    genre_counts = df["genre"].value_counts()
    sparse = sorted(g for g in config.GENRES if 0 < genre_counts.get(g, 0) < 50)
    lines = [
        "# MSD -> curator coverage report",
        "",
        "Generated by `python main.py --rebuild-msd-library`. Verdict and "
        "methodology: `docs/msd_feasibility.md`.",
        "",
        "## Funnel",
        "",
        "| stage | tracks |",
        "|---|---|",
        *(f"| {k} | {v:,} |" for k, v in {**tag_funnel, **funnel}.items()),
        "",
        "## Final tracks per genre",
        "",
        "| genre | tracks | share |",
        "|---|---|---|",
        *(f"| {g} | {genre_counts.get(g, 0):,} | "
          f"{100 * genre_counts.get(g, 0) / max(len(df), 1):.1f}% |"
          for g in config.GENRES),
        "",
        f"Bridge tracks: {int(df['is_bridge'].sum()):,}",
        "",
        "## Energy proxy",
        "",
        f"`energy = clip({proxy['slope']:.4f} x loudness + {proxy['intercept']:.4f}, 0, 1)` "
        f"- linear fit on {proxy['n']:,} Spotify rows, **R² = {proxy['r2']:.3f}**. "
        "In this dance-heavy five-genre domain the loudness range is "
        "compressed, so loudness recovers only a coarse energy signal - "
        "usable for arc-following, too lossy for fine-grained comparisons "
        "between similar tracks. Treat every MSD energy value as approximate.",
        "",
        "## Genre adjacency (z-space centroid distance)",
        "",
        "```",
        adjacency.to_string(),
        "```",
        "",
        *([f"Sparse genres ({', '.join(sparse)}: fewer than 50 tracks) have "
           "noise centroids - read their adjacency rows as artifacts, not "
           "structure. MSD's catalog is frozen at 2011, so k-pop essentially "
           "does not exist in it."] if sparse else []),
        "",
    ]
    return "\n".join(lines)


def build(summary_path: str = SUMMARY_PATH, tags_path: str = TAGS_PATH,
          library_path: str = MSD_LIBRARY_PATH,
          coverage_path: str = COVERAGE_PATH) -> pd.DataFrame:
    rows = tag_rows(tags_path)
    genres = resolve_genres(rows)
    tag_funnel = {
        "tracks with a mapped Last.fm tag": rows["track_id"].nunique(),
        "unambiguous winner genre": len(genres),
    }

    feats = read_summary(genres["track_id"], summary_path)
    feats = feats.merge(genres[["track_id", "genre"]], on="track_id", how="inner")

    proxy = fit_energy_proxy()
    df, adjacency, funnel = assemble(feats, proxy)

    os.makedirs(os.path.dirname(library_path), exist_ok=True)
    os.makedirs(os.path.dirname(coverage_path), exist_ok=True)
    df.to_csv(library_path, index=False)
    with open(coverage_path, "w") as f:
        f.write(coverage_report(df, adjacency, proxy, funnel, tag_funnel))
    return df
