"""
Deterministic transition rationales and quantitative set metrics.
Every explanation is derived from the actual score components — nothing
is generated, so nothing can be hallucinated.
"""

import numpy as np
import pandas as pd

from .sequencer import SetResult

HARMONIC_OK = 0.65  # diagonal or better counts as harmonically compatible


def hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def rationale(prev: pd.Series, row: pd.Series) -> str:
    parts = [f"{prev['camelot']}→{row['camelot']}"]
    if row["t_harmonic"] >= 0.95:
        parts[-1] += " (same/relative key)"
    elif row["t_harmonic"] >= HARMONIC_OK:
        parts[-1] += " (compatible)"
    else:
        parts[-1] += " (clash)"

    ratio = row.get("bpm_ratio", 1.0)
    bpm_bit = f"{prev['bpm']:.0f}→{row['bpm']:.0f} BPM ({row['bpm_delta_pct']:+.1f}%)"
    if ratio == 2.0:
        bpm_bit += " double-time"
    elif ratio == 0.5:
        bpm_bit += " half-time"
    parts.append(bpm_bit)

    parts.append(f"energy {prev['energy']:.2f}→{row['energy']:.2f}")

    if row["crossing"]:
        cross = f"{prev['genre']}→{row['genre']}"
        if row["t_bridge"]:
            cross += " via bridge"
        parts.append(cross)

    if row["relaxed"]:
        parts.append(f"[{row['relaxed']}]")
    return " · ".join(parts)


def add_rationales(result: SetResult) -> pd.DataFrame:
    df = result.tracklist.copy()
    notes = [""]
    for i in range(1, len(df)):
        notes.append(rationale(df.iloc[i - 1], df.iloc[i]))
    df["rationale"] = notes
    df["start_hms"] = df["start_s"].map(hms)
    return df


def set_report(result: SetResult) -> dict:
    df = result.tracklist
    trans = df.iloc[1:]
    duration_s = float(df["start_s"].iloc[-1] + df["playtime_s"].iloc[-1])
    genre_counts = df["genre"].value_counts().to_dict()

    return {
        "vibe": result.vibe,
        "template": result.template.name,
        "seed": result.seed,
        "n_tracks": len(df),
        "duration": hms(duration_s),
        "duration_s": round(duration_s, 1),
        "harmonic_pct": round(100 * (trans["t_harmonic"] >= HARMONIC_OK).mean(), 1),
        "mean_abs_bpm_delta_pct": round(trans["bpm_delta_pct"].abs().mean(), 2),
        "mean_transition_score": round(trans["t_score"].mean(), 3),
        "arc_rmse": round(
            float(np.sqrt(((df["energy"] - df["target_energy"]) ** 2).mean())), 3
        ),
        "genre_counts": genre_counts,
        "genre_crossings": int(trans["crossing"].sum()),
        "bridge_crossings": int(trans["t_bridge"].sum()),
        "bridge_tracks_used": int(df["is_bridge"].sum()),
        "relaxed_transitions": int((df["relaxed"] != "").sum()),
        "artists_unique": int(df["artists"].nunique()),
    }


def format_report(report: dict) -> str:
    genres = ", ".join(f"{g} {c}" for g, c in report["genre_counts"].items())
    return "\n".join(
        [
            f"vibe:        {report['vibe'] or '(none)'}",
            f"template:    {report['template']}  (seed {report['seed']})",
            f"tracks:      {report['n_tracks']}  over {report['duration']}",
            f"harmonic:    {report['harmonic_pct']}% of transitions compatible",
            f"tempo:       mean |ΔBPM| {report['mean_abs_bpm_delta_pct']}%",
            f"arc fit:     RMSE {report['arc_rmse']} vs target curve",
            f"transitions: mean score {report['mean_transition_score']}, "
            f"{report['genre_crossings']} genre crossings "
            f"({report['bridge_crossings']} via bridge tracks), "
            f"{report['relaxed_transitions']} relaxed",
            f"genres:      {genres}",
            f"artists:     {report['artists_unique']} unique",
        ]
    )
