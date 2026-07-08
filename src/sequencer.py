"""
Beam-search sequencer: turns (library, arc template, duration) into an
ordered tracklist that follows the energy curve.

Hard constraints: no track repeats, artist cooldown, tempo hard cap.
Everything else (key clashes, energy jumps, genre distance) is a soft score.
Dead ends trigger a relaxation ladder instead of failing, tagging the
transition so it can be surfaced as an "echo out / creative" moment.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .arcs import ArcTemplate
from .transitions import TransitionScorer


@dataclass
class Beam:
    tracks: list[int]
    used: set[int]
    seconds: float
    score: float                      # cumulative search score
    relaxed: dict[int, str] = field(default_factory=dict)  # position -> label


@dataclass
class SetResult:
    tracklist: pd.DataFrame
    template: ArcTemplate
    vibe: str
    hours: float
    seed: int
    agent_notes: dict | None = None   # set by src/agent.py; rides into exports


def playtimes(library: pd.DataFrame) -> np.ndarray:
    secs = library["duration_ms"].to_numpy(dtype=float) / 1000.0 * config.PLAY_FRACTION
    return np.clip(secs, config.PLAY_MIN_S, config.PLAY_MAX_S)


def _genre_weight_array(weights: dict[str, float], boosts: dict[str, float]) -> np.ndarray:
    w = np.array([weights[g] * boosts.get(g, 1.0) for g in config.GENRES])
    return w / w.mean()


def build_set(
    library: pd.DataFrame,
    adjacency: pd.DataFrame,
    template: ArcTemplate,
    hours: float,
    boosts: dict[str, float] | None = None,
    vibe: str = "",
    seed: int = 42,
) -> SetResult:
    boosts = boosts or {}
    scorer = TransitionScorer(library, adjacency)
    lib = scorer.library
    n = scorer.n
    play = playtimes(lib)
    artist_idx = pd.factorize(lib["artists"])[0]
    total_s = hours * 3600.0
    rng = np.random.default_rng(seed)

    # ---- opening: rank by fit to the arc's starting target -------------
    e0, w0 = template.target(0.0)
    w_arr = _genre_weight_array(w0, boosts)
    open_fit = (
        np.exp(-(((scorer.energy - e0) / config.ENERGY_SIGMA) ** 2))
        * w_arr[scorer.genre_idx]
        + rng.normal(0, config.TIE_JITTER, n)
    )
    beams: list[Beam] = []
    seen_artists: set[int] = set()
    for i in np.argsort(-open_fit):
        if artist_idx[i] in seen_artists:
            continue
        seen_artists.add(artist_idx[i])
        beams.append(Beam([int(i)], {int(i)}, float(play[i]), 0.0))
        if len(beams) >= config.BEAM_WIDTH:
            break

    # ---- beam expansion -------------------------------------------------
    finished: list[Beam] = []
    while beams:
        candidates: list[Beam] = []
        for beam in beams:
            a = beam.tracks[-1]
            t_frac = min(beam.seconds / total_s, 1.0)
            target_e, weights = template.target(t_frac)
            w_arr = _genre_weight_array(weights, boosts)

            blocked = np.zeros(n, dtype=bool)
            blocked[list(beam.used)] = True
            recent_artists = {artist_idx[t] for t in beam.tracks[-config.ARTIST_COOLDOWN:]}
            blocked |= np.isin(artist_idx, list(recent_artists))

            relax_label = None
            scores, allowed = scorer.score_candidates(a, target_e)
            total = scores * w_arr[scorer.genre_idx] + rng.normal(0, config.TIE_JITTER, n)
            total[~allowed | blocked] = -np.inf

            if total.max() < config.SCORE_FLOOR:
                for tol_mult, harm_mult, label in config.RELAX_LADDER:
                    scores, allowed = scorer.score_candidates(a, target_e, tol_mult, harm_mult)
                    total = scores * w_arr[scorer.genre_idx] + rng.normal(0, config.TIE_JITTER, n)
                    total[~allowed | blocked] = -np.inf
                    if total.max() > -np.inf:
                        relax_label = label
                        break
                if total.max() == -np.inf:
                    continue  # beam truly dead; drop it

            top = np.argpartition(-total, config.EXPAND_PER_BEAM)[: config.EXPAND_PER_BEAM]
            for b in top:
                if total[b] == -np.inf:
                    continue
                nb = Beam(
                    beam.tracks + [int(b)],
                    beam.used | {int(b)},
                    beam.seconds + float(play[b]),
                    beam.score + float(total[b]),
                    dict(beam.relaxed),
                )
                if relax_label:
                    nb.relaxed[len(nb.tracks) - 1] = relax_label
                candidates.append(nb)

        if not candidates:
            break
        candidates.sort(key=lambda x: -x.score)
        beams = []
        for c in candidates[: config.BEAM_WIDTH * 2]:
            if c.seconds >= total_s:
                finished.append(c)
            else:
                beams.append(c)
            if len(beams) >= config.BEAM_WIDTH:
                break

    pool = finished if finished else beams
    if not pool:
        raise RuntimeError("sequencer found no viable set — library too constrained")
    best = max(pool, key=lambda b: b.score / max(len(b.tracks) - 1, 1))

    return SetResult(
        tracklist=_assemble(best, scorer, play, template, total_s),
        template=template,
        vibe=vibe,
        hours=hours,
        seed=seed,
    )


def _assemble(
    beam: Beam,
    scorer: TransitionScorer,
    play: np.ndarray,
    template: ArcTemplate,
    total_s: float,
) -> pd.DataFrame:
    lib = scorer.library
    rows = []
    t = 0.0
    for pos, idx in enumerate(beam.tracks):
        track = lib.iloc[idx]
        target_e, _ = template.target(min(t / total_s, 1.0))
        row = {
            "position": pos + 1,
            "start_s": round(t, 1),
            "track_name": track["track_name"],
            "artists": track["artists"],
            "genre": track["genre"],
            "bpm": round(float(track["bpm"]), 1),
            "camelot": track["camelot"],
            "energy": round(float(track["energy"]), 3),
            "playtime_s": round(float(play[idx]), 1),
            "is_bridge": bool(track["is_bridge"]),
            "bridge_to": track["bridge_to"] if track["is_bridge"] else "",
            "target_energy": round(target_e, 3),
            "relaxed": beam.relaxed.get(pos, ""),
        }
        if pos > 0:
            comp = scorer.pair_components(beam.tracks[pos - 1], idx, target_e)
            row.update(
                {
                    "t_harmonic": round(comp["harmonic"], 3),
                    "t_tempo": round(comp["tempo"], 3),
                    "t_energy": round(comp["energy"], 3),
                    "t_genre": round(comp["genre"], 3),
                    "t_bridge": comp["bridge"],
                    "t_score": round(comp["composite"], 3),
                    "bpm_ratio": comp["bpm_ratio"],
                    "bpm_delta_pct": round(comp["bpm_delta_pct"], 2),
                    "crossing": comp["crossing"],
                }
            )
        rows.append(row)
        t += float(play[idx])
    return pd.DataFrame(rows)
