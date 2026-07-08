"""
Vectorized transition scoring.

No precomputed pairwise matrix: each beam expansion scores the current track
against the whole library in O(N) numpy. Tempo compatibility natively
considers straight, double-time, and half-time blends; harmonic and genre
compatibility come from precomputed lookup tables.
"""

import numpy as np
import pandas as pd

from . import camelot, config


class TransitionScorer:
    def __init__(self, library: pd.DataFrame, adjacency: pd.DataFrame):
        self.library = library.reset_index(drop=True)
        n = len(self.library)

        self.bpm = self.library["bpm"].to_numpy(dtype=float)
        self.log_bpm = np.log(self.bpm)
        self.energy = self.library["energy"].to_numpy(dtype=float)
        self.camelot_idx = np.array([camelot.to_index(c) for c in self.library["camelot"]])
        self.genre_idx = np.array([config.GENRES.index(g) for g in self.library["genre"]])
        self.is_bridge = self.library["is_bridge"].to_numpy(dtype=bool)
        self.bridge_to_idx = np.array(
            [config.GENRES.index(g) for g in self.library["bridge_to"]]
        )

        self.genre_tol = np.array(
            [config.GENRE_BPM_TOLERANCE[config.GENRES[i]] for i in range(len(config.GENRES))]
        )
        self.adjacency = np.array(
            [[adjacency.loc[a, b] for b in config.GENRES] for a in config.GENRES],
            dtype=float,
        )
        self.log_ratios = np.log(np.array(config.TEMPO_RATIOS))
        self.n = n

    # -----------------------------------------------------------
    def tempo_scores(self, a: int, tol_mult: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """(scores in [0,1], allowed mask) for track a -> every track."""
        # delta(b, r) = |log(bpm_a * r) - log(bpm_b)|, minimized over ratios
        deltas = np.abs(
            (self.log_bpm[a] + self.log_ratios)[:, None] - self.log_bpm[None, :]
        )
        best = deltas.argmin(axis=0)
        pct = deltas[best, np.arange(self.n)]

        tol = 0.5 * (self.genre_tol[self.genre_idx[a]] + self.genre_tol[self.genre_idx])
        tol = tol * tol_mult
        scores = np.exp(-((pct / tol) ** 2))
        scores = np.where(best != 0, scores * config.HALFTIME_BLEND_PENALTY, scores)
        allowed = pct <= config.TEMPO_HARD_CAP_MULT * tol
        return scores, allowed

    def score_candidates(
        self,
        a: int,
        target_energy: float,
        tol_mult: float = 1.0,
        harmonic_mult: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Composite transition score from track a to every track.

        Returns (scores, allowed). Harmonic clashes are soft; only the tempo
        hard cap forbids a pair (relaxation ladder widens tol_mult and can
        zero out harmonic_mult).
        """
        tempo, allowed = self.tempo_scores(a, tol_mult)
        harm = camelot.MATRIX[self.camelot_idx[a], self.camelot_idx]
        energy = np.exp(-(((self.energy - target_energy) / config.ENERGY_SIGMA) ** 2))
        genre = np.exp(-self.adjacency[self.genre_idx[a], self.genre_idx] / config.GENRE_DIST_SCALE)

        crossing = self.genre_idx != self.genre_idx[a]
        bridge_out = self.is_bridge[a] & (self.bridge_to_idx[a] == self.genre_idx)
        bridge_in = self.is_bridge & (self.bridge_to_idx == self.genre_idx[a])
        bonus = config.BRIDGE_BONUS * (crossing & (bridge_out | bridge_in))

        scores = (
            config.W_HARMONIC * harmonic_mult * harm
            + config.W_TEMPO * tempo
            + config.W_ENERGY * energy
            + config.W_GENRE * genre
            + bonus
        )
        return scores, allowed

    # -----------------------------------------------------------
    def pair_components(self, a: int, b: int, target_energy: float) -> dict:
        """Score breakdown for a chosen transition (for explanations/report)."""
        deltas = np.abs(self.log_bpm[a] + self.log_ratios - self.log_bpm[b])
        ratio_i = int(deltas.argmin())
        pct = float(deltas[ratio_i])
        tol = 0.5 * float(
            self.genre_tol[self.genre_idx[a]] + self.genre_tol[self.genre_idx[b]]
        )
        tempo = float(np.exp(-((pct / tol) ** 2)))
        if ratio_i != 0:
            tempo *= config.HALFTIME_BLEND_PENALTY

        harm = float(camelot.MATRIX[self.camelot_idx[a], self.camelot_idx[b]])
        energy = float(
            np.exp(-(((self.energy[b] - target_energy) / config.ENERGY_SIGMA) ** 2))
        )
        genre = float(
            np.exp(-self.adjacency[self.genre_idx[a], self.genre_idx[b]] / config.GENRE_DIST_SCALE)
        )
        crossing = self.genre_idx[a] != self.genre_idx[b]
        via_bridge = bool(
            crossing
            and (
                (self.is_bridge[a] and self.bridge_to_idx[a] == self.genre_idx[b])
                or (self.is_bridge[b] and self.bridge_to_idx[b] == self.genre_idx[a])
            )
        )
        composite = (
            config.W_HARMONIC * harm
            + config.W_TEMPO * tempo
            + config.W_ENERGY * energy
            + config.W_GENRE * genre
            + (config.BRIDGE_BONUS if via_bridge else 0.0)
        )
        return {
            "harmonic": harm,
            "tempo": tempo,
            "energy": energy,
            "genre": genre,
            "bridge": via_bridge,
            "composite": composite,
            "bpm_ratio": config.TEMPO_RATIOS[ratio_i],
            "bpm_delta_pct": (np.exp(pct) - 1) * 100,
            "crossing": bool(crossing),
        }
