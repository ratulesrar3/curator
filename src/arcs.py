"""
Set arc model: continuous energy curves with per-phase genre weights,
plus a deterministic vibe parser (keyword table) that maps free-text
descriptions to a template + genre lean. The parser doubles as the
no-credential fallback for the future LLM agent layer.
"""

import re
from dataclasses import dataclass, field

from . import config

EVEN = {g: 1.0 for g in config.GENRES}


def _weights(**overrides) -> dict[str, float]:
    w = dict(EVEN)
    w.update(overrides)
    return w


@dataclass(frozen=True)
class ArcPhase:
    name: str
    t0: float            # phase start, fraction of set [0, 1)
    t1: float            # phase end
    e0: float            # target energy at t0
    e1: float            # target energy at t1
    genre_weights: dict[str, float] = field(default_factory=lambda: dict(EVEN))

    def energy_at(self, t: float) -> float:
        span = max(self.t1 - self.t0, 1e-9)
        frac = (t - self.t0) / span
        return self.e0 + (self.e1 - self.e0) * frac


@dataclass(frozen=True)
class ArcTemplate:
    name: str
    description: str
    phases: tuple[ArcPhase, ...]

    def phase_at(self, t: float) -> ArcPhase:
        t = min(max(t, 0.0), 1.0)
        for p in self.phases:
            if t < p.t1:
                return p
        return self.phases[-1]

    def target(self, t: float) -> tuple[float, dict[str, float]]:
        """(target energy, genre weights) at set-position t in [0, 1]."""
        t = min(max(t, 0.0), 1.0)
        p = self.phase_at(t)
        return p.energy_at(t), p.genre_weights


TEMPLATES: dict[str, ArcTemplate] = {
    "full_journey": ArcTemplate(
        "full_journey",
        "open low, build, sustain a peak, land — the default long-set shape",
        (
            ArcPhase("opening", 0.00, 0.20, 0.35, 0.55,
                     _weights(**{"house": 1.4, "hip hop": 1.2, "k-pop": 1.1})),
            ArcPhase("build", 0.20, 0.45, 0.55, 0.75,
                     _weights(house=1.4, techno=1.2, **{"k-pop": 1.1})),
            ArcPhase("peak", 0.45, 0.75, 0.78, 0.90,
                     _weights(techno=1.5, house=1.2, **{"drum-and-bass": 1.2})),
            ArcPhase("land", 0.75, 1.00, 0.85, 0.50,
                     _weights(house=1.4, **{"hip hop": 1.1})),
        ),
    ),
    "warmup": ArcTemplate(
        "warmup",
        "low-to-mid opening room energy, no peaks",
        (
            ArcPhase("ease_in", 0.00, 0.50, 0.35, 0.48, _weights(house=1.4, **{"hip hop": 1.2})),
            ArcPhase("simmer", 0.50, 1.00, 0.48, 0.62, _weights(house=1.5, techno=1.1)),
        ),
    ),
    "peak_time": ArcTemplate(
        "peak_time",
        "sustained high energy with minimal genre jumping",
        (
            ArcPhase("lock_in", 0.00, 0.15, 0.72, 0.82, _weights(techno=1.6, house=1.2)),
            ArcPhase("peak", 0.15, 0.85, 0.82, 0.90, _weights(techno=1.8, house=1.1, **{"drum-and-bass": 1.1})),
            ArcPhase("hold", 0.85, 1.00, 0.90, 0.84, _weights(techno=1.6, house=1.2)),
        ),
    ),
    "closing": ArcTemplate(
        "closing",
        "come down from a peak into a melodic landing",
        (
            ArcPhase("descend", 0.00, 0.40, 0.80, 0.65, _weights(house=1.5, techno=1.2)),
            ArcPhase("drift", 0.40, 0.80, 0.65, 0.52, _weights(house=1.7, **{"k-pop": 1.1})),
            ArcPhase("landing", 0.80, 1.00, 0.52, 0.42, _weights(house=1.8, **{"hip hop": 1.1})),
        ),
    ),
    "wave": ArcTemplate(
        "wave",
        "two peaks with a breather between",
        (
            ArcPhase("build_one", 0.00, 0.20, 0.45, 0.75, _weights(house=1.4)),
            ArcPhase("peak_one", 0.20, 0.40, 0.78, 0.86, _weights(techno=1.5, house=1.2)),
            ArcPhase("breather", 0.40, 0.55, 0.80, 0.60, _weights(house=1.5, **{"hip hop": 1.2})),
            ArcPhase("build_two", 0.55, 0.70, 0.60, 0.80, _weights(techno=1.3, **{"drum-and-bass": 1.3})),
            ArcPhase("peak_two", 0.70, 0.90, 0.82, 0.90, _weights(**{"drum-and-bass": 1.5, "techno": 1.4})),
            ArcPhase("out", 0.90, 1.00, 0.85, 0.60, _weights(house=1.4)),
        ),
    ),
}

# Keyword -> (template vote, genre boosts). First template with the most
# votes wins; genre boosts multiply the template's phase weights.
VIBE_KEYWORDS: dict[str, tuple[str | None, dict[str, float]]] = {
    "warehouse":  ("peak_time", {"techno": 1.4}),
    "dark":       ("peak_time", {"techno": 1.3}),
    "peak":       ("peak_time", {}),
    "hard":       ("peak_time", {"techno": 1.3, "drum-and-bass": 1.2}),
    "rave":       ("peak_time", {"techno": 1.2, "drum-and-bass": 1.2}),
    "industrial": ("peak_time", {"techno": 1.4}),
    "sunrise":    ("closing", {"house": 1.3}),
    "closing":    ("closing", {}),
    "comedown":   ("closing", {"house": 1.2}),
    "afterhours": ("closing", {"house": 1.2, "techno": 1.1}),
    "melodic":    (None, {"house": 1.3, "k-pop": 1.1}),
    "warmup":     ("warmup", {}),
    "warm":       ("warmup", {}),
    "opening":    ("warmup", {}),
    "sunset":     ("warmup", {"house": 1.3}),
    "journey":    ("full_journey", {}),
    "open format": ("full_journey", {"hip hop": 1.3, "k-pop": 1.3}),
    "party":      ("full_journey", {"hip hop": 1.3, "k-pop": 1.2}),
    "wave":       ("wave", {}),
    "rollercoaster": ("wave", {}),
    "techno":     (None, {"techno": 1.5}),
    "house":      (None, {"house": 1.5}),
    "hip hop":    (None, {"hip hop": 1.6}),
    "hip-hop":    (None, {"hip hop": 1.6}),
    "rap":        (None, {"hip hop": 1.5}),
    "k-pop":      (None, {"k-pop": 1.6}),
    "kpop":       (None, {"k-pop": 1.6}),
    "drum and bass": (None, {"drum-and-bass": 1.6}),
    "drum-and-bass": (None, {"drum-and-bass": 1.6}),
    "dnb":        (None, {"drum-and-bass": 1.6}),
    "jungle":     (None, {"drum-and-bass": 1.5}),
    "bass":       (None, {"drum-and-bass": 1.3}),
}


def parse_vibe(text: str, default_template: str = "full_journey") -> tuple[ArcTemplate, dict[str, float]]:
    """Map a free-text vibe to (template, genre boost multipliers)."""
    low = text.lower()
    votes: dict[str, int] = {}
    boosts = {g: 1.0 for g in config.GENRES}
    for kw, (template, genre_boosts) in VIBE_KEYWORDS.items():
        # word-boundary match so e.g. "warehouse" doesn't trigger "house"
        if re.search(rf"\b{re.escape(kw)}\b", low):
            if template:
                votes[template] = votes.get(template, 0) + 1
            for g, b in genre_boosts.items():
                boosts[g] = max(boosts[g], b)
    name = max(votes, key=votes.get) if votes else default_template
    return TEMPLATES[name], boosts
