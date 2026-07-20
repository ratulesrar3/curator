"""
Central configuration for the curator sequencing engine.
Every tunable weight, tolerance, and palette lives here so tuning is one edit.
"""

# ---------------------------------------------------------------
# Genres
# ---------------------------------------------------------------
# Raw dataset labels -> curator genre buckets (mirrors genre-crossover)
GENRE_MAP = {
    "hip-hop":       "hip hop",
    "k-pop":         "k-pop",
    "house":         "house",
    "chicago-house": "house",
    "deep-house":    "house",
    "techno":        "techno",
    "drum-and-bass": "drum-and-bass",
}

GENRES = ["hip hop", "k-pop", "house", "techno", "drum-and-bass"]

# Genres Spotify frequently detects at half tempo
HALFTIME_GENRES = {"hip hop", "drum-and-bass"}

# ---------------------------------------------------------------
# Library cleaning
# ---------------------------------------------------------------
MIN_DURATION_MS = 90_000        # drop interludes / bad rows
MAX_DURATION_MS = 12 * 60_000   # drop DJ-mix compilations

# Bridge detection (ported thresholds from genre-crossover)
BRIDGE_DIST_MAX = 0.60          # z-space distance to foreign centroid
BRIDGE_TEMPO_TOL = 0.06         # pitch-fader reach

# ---------------------------------------------------------------
# Tempo compatibility
# ---------------------------------------------------------------
# Mixing tolerance as a fraction of BPM, by genre. Cross-genre pairs
# use the mean of the two genres' tolerances.
GENRE_BPM_TOLERANCE = {
    "techno":        0.04,
    "drum-and-bass": 0.05,
    "house":         0.06,
    "k-pop":         0.06,
    "hip hop":       0.10,
}
TEMPO_RATIOS = (1.0, 2.0, 0.5)   # straight, double-time, half-time blends
HALFTIME_BLEND_PENALTY = 0.9     # multiplier when the match is 2x / 0.5x
TEMPO_HARD_CAP_MULT = 2.0        # beyond cap * tolerance the pair is forbidden

# ---------------------------------------------------------------
# Camelot harmonic scoring
# ---------------------------------------------------------------
CAMELOT_SCORES = {
    "same":          1.00,  # 8A -> 8A
    "relative":      0.95,  # 8A -> 8B
    "neighbor":      0.85,  # 8A -> 7A / 9A
    "diagonal":      0.65,  # 8A -> 7B / 9B
    "energy_boost":  0.55,  # 8A -> 10A (+2, classic energy jump)
    "step3":         0.30,  # +/-3 on the wheel
    "clash":         0.10,  # everything further
}

# ---------------------------------------------------------------
# Transition composite weights
# ---------------------------------------------------------------
W_HARMONIC = 0.35
W_TEMPO = 0.30
W_ENERGY = 0.20
W_GENRE = 0.15
BRIDGE_BONUS = 0.10          # added when a genre crossing uses a bridge track
ENERGY_SIGMA = 0.15          # gaussian width for energy-vs-arc-target fit
GENRE_DIST_SCALE = 1.0       # exp(-dist/scale) for genre adjacency score

# ---------------------------------------------------------------
# Sequencer
# ---------------------------------------------------------------
BEAM_WIDTH = 48
EXPAND_PER_BEAM = 8          # candidates kept per beam before global prune
ARTIST_COOLDOWN = 10         # no artist repeat within N positions
PLAY_FRACTION = 0.72         # portion of a track actually played in the mix
PLAY_MIN_S = 150
PLAY_MAX_S = 390
SCORE_FLOOR = 0.35           # below this, try the relaxation ladder
TIE_JITTER = 0.01            # seeded noise so equal scores break reproducibly
# Relaxation ladder: (tempo tolerance multiplier, harmonic weight multiplier, label)
RELAX_LADDER = (
    (1.5, 1.0, "stretched tempo"),
    (1.5, 0.0, "key clash - echo out"),
    (2.5, 0.0, "hard cut / creative transition"),
)

# ---------------------------------------------------------------
# MSD adapter (see src/msd_adapter.py, docs/msd_feasibility.md)
# ---------------------------------------------------------------
MSD_TAG_VAL_MIN = 50.0   # Last.fm tag weight floor (val is 0-100)
MSD_KEY_CONF_MIN = 0.30  # drop tracks with less confident key detection

# ---------------------------------------------------------------
# Perplexity agent
# ---------------------------------------------------------------
# The Perplexity API has no native tool calling (verified against the
# official OpenAPI spec); the agent runs on json_schema structured outputs.
PPLX_URL = "https://api.perplexity.ai/chat/completions"
PPLX_MODEL = "sonar-pro"      # plan / critique / research
PPLX_TIMEOUT_S = 60
PPLX_RETRIES = 2              # extra attempts on 429/5xx
AGENT_MAX_REVISIONS = 2       # critique -> revise loop cap
AGENT_BOOST_MIN = 0.5         # whitelisted genre boost range the agent may set
AGENT_BOOST_MAX = 2.0
AGENT_MAX_TOKENS = 1024

# ---------------------------------------------------------------
# Spotify export (see src/spotify_export.py, docs/spotify_export.md)
# ---------------------------------------------------------------
# Auth is Authorization Code with PKCE (a public client - only a client ID,
# no secret). Since the 2025 OAuth migration Spotify rejects `localhost`, so
# the redirect must be a loopback IP literal.
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8888/callback"
SPOTIFY_SCOPE = "playlist-modify-private playlist-modify-public"
SPOTIFY_ADD_CHUNK = 100       # Spotify caps playlist-add at 100 tracks/call
SPOTIFY_SEARCH_LIMIT = 5      # candidates fetched per search (best hit wins)

# ---------------------------------------------------------------
# Palette
# ---------------------------------------------------------------
# Chart-safe variants of the crossbridge brand genre colors
# (#f97316/#ec4899/#eab308/#22d3ee/#4ade80): same OKLCH hues, lightness
# stepped inside the dark-surface band, grid-searched so every pair stays
# distinguishable under protan/deutan/tritan simulation (worst-case
# CIE76 dE 13.1, all pairs) with >=3:1 contrast on #0a0a0a.
GENRE_COLORS = {
    "hip hop":        "#b04d01",
    "k-pop":          "#ed499b",
    "house":          "#ad9002",
    "techno":         "#16a4ba",
    "drum-and-bass":  "#04934a",
}

UI = {
    "bg":             "#0a0a0a",
    "surface":        "#111111",
    "surface_raised": "#1a1a1a",
    "border":         "#222222",
    "border_active":  "#333333",
    "text":           "#f0ede8",
    "text_secondary": "#888888",
    "text_muted":     "#555555",
    "accent":         "#9f8fff",
    "score_high":     "#4ade80",
    "score_mid":      "#eab308",
    "score_low":      "#f87171",
}
