# Plan: AI DJ Set Builder Agent

## Context

The `genre-crossover` project established a solid analytical foundation: 250 tracks across 5 genres with normalized audio features (BPM, energy, Camelot harmonic keys, halftime correction, bridge track identification, genre adjacency matrix). The next step is to use this as the data layer for an AI agent that sequences those tracks — and any expanded library — into coherent 4+ hour DJ sets with a defined energy narrative.

---

## Requirements

### Functional Requirements

1. **Track Library Management**
   - Ingest track metadata + audio features (BPM, energy, key/Camelot, danceability, valence, genre)
   - Normalize BPM via halftime correction (already built)
   - Compute pairwise transition scores for all track pairs

2. **Transition Scoring Engine**
   - Harmonic compatibility: Camelot wheel adjacency (±1 key, relative major/minor, same key)
   - Tempo compatibility: BPM delta within ±6% (already defined), weighted by genre norms
   - Energy delta: penalize large energy jumps unless intentional (drop/build moments)
   - Genre adjacency: use existing genre distance matrix to cost inter-genre jumps

3. **Set Arc Model**
   - Define energy curve templates: warmup (low → mid), peak hour (sustained high), closing (high → low), full arc (low → peak → close)
   - Parameterize arc by duration and target intensity profile
   - Divide set into time segments (e.g., 30-min windows), assign energy/genre targets per window

4. **Sequencing Engine (Graph Pathfinding)**
   - Model tracks as nodes, transition scores as weighted directed edges
   - Use constrained pathfinding (Dijkstra / beam search) to find optimal track sequence that follows the arc
   - Hard constraints: no key clashes, BPM within tolerance
   - Soft constraints: energy progression, genre diversity, no artist repetition within N tracks

5. **LLM Agent Layer**
   - Accept natural-language vibe instructions: "dark warehouse peak hour techno set", "sunrise melodic house closing"
   - Map vibe → arc template + genre weights + energy floor/ceiling
   - Tool use: call sequencer, inspect proposed set, iterate/swap tracks
   - Explain transition rationale ("bridge track X connects house → techno via shared 128 BPM pocket")

6. **Output Formats**
   - Timestamped tracklist (track, artist, genre, BPM, key, start time)
   - Transition notes per track pair
   - Rekordbox XML export (for actual CDJ/mixer use)
   - Optional: energy curve visualization

### Non-Functional Requirements
- Library of 500–1,000 tracks minimum for meaningful 4-hour set variety (current: 250)
- Sequencer should run in <10 seconds for a 60-track set selection
- LLM agent should handle ambiguous vibe inputs gracefully

---

## Architecture

```
[Track Library / CSV]
        ↓
[Feature Normalizer]  ← halftime correction, Camelot key mapping (existing)
        ↓
[Transition Scorer]   ← pairwise BPM + harmonic + energy + genre delta matrix
        ↓
[Arc Planner]         ← energy curve template → per-segment constraints
        ↓
[Graph Sequencer]     ← beam search over transition graph
        ↓
[LLM Agent]           ← interprets vibe, calls sequencer tools, refines output
        ↓
[Output Formatter]    ← tracklist, Rekordbox XML, visualization
```

---

## Phased Implementation Plan

### Phase 1 — Data & Transition Layer (1–2 weeks)
**Goal:** Expand track library and build pairwise transition scoring.

- Expand track library to 500–1,000 tracks by pulling more from HuggingFace dataset (85K row file already exists at `/data/spotify_dataset.csv`)
- Add missing features: `danceability`, `valence`, `acousticness` from Spotify data
- Build `TransitionScorer`: for each track pair, compute composite score (harmonic weight 0.4, tempo weight 0.3, energy weight 0.3)
- Store scores as sparse matrix (only score pairs within BPM tolerance to keep it tractable)

**Files to build:**
- `src/transition_scorer.py` — pairwise scoring
- `src/camelot.py` — Camelot wheel adjacency lookup (extend existing key mapping)
- `data/tracks_expanded.csv` — expanded library

### Phase 2 — Set Arc Model + Sequencer (2–3 weeks)
**Goal:** Produce a valid track sequence from a vibe specification.

- Define arc templates as piecewise linear energy curves in `config/arc_templates.yaml`
- Build `ArcPlanner`: divides set duration into segments, assigns energy target + genre weight per segment
- Build `GraphSequencer`: beam search over transition graph respecting arc constraints
  - Start node: any track matching opening constraints
  - At each step: rank candidate next tracks by (transition score × arc conformance)
  - Backtrack if path hits a dead end

**Files to build:**
- `src/arc_planner.py`
- `src/graph_sequencer.py`
- `config/arc_templates.yaml`
- `output/set_tracklist.csv` / `output/set_tracklist.json`

### Phase 3 — LLM Agent Layer (2–3 weeks)
**Goal:** Natural language → DJ set via tool-using agent.

- Use Claude API with tool use
- Define tools:
  - `generate_set(duration_mins, arc_template, genre_weights, energy_floor, energy_ceil)` → calls sequencer
  - `swap_track(position, constraints)` → replaces one track in existing set
  - `explain_transition(track_a, track_b)` → returns human-readable rationale
  - `preview_arc(set)` → returns energy curve data
- Prompt: system prompt encodes DJ mixing rules (Camelot wheel, energy flow norms, genre mixing etiquette)
- Vibe → parameter mapping via few-shot examples in prompt

**Files to build:**
- `src/agent.py` — Claude API agent with tools
- `src/tools.py` — tool implementations
- `prompts/system.md` — DJ knowledge base prompt

### Phase 4 — Output & Integration (1 week)
**Goal:** Make output DJ-software ready.

- Rekordbox XML exporter (map tracklist → Rekordbox playlist schema)
- Energy curve visualization (extend existing matplotlib setup)
- CLI entrypoint: `python main.py --vibe "dark peak hour techno" --duration 240`

**Files to build:**
- `src/rekordbox_exporter.py`
- `src/visualizer.py` (extend existing)
- `main.py` — CLI entrypoint

### Phase 5 — Testing & Refinement (1–2 weeks)
- Generate 5–10 test sets, manually evaluate transition quality
- Tune transition score weights based on real DJ feedback
- Edge cases: track library too small for arc, no valid harmonic path, genre monoculture

---

## Key Considerations

### Data
- **Library size is the binding constraint.** 250 tracks across 5 genres = limited path diversity. The 85K-row `/data/spotify_dataset.csv` is already available; mining it for 1,000 curated tracks is the fastest unlock.
- Spotify metadata has no audio preview. The agent sequences tracks but cannot actually play them — output is a tracklist for a human DJ or Rekordbox import.

### Transition Logic Nuance
- Camelot harmonic mixing is a rule of thumb, not gospel. The agent should treat key clashes as a soft penalty, not a hard block — some of the best DJ transitions intentionally violate harmonic rules.
- BPM tolerance varies by genre: techno DJs mix within ±2 BPM; hip hop DJs may jump 10+ BPM. Weight tolerance by genre context.

### Energy Arc Realism
- A 4-hour set has a dramatically different arc than a 1-hour set. Segment granularity matters: 30-min windows for a 4-hour set give 8 segments, each with different energy/genre targets.
- "Peak hour" is not just high energy — it's sustained energy with minimal genre jumping. Build this into arc templates explicitly.

### LLM Agent Grounding
- The LLM must be grounded in actual track data, not hallucinate track names. Pass track library as context or give the agent a `search_tracks(query)` tool that queries real data.
- Agent should explain why it chose each transition — this builds trust and lets the DJ override intelligently.

### Legal / Licensing
- The agent produces a tracklist, not audio. No playback rights issues at this layer.
- Rekordbox XML import requires the DJ to already own the tracks locally. This is the correct architecture.

---

## Timeline Summary

| Phase | What | Duration |
|-------|------|----------|
| 1 | Data expansion + transition scoring | 1–2 weeks |
| 2 | Arc model + graph sequencer | 2–3 weeks |
| 3 | LLM agent layer | 2–3 weeks |
| 4 | Output formats + CLI | 1 week |
| 5 | Testing + tuning | 1–2 weeks |
| **Total** | **Working prototype** | **7–11 weeks** |

A minimal end-to-end prototype (CLI in, tracklist out, no LLM agent) could be ready in **3–4 weeks** using just Phases 1 + 2.

---

## Reuse from Existing Project

| Existing artifact | Reuse in agent |
|---|---|
| `src/features.py` — halftime correction, Camelot mapping | TransitionScorer, Feature Normalizer |
| `src/analysis.py` — genre centroids, bridge track logic | ArcPlanner genre weights, GraphSequencer genre cost |
| `data/tracks.csv` — 250 processed tracks with features | Seed library for Phase 1 expansion |
| `output/genre_adjacency.csv` — pairwise genre distances | Genre transition cost matrix in sequencer |
| `src/viz.py` — scatter plot renderer | Energy arc visualization in Phase 4 |

---

## Verification

1. **Unit:** `transition_scorer.py` — assert harmonic adjacency scores match known Camelot rules
2. **Integration:** Generate a 60-track set from the 250-track library; verify no BPM jumps >10%, no key clashes
3. **End-to-end:** Run `python main.py --vibe "peak hour techno" --duration 240`; inspect output tracklist manually for coherence
4. **Agent:** Prompt agent with "build me a sunrise closing set", verify it selects a downward-energy arc and favors melodic tracks
