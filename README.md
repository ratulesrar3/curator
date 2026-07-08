# curator

**Long-form DJ sets from song feature analytics.** Give it a vibe and a duration; it sequences a 4+ hour tracklist that follows an energy arc, mixes in key, respects genre-specific BPM tolerance, and crosses genres through the bridge tracks that make those crossings work.

```
python main.py --vibe "dark warehouse peak hour techno" --hours 4
```

```
vibe:        dark warehouse peak hour techno
template:    peak_time  (seed 42)
tracks:      59  over 4:01:03
harmonic:    100.0% of transitions compatible
tempo:       mean |ΔBPM| 0.4%
arc fit:     RMSE 0.034 vs target curve
transitions: mean score 0.979, 0 genre crossings, 0 relaxed
```

Part of a three-project arc:

| project | role |
|---|---|
| [genre-crossover](../genre-crossover) | the data layer — maps where five genres share a tempo/energy pocket and which tracks bridge them |
| [crossbridge](../crossbridge) | pairwise scoring — a virtual mixer that scores one crossfade at a time |
| **curator** | full-set sequencing — chains thousands of those pairwise decisions into a coherent multi-hour set |

## How it works

1. **Library** (`src/library.py`) — 5,906 tracks across hip hop, k-pop, house, techno, and drum-and-bass, cleaned and deduped from the cached HuggingFace [spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) dump. Camelot keys from Spotify pitch class + mode; genre centroids, adjacency matrix, and bridge scores recomputed over the full library. An optional **Million Song Dataset supplement** (`src/msd_adapter.py`) adds 15,326 pre-2011 tracks by joining Last.fm genre tags with a loudness-derived energy proxy — why that adapter is needed (and why MSD can't be the base) is documented in [docs/msd_feasibility.md](docs/msd_feasibility.md).
2. **Transition scoring** (`src/transitions.py`) — every candidate next-track is scored against the whole library in vectorized numpy: harmonic (Camelot wheel, *soft* — the best DJs break key rules on purpose), tempo (ratio-aware, so 87 BPM hip hop legitimately blends with 174 BPM DnB; tolerance varies by genre), energy fit against the arc target, genre adjacency, and a bonus for crossing genres via a bridge track.
3. **Arc model** (`src/arcs.py`) — continuous energy curves (`full_journey`, `peak_time`, `closing`, `warmup`, `wave`) with per-phase genre weights. A keyword parser maps free-text vibes onto templates.
4. **Sequencer** (`src/sequencer.py`) — beam search over the transition graph. Hard constraints: no repeats, artist spacing, tempo cap. Dead ends trigger a relaxation ladder ("stretched tempo" → "echo out" → "hard cut") instead of failing, and those moments are labeled in the output.
5. **Receipts** (`src/explain.py`) — every transition gets a deterministic rationale derived from its actual score components, and every set gets quantitative metrics (harmonic %, mean |ΔBPM|, arc RMSE, crossings via bridges).

## Outputs

Each run writes to `output/`: a **self-contained HTML set report** (energy-arc chart with hover, timestamped tracklist with per-transition scores), an energy-arc **PNG**, plus **CSV / JSON / Markdown / M3U8 / Rekordbox XML**. The Rekordbox export carries metadata only — point its `Location` fields at your local files.

Chart colors are CVD-validated derivatives of the genre palette (all pairs distinguishable under protan/deutan/tritan simulation on the dark surface).

## Agent

`--agent` hands planning to a Perplexity-backed agent (`src/agent.py`). The Perplexity API has no native tool calling — verified against its OpenAPI spec — so this is a staged loop built on `json_schema` structured outputs: **research** the vibe with live web search (venues, scenes, events get looked up, not guessed) → **plan** template + genre boosts → **build** deterministically → **critique** against the set metrics (≤ 2 revisions). Every LLM reply is schema-constrained and every lever is clamped to a whitelist before it touches the sequencer; the research citations land in the HTML/JSON/Markdown reports.

```
python main.py --agent --vibe "sunrise rooftop closing set in Lisbon" --hours 2
```

The key is read from `PERPLEXITY_API_KEY` or a gitignored `secrets.md` — never hardcode or commit it. Without a key (or on any API failure) the deterministic keyword parser takes over and the run still produces a set.

## Run it

```bash
pip install -r requirements.txt
python main.py --rebuild-library                      # once; needs data/spotify_raw.csv
python main.py --vibe "sunrise melodic closing" --hours 2
python main.py --vibe "open format party" --hours 4 --template wave --seed 7
python main.py --agent --vibe "closing set at a Lisbon rooftop"    # Perplexity agent
python main.py --rebuild-msd-library                  # optional 15K-track MSD supplement
python main.py --library data/library_msd.csv --vibe "warehouse techno" --hours 2
pytest -q                                             # 48 tests
```

Same seed → same set, reproducibly. A 4-hour set sequences in about a second.

## Roadmap

- [x] **LLM agent layer** — built on Perplexity `sonar-pro` (see [Agent](#agent)); the keyword parser stays as the no-credential fallback
- [x] **Million Song Dataset supplement** — feasibility assessed ([docs/msd_feasibility.md](docs/msd_feasibility.md)) and adapter built: not viable as the base (all-zero energy, no genre labels, no k-pop), viable as a 15K-track supplement via Last.fm tags
- [ ] Score your actual crates: import a Rekordbox collection XML as the library
- [ ] Audio-preview links in the HTML report

---

*Part of my analytics portfolio. [More work here](https://github.com/ratulesrar3).*
