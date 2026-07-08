# Can the Million Song Dataset be curator's base dataset?

**Verdict: no as a base, yes as a supplement.** MSD cannot replace the
Spotify library — its `energy` field is zero for every track, it has no
genre labels, and curator's k-pop lane effectively does not exist in a
catalog frozen at 2011. It *can* feed the sequencer through the adapter
built from this assessment (`src/msd_adapter.py`), which recovers
**15,326 usable tracks** — 2.6× the base library — for four of the five
genres. The Spotify library stays the default.

```
python main.py --rebuild-msd-library                     # build data/library_msd.csv
python main.py --library data/library_msd.csv --vibe "warehouse techno" --hours 2
```

## What MSD is

1,000,000 commercial tracks (1922–2011) with Echo Nest audio analysis,
distributed as ~280 GB of per-track HDF5 — impractical and unnecessary
here. The **summary file** (`msd_summary_file.h5`, ~301 MB) carries every
per-track scalar the sequencer needs. Hosting note: millionsongdataset.com
serves **plain HTTP only** (no TLS listener), so tooling that force-upgrades
to HTTPS fails; `curl` works.

## Why it is not a drop-in base

| curator needs | MSD reality |
|---|---|
| `energy` (scored in every transition + arc fit) | Field exists but is **0.0 for all tracks** — verified on the official example track (energy 0.0, danceability 0.0, while tempo/loudness/key are real) |
| `genre` (adjacency, boosts, arc weights) | **No genre labels.** The `metadata/genre` field is empty (0 non-empty in a 20K sample). tagtraum annotations exist but collapse house/techno/dnb into one "Electronic" bucket |
| k-pop lane | Catalog frozen 2011 → k-pop barely exists (13 tracks survive the pipeline) |
| `valence`, `danceability`, `acousticness`, `instrumentalness`, `popularity` | Absent or zero — but these are pass-through metadata in curator (only `library.py` KEEP_COLS touches them), so NaN is acceptable |
| `tempo`, `key`+`mode`, `duration`, artist/title | Present and real ✓ |

## The adaptation that makes it a usable supplement

1. **Genres from Last.fm tags** (`lastfm_tags.db`, 594 MB SQLite; 505,216
   MSD tracks carry tags). This is the only MSD companion with granular
   electronic subgenres. A curated map of ~40 tag aliases → the five
   curator genres, at tag weight ≥ 50, strongest tag wins, exact ties
   dropped as ambiguous.
2. **Energy proxy from loudness**, fit on the Spotify library (which has
   both columns): `energy = clip(0.0308·loudness + 0.9475, 0, 1)`,
   **R² = 0.342** on 7,000 in-domain rows. In this dance-heavy domain the
   loudness range is compressed, so this recovers a coarse signal only —
   fine for arc-following, too lossy to compare similar tracks.
3. **Quality filters**: playable tempo, DJ-set duration (90 s–12 min),
   key confidence ≥ 0.30 (Echo Nest key detection is the weakest input —
   this single filter costs 41% of candidates), title+artist dedupe.

### Empirical funnel (2026-07-08 build)

| stage | tracks |
|---|---|
| MSD tracks with a mapped Last.fm tag | 31,802 |
| unambiguous winner genre | 31,119 |
| playable tempo/key & duration | 29,845 |
| key confidence ≥ 0.3 | 17,457 |
| deduped → **final** | **15,326** |

| genre | tracks |
|---|---|
| hip hop | 8,960 |
| house | 3,317 |
| techno | 1,884 |
| drum-and-bass | 1,152 |
| k-pop | 13 |

The full funnel regenerates into `output/msd_coverage.md` on every
`--rebuild-msd-library` run. A 2-hour peak-time set sequenced purely from
this library came out 100% harmonically compatible with arc RMSE 0.073.

## Standing limitations

- **Era**: nothing after 2011 — no modern techno/house/dnb, and the k-pop
  wave postdates the catalog entirely.
- **Tagger bias**: Last.fm users over-tag hip hop/rap (58.5% of the yield).
- **Energy is approximate** (R² 0.342) and Camelot keys inherit Echo Nest
  key-detection uncertainty even after the confidence filter.
- **Licensing**: MSD and the Last.fm companion data are research-only,
  non-commercial.
- The 13 k-pop tracks give that genre a noise centroid; the library
  machinery guards the degenerate cases (empty genres fall back to the
  global-mean centroid and are excluded as bridge targets).

## Sources

- Getting the dataset: http://millionsongdataset.com/pages/getting-dataset/
- Field list (FAQ): http://millionsongdataset.com/faq/
- Example track (all-zero energy/danceability): http://millionsongdataset.com/pages/example-track-description/
- Last.fm dataset: http://millionsongdataset.com/lastfm/
- tagtraum genre annotations (evaluated, too coarse): https://www.tagtraum.com/msd_genre_datasets.html
