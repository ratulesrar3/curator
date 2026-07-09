"""
Build the GitHub Pages gallery: sequence every set in SETS, then assemble
index.html from their JSON exports.

    python scripts/build_pages.py --out _site

Run from anywhere; paths resolve against the repo root. The out dir is
recreated from scratch. Set JSONs are build-side inputs only — the published
site is index.html plus one self-contained HTML report per set.

The agent set re-plans live when a Perplexity key is available
(PERPLEXITY_API_KEY env or gitignored secrets.md) and falls back to the
deterministic keyword parser otherwise — the build never fails for lack of a
key. The MSD set ships prebuilt from site/prebuilt/ because its research-only
source data (~900 MB) stays out of the repo; regenerate locally and re-commit
when the MSD library changes.
"""

import argparse
import html
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

REPO = "https://github.com/ratulesrar3/curator"
PREBUILT = ROOT / "site" / "prebuilt"
SEED = 42

# Cards render in this order. Templates are pinned so a future tweak to the
# vibe parser cannot silently change the published sets; the agent set stays
# unpinned — template choice is the agent's job.
SETS = [
    {"slug": "dark_warehouse_peak_hour_techno",
     "vibe": "dark warehouse peak hour techno", "hours": 4, "template": "peak_time"},
    {"slug": "sunrise_rooftop_closing_set_in_lisbon",
     "vibe": "sunrise rooftop closing set in Lisbon", "hours": 2, "agent": True},
    {"slug": "open_format_party",
     "vibe": "open format party", "hours": 4, "template": "full_journey"},
    {"slug": "festival_rollercoaster",
     "vibe": "festival rollercoaster", "hours": 4, "template": "wave"},
    {"slug": "golden_hour_sunset_warmup",
     "vibe": "golden hour sunset warmup", "hours": 2, "template": "warmup"},
    {"slug": "sunrise_melodic_closing",
     "vibe": "sunrise melodic closing", "hours": 2, "template": "closing"},
    {"slug": "warehouse_techno_peak", "static": True, "badge": "MSD library"},
]

esc = html.escape
ui = config.UI


def sequence_set(s: dict, out: Path) -> None:
    cmd = [sys.executable, "main.py", "--vibe", s["vibe"], "--hours", str(s["hours"]),
           "--seed", str(SEED), "--out", str(out), "--formats", "json,html"]
    if s.get("template"):
        cmd += ["--template", s["template"]]
    if s.get("agent"):
        cmd.append("--agent")
    print(f"== {s['slug']}" + (" (agent)" if s.get("agent") else ""), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not (out / f"{s['slug']}.json").exists():
        sys.exit(f"expected {s['slug']}.json from vibe {s['vibe']!r} — slug drift?")


def genre_strip(counts: dict) -> str:
    segs, chips = [], []
    for g in config.GENRES:
        n = counts.get(g, 0)
        if not n:
            continue
        segs.append(f'<div style="flex:{n};background:{config.GENRE_COLORS[g]}"></div>')
        chips.append(
            f'<span class="chip"><span class="dot" '
            f'style="background:{config.GENRE_COLORS[g]}"></span>{esc(g)} {n}</span>'
        )
    return (f'<div class="strip">{"".join(segs)}</div>'
            f'<div class="chips">{"".join(chips)}</div>')


def card(s: dict, payload: dict) -> str:
    r = payload["report"]
    agent = payload.get("agent")
    if agent and not agent.get("fallback"):
        badge = (f'<span class="badge accent">agent · {esc(agent["model"])} · '
                 f'{len(agent.get("citations", []))} sources</span>')
    elif s.get("badge"):
        badge = f'<span class="badge">{esc(s["badge"])}</span>'
    else:
        badge = ""
    return f"""
  <a class="card" href="{s["slug"]}.html">
    <div class="card-top">
      <span class="template">{esc(r["template"])}</span>{badge}
    </div>
    <h2>{esc(r["vibe"])}</h2>
    <div class="stats">{r["duration"]} · {r["n_tracks"]} tracks ·
      {r["harmonic_pct"]}% harmonic · {r["genre_crossings"]} crossings ·
      seed {r["seed"]}</div>
    {genre_strip(r["genre_counts"])}
  </a>"""


def index_doc(payloads: list[tuple[dict, dict]]) -> str:
    cards = "".join(card(s, p) for s, p in payloads)
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Long-form DJ sets sequenced from song feature analytics — energy arcs, harmonic mixing, genre bridges.">
<title>curator · set gallery</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Archivo:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:{ui["bg"]}; color:{ui["text"]}; font-family:Archivo,system-ui,sans-serif;
         padding:40px 24px 64px; max-width:1100px; margin:0 auto; }}
  .eyebrow {{ font-family:"IBM Plex Mono",monospace; font-size:12px; letter-spacing:.08em;
              color:{ui["accent"]}; text-transform:uppercase; }}
  h1 {{ font-size:30px; font-weight:700; margin:6px 0 8px; }}
  .sub {{ color:{ui["text_secondary"]}; font-size:14px; max-width:640px; line-height:1.5; }}
  .sub a {{ color:{ui["accent"]}; text-decoration:none; }}
  .sub a:hover {{ text-decoration:underline; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
           gap:14px; margin-top:28px; }}
  .card {{ display:block; background:{ui["surface"]}; border:1px solid {ui["border"]};
           border-radius:12px; padding:18px; text-decoration:none; color:{ui["text"]};
           transition:border-color .15s, transform .15s; }}
  .card:hover {{ border-color:{ui["border_active"]}; transform:translateY(-2px); }}
  .card-top {{ display:flex; align-items:center; gap:8px; margin-bottom:10px; }}
  .template {{ font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.06em;
               text-transform:uppercase; color:{ui["text_muted"]}; }}
  .badge {{ font-family:"IBM Plex Mono",monospace; font-size:10px; letter-spacing:.04em;
            padding:2px 8px; border-radius:99px; background:{ui["surface_raised"]};
            color:{ui["text_secondary"]}; border:1px solid {ui["border"]}; }}
  .badge.accent {{ color:{ui["accent"]}; border-color:{ui["accent"]}44; }}
  .card h2 {{ font-size:17px; font-weight:600; margin-bottom:8px; }}
  .stats {{ font-family:"IBM Plex Mono",monospace; font-size:11px; color:{ui["text_secondary"]};
            line-height:1.6; margin-bottom:14px; }}
  .strip {{ display:flex; gap:2px; height:6px; border-radius:3px; overflow:hidden;
            margin-bottom:10px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:10px; }}
  .chip {{ display:inline-flex; align-items:center; gap:5px;
           font-family:"IBM Plex Mono",monospace; font-size:10px; color:{ui["text_secondary"]}; }}
  .dot {{ width:8px; height:8px; border-radius:50%; }}
  footer {{ font-family:"IBM Plex Mono",monospace; font-size:11px; color:{ui["text_muted"]};
            margin-top:36px; }}
  footer a {{ color:{ui["text_secondary"]}; }}
</style></head><body>
<header>
  <div class="eyebrow">curator · set gallery</div>
  <h1>Long-form DJ sets from song feature analytics</h1>
  <p class="sub">Each report is a sequenced multi-hour set: an energy arc followed
    track-by-track, mixed in key on the Camelot wheel, with genre crossings routed
    through bridge tracks. Built by <a href="{REPO}">curator</a> — vibes in,
    tracklists out. One set is planned by a web-searching Perplexity agent; one is
    sequenced from a 15K-track Million Song Dataset supplement.</p>
</header>
<div class="grid">{cards}
</div>
<footer>generated {date.today().isoformat()} · <a href="{REPO}">github.com/ratulesrar3/curator</a></footer>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Pages gallery into --out.")
    ap.add_argument("--out", default="_site",
                    help="build dir, recreated from scratch (default: _site)")
    args = ap.parse_args()

    if not (ROOT / "data" / "library.csv").exists():
        sys.exit("data/library.csv missing — run: python main.py --rebuild-library")

    out = Path(args.out).resolve()
    if out == ROOT or (out / "main.py").exists():
        sys.exit(f"refusing to clobber {out} — not a build dir")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    for s in SETS:
        if s.get("static"):
            for ext in ("html", "json"):
                asset = PREBUILT / f"{s['slug']}.{ext}"
                if not asset.exists():
                    sys.exit(f"{asset} missing — rebuild the MSD library locally, "
                             "regenerate its report, and commit it to site/prebuilt/")
                shutil.copy(asset, out)
            print(f"== {s['slug']} (prebuilt)", flush=True)
        else:
            sequence_set(s, out)

    payloads = []
    for s in SETS:
        with open(out / f"{s['slug']}.json") as f:
            payloads.append((s, json.load(f)))
    (out / "index.html").write_text(index_doc(payloads))

    for j in out.glob("*.json"):
        j.unlink()
    print(f"built {out}: {len(payloads)} cards, "
          f"{sum(1 for _ in out.glob('*.html'))} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
