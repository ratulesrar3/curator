"""
Self-contained HTML set report in the crossbridge design language:
dark surfaces, monospace data readouts, genre-colored marks. Single file,
inline CSS/JS, no runtime data fetches (fonts come from the Google CDN with
system fallbacks, matching crossbridge).
"""

import html
from datetime import date

import pandas as pd

from . import config
from .explain import set_report
from .sequencer import SetResult

W, H = 1000, 300  # SVG plot area


def _esc(v) -> str:
    return html.escape(str(v))


def _score_class(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "mid"
    return "low"


def _svg_arc(result: SetResult, df: pd.DataFrame) -> str:
    ui = config.UI
    total_s = float(df["start_s"].iloc[-1] + df["playtime_s"].iloc[-1])
    x = lambda s: s / total_s * W
    y = lambda e: (1 - e) * H

    parts = [
        f'<svg viewBox="-46 -12 {W + 62} {H + 48}" role="img" '
        f'aria-label="Energy arc: target curve and per-track energy over set time">'
    ]

    # phase bands + names
    for p in result.template.phases:
        x0, x1 = p.t0 * W, p.t1 * W
        parts.append(
            f'<rect x="{x0:.0f}" y="0" width="{x1 - x0:.0f}" height="{H}" '
            f'fill="{ui["surface_raised"]}" opacity="0.35"/>'
        )
        parts.append(
            f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 6}" text-anchor="middle" '
            f'class="phase">{_esc(p.name.replace("_", " "))}</text>'
        )

    # gridlines + y labels
    for e in (0.25, 0.5, 0.75):
        parts.append(
            f'<line x1="0" y1="{y(e):.0f}" x2="{W}" y2="{y(e):.0f}" '
            f'stroke="{ui["border"]}" stroke-width="1"/>'
        )
        parts.append(f'<text x="-10" y="{y(e) + 4:.0f}" text-anchor="end" class="tick">{e}</text>')

    # x labels: hour marks
    hours = int(total_s // 3600) + 1
    for h in range(hours + 1):
        s = h * 3600
        if s > total_s:
            break
        parts.append(
            f'<text x="{x(s):.0f}" y="{H + 20}" text-anchor="middle" class="tick">{h}:00</text>'
        )

    # target curve
    pts = []
    for i in range(101):
        t = i / 100
        e, _ = result.template.target(t)
        pts.append(f"{t * W:.1f},{y(e):.1f}")
    parts.append(
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="{ui["accent"]}" '
        f'stroke-width="2" stroke-dasharray="5 4" opacity="0.9"/>'
    )

    # track dots (+ larger transparent hit targets for hover)
    for _, r in df.iterrows():
        cx, cy = x(r["start_s"]), y(r["energy"])
        color = config.GENRE_COLORS[r["genre"]]
        if r["is_bridge"]:
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8.5" fill="none" '
                f'stroke="{ui["text"]}" stroke-width="1"/>'
            )
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}" '
            f'stroke="{ui["bg"]}" stroke-width="1"/>'
        )
        tip = (
            f"{r['artists']} — {r['track_name']}||{r['genre']} · {r['bpm']} BPM · "
            f"{r['camelot']} · energy {r['energy']}||starts {r['start_hms']}"
        )
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="transparent" '
            f'class="hit" data-tip="{_esc(tip)}"/>'
        )

    parts.append(f'<text x="-34" y="{H / 2:.0f}" class="axis" transform="rotate(-90 -34 {H / 2:.0f})">energy</text>')
    parts.append("</svg>")
    return "".join(parts)


def _legend(df: pd.DataFrame) -> str:
    ui = config.UI
    chips = []
    for g in config.GENRES:
        if g not in set(df["genre"]):
            continue
        chips.append(
            f'<span class="chip"><span class="dot" style="background:{config.GENRE_COLORS[g]}"></span>{_esc(g)}</span>'
        )
    chips.append(
        f'<span class="chip"><span class="dot ring"></span>bridge track</span>'
    )
    chips.append(
        f'<span class="chip"><span class="dash" style="background:{ui["accent"]}"></span>target arc</span>'
    )
    return f'<div class="legend">{"".join(chips)}</div>'


def _tiles(report: dict) -> str:
    tiles = [
        ("duration", report["duration"]),
        ("tracks", report["n_tracks"]),
        ("harmonic", f"{report['harmonic_pct']}%"),
        ("mean |ΔBPM|", f"{report['mean_abs_bpm_delta_pct']}%"),
        ("arc RMSE", report["arc_rmse"]),
        ("crossings", f"{report['genre_crossings']} ({report['bridge_crossings']} bridge)"),
    ]
    cells = "".join(
        f'<div class="tile"><div class="tile-label">{_esc(k)}</div>'
        f'<div class="tile-value">{_esc(v)}</div></div>'
        for k, v in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _table(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        color = config.GENRE_COLORS[r["genre"]]
        if r["position"] == 1:
            score_cell = '<td class="score">—</td>'
        else:
            s = float(r["t_score"])
            score_cell = f'<td class="score"><span class="badge {_score_class(s)}">{s * 100:.0f}</span></td>'
        energy_pct = float(r["energy"]) * 100
        rows.append(
            "<tr>"
            f'<td class="num">{r["position"]}</td>'
            f'<td class="mono">{_esc(r["start_hms"])}</td>'
            f'<td><div class="track">{_esc(r["track_name"])}</div>'
            f'<div class="artist">{_esc(r["artists"])}</div></td>'
            f'<td><span class="chip"><span class="dot" style="background:{color}"></span>{_esc(r["genre"])}</span></td>'
            f'<td class="mono">{r["bpm"]}</td>'
            f'<td class="mono">{_esc(r["camelot"])}</td>'
            f'<td><div class="ebar"><div style="width:{energy_pct:.0f}%"></div></div></td>'
            f"{score_cell}"
            f'<td class="rationale">{_esc(r["rationale"])}</td>'
            "</tr>"
        )
    return (
        '<table><thead><tr><th>#</th><th>start</th><th>track</th><th>genre</th>'
        "<th>BPM</th><th>key</th><th>energy</th><th>score</th><th>transition</th>"
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _agent_section(result: SetResult) -> str:
    notes = result.agent_notes
    if not notes or notes.get("fallback"):
        return ""
    cites = "".join(
        f'<li><a href="{_esc(c["url"])}" rel="noopener">{_esc(c["title"])}</a></li>'
        for c in notes.get("citations", [])
    )
    reading = f"<p>{_esc(notes['vibe_reading'])}</p>" if notes.get("vibe_reading") else ""
    rationale = (f'<p class="agent-meta">plan: {_esc(notes["plan_rationale"])} · '
                 f'{notes["revisions"]} revision(s) · model {_esc(notes["model"])}</p>'
                 if notes.get("plan_rationale") else "")
    return (f'<section><h2>agent notes</h2>{reading}{rationale}'
            + (f'<ol class="cites">{cites}</ol>' if cites else "")
            + "</section>")


def render(result: SetResult, df: pd.DataFrame, path: str) -> None:
    ui = config.UI
    report = set_report(result)
    title = result.vibe or result.template.name

    doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} — curator set report</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Archivo:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:{ui["bg"]}; --surface:{ui["surface"]}; --raised:{ui["surface_raised"]};
    --border:{ui["border"]}; --text:{ui["text"]}; --text2:{ui["text_secondary"]};
    --muted:{ui["text_muted"]}; --accent:{ui["accent"]};
    --high:{ui["score_high"]}; --mid:{ui["score_mid"]}; --low:{ui["score_low"]};
  }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:Archivo,system-ui,sans-serif;
         padding:32px 24px 64px; max-width:1200px; margin:0 auto; }}
  .eyebrow {{ font-family:"IBM Plex Mono",monospace; font-size:12px; letter-spacing:.08em;
              color:var(--accent); text-transform:uppercase; }}
  h1 {{ font-size:28px; font-weight:700; margin:6px 0 4px; }}
  .meta {{ font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--text2); }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
            gap:10px; margin:24px 0; }}
  .tile {{ background:var(--surface); border:1px solid var(--border); border-radius:10px;
           padding:14px 16px; }}
  .tile-label {{ font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--text2);
                 letter-spacing:.06em; text-transform:uppercase; }}
  .tile-value {{ font-family:"IBM Plex Mono",monospace; font-size:20px; margin-top:6px; }}
  section {{ background:var(--surface); border:1px solid var(--border); border-radius:12px;
             padding:20px; margin-bottom:20px; }}
  h2 {{ font-family:"IBM Plex Mono",monospace; font-size:12px; letter-spacing:.08em;
        text-transform:uppercase; color:var(--text2); margin-bottom:14px; }}
  svg {{ width:100%; height:auto; display:block; }}
  .tick,.axis,.phase {{ font-family:"IBM Plex Mono",monospace; font-size:11px; fill:var(--muted); }}
  .legend {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:12px; }}
  .chip {{ display:inline-flex; align-items:center; gap:6px;
           font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--text2); }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  .dot.ring {{ background:none; border:1.5px solid var(--text); }}
  .dash {{ width:16px; height:2px; display:inline-block; }}
  .table-wrap {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th {{ font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.06em;
        text-transform:uppercase; color:var(--muted); text-align:left;
        padding:8px 10px; border-bottom:1px solid var(--border); }}
  td {{ padding:8px 10px; border-bottom:1px solid var(--border); vertical-align:middle; }}
  tr:hover td {{ background:var(--raised); }}
  .num {{ color:var(--muted); font-family:"IBM Plex Mono",monospace; }}
  .mono {{ font-family:"IBM Plex Mono",monospace; font-size:12px; }}
  .track {{ font-weight:600; }}
  .artist {{ color:var(--text2); font-size:12px; }}
  .ebar {{ width:70px; height:4px; background:var(--raised); border-radius:2px; }}
  .ebar div {{ height:4px; background:var(--accent); border-radius:2px; }}
  .badge {{ font-family:"IBM Plex Mono",monospace; font-size:12px; padding:2px 7px;
            border-radius:4px; background:var(--raised); }}
  .badge.high {{ color:var(--high); }} .badge.mid {{ color:var(--mid); }} .badge.low {{ color:var(--low); }}
  .rationale {{ font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--text2);
                max-width:360px; }}
  .agent-meta {{ font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--text2);
                 margin-top:8px; }}
  .cites {{ margin:10px 0 0 18px; font-size:12px; }}
  .cites li {{ margin-top:4px; }}
  .cites a {{ color:var(--accent); text-decoration:none; }}
  .cites a:hover {{ text-decoration:underline; }}
  #tip {{ position:fixed; display:none; background:var(--raised); border:1px solid var(--border);
          border-radius:8px; padding:10px 12px; font-size:12px; pointer-events:none;
          max-width:280px; z-index:10; }}
  #tip .t1 {{ font-weight:600; }} #tip .t2,#tip .t3 {{ font-family:"IBM Plex Mono",monospace;
          font-size:11px; color:var(--text2); margin-top:3px; }}
  footer {{ font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--muted);
            margin-top:8px; }}
</style></head><body>
<header>
  <div class="eyebrow">curator · set report</div>
  <h1>{_esc(title)}</h1>
  <div class="meta">template {_esc(report["template"])} · seed {report["seed"]} ·
    {report["artists_unique"]} artists · generated {date.today().isoformat()}</div>
</header>
{_tiles(report)}
{_agent_section(result)}
<section>
  <h2>energy arc</h2>
  {_svg_arc(result, df)}
  {_legend(df)}
</section>
<section>
  <h2>tracklist</h2>
  <div class="table-wrap">{_table(df)}</div>
</section>
<footer>transition scores: harmonic ·35 / tempo ·30 / energy ·20 / genre ·15 + bridge bonus.
Rekordbox export carries metadata only — point Locations at your local files.</footer>
<div id="tip"></div>
<script>
  const tip = document.getElementById("tip");
  document.querySelectorAll(".hit").forEach(el => {{
    el.addEventListener("mouseenter", () => {{
      const [t1, t2, t3] = el.dataset.tip.split("||");
      tip.innerHTML = `<div class="t1">${{t1}}</div><div class="t2">${{t2}}</div><div class="t3">${{t3}}</div>`;
      tip.style.display = "block";
    }});
    el.addEventListener("mousemove", e => {{
      const pad = 14, w = tip.offsetWidth;
      tip.style.left = (e.clientX + pad + w > innerWidth ? e.clientX - w - pad : e.clientX + pad) + "px";
      tip.style.top = (e.clientY + 12) + "px";
    }});
    el.addEventListener("mouseleave", () => tip.style.display = "none");
  }});
</script>
</body></html>"""
    with open(path, "w") as f:
        f.write(doc)
