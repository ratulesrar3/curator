"""
Set exporters: CSV, JSON, Markdown tracklist, M3U8, and a Rekordbox XML
playlist skeleton.

The Rekordbox export carries full metadata but placeholder Locations —
Rekordbox resolves tracks to local audio files, which this dataset does not
include. Import it as a naming/ordering reference, not a ready-to-play crate.
"""

import json
import os
from xml.etree import ElementTree as ET
from xml.dom import minidom

import pandas as pd

from .explain import add_rationales, hms, set_report
from .sequencer import SetResult

# C0 control codepoints outside the XML 1.0 Char production (everything below
# 0x20 except tab, newline, CR). Some source metadata — notably the Million
# Song Dataset, where an apostrophe is stored as 0x19 — carries these bytes;
# ElementTree serializes them but expat then refuses to re-parse, so strip them
# before they reach the Rekordbox XML.
_XML_ILLEGAL = {c: None for c in range(0x20) if c not in (0x09, 0x0A, 0x0D)}


def _xml_safe(text) -> str:
    return str(text).translate(_XML_ILLEGAL)


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "" for c in text)
    return "_".join(keep.lower().split())[:60] or "set"


def export_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def export_json(result: SetResult, df: pd.DataFrame, path: str) -> None:
    payload = {
        "report": set_report(result),
        "tracklist": df.to_dict(orient="records"),
    }
    if result.agent_notes:
        payload["agent"] = result.agent_notes
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def export_markdown(result: SetResult, df: pd.DataFrame, path: str) -> None:
    report = set_report(result)
    lines = [
        f"# {result.vibe or result.template.name}",
        "",
        f"{report['n_tracks']} tracks · {report['duration']} · "
        f"{report['harmonic_pct']}% harmonic · template `{report['template']}` "
        f"· seed {report['seed']}",
        "",
    ]
    notes = result.agent_notes
    if notes and not notes.get("fallback"):
        if notes.get("vibe_reading"):
            lines += [f"> {notes['vibe_reading']}", ">"]
        lines.append(f"> — agent ({notes['model']}), {notes['revisions']} revision(s)")
        lines += [f"> [{c['title']}]({c['url']})" for c in notes.get("citations", [])]
        lines.append("")
    lines += [
        "| # | start | artist — track | genre | BPM | key | energy | transition |",
        "|---|-------|----------------|-------|-----|-----|--------|------------|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['position']} | {r['start_hms']} | {r['artists']} — {r['track_name']} "
            f"| {r['genre']} | {r['bpm']} | {r['camelot']} | {r['energy']} "
            f"| {r['rationale']} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_m3u8(df: pd.DataFrame, path: str) -> None:
    lines = ["#EXTM3U"]
    for _, r in df.iterrows():
        secs = int(round(r["playtime_s"]))
        lines.append(f"#EXTINF:{secs},{r['artists']} - {r['track_name']}")
        lines.append(f"{_slug(r['artists'])}-{_slug(r['track_name'])}.mp3")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_rekordbox_xml(result: SetResult, df: pd.DataFrame, path: str) -> None:
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="curator", Version="0.1", Company="")
    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(df)))
    for _, r in df.iterrows():
        ET.SubElement(
            collection,
            "TRACK",
            TrackID=str(r["position"]),
            Name=_xml_safe(r["track_name"]),
            Artist=_xml_safe(r["artists"]),
            Genre=_xml_safe(r["genre"]),
            AverageBpm=f"{r['bpm']:.2f}",
            Tonality=str(r["camelot"]),
            TotalTime=str(int(round(r["playtime_s"]))),
            Location=f"file://localhost/REPLACE_WITH_LOCAL_PATH/{_slug(r['track_name'])}.mp3",
        )
    playlists = ET.SubElement(root, "PLAYLISTS")
    playlist_root = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="1")
    node = ET.SubElement(
        playlist_root,
        "NODE",
        Name=_xml_safe(result.vibe or result.template.name),
        Type="1",
        KeyType="0",
        Entries=str(len(df)),
    )
    for _, r in df.iterrows():
        ET.SubElement(node, "TRACK", Key=str(r["position"]))

    pretty = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(path, "w") as f:
        f.write(pretty)


def export_all(
    result: SetResult,
    out_dir: str,
    formats: set[str],
    png_fn=None,
    html_fn=None,
) -> list[str]:
    """Write requested formats; returns paths written. png_fn/html_fn are
    injected renderers (viz.render_png / report_html.render) to keep this
    module free of matplotlib imports."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, _slug(result.vibe or result.template.name))
    df = add_rationales(result)
    written = []

    if "csv" in formats:
        export_csv(df, base + ".csv")
        written.append(base + ".csv")
    if "json" in formats:
        export_json(result, df, base + ".json")
        written.append(base + ".json")
    if "md" in formats:
        export_markdown(result, df, base + ".md")
        written.append(base + ".md")
    if "m3u8" in formats:
        export_m3u8(df, base + ".m3u8")
        written.append(base + ".m3u8")
    if "rekordbox" in formats:
        export_rekordbox_xml(result, df, base + ".rekordbox.xml")
        written.append(base + ".rekordbox.xml")
    if "png" in formats and png_fn:
        png_fn(result, df, base + ".png")
        written.append(base + ".png")
    if "html" in formats and html_fn:
        html_fn(result, df, base + ".html")
        written.append(base + ".html")
    return written
