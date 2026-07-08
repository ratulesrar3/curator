import json
import xml.dom.minidom

import pandas as pd
import pytest

from src import arcs, exporters, report_html, viz
from src.explain import add_rationales, set_report
from src.sequencer import build_set


@pytest.fixture(scope="module")
def result(mini_library, mini_adjacency):
    return build_set(
        mini_library, mini_adjacency, arcs.TEMPLATES["warmup"],
        hours=0.5, vibe="export smoke", seed=5,
    )


@pytest.fixture(scope="module")
def written(result, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    return dict.fromkeys(
        exporters.export_all(
            result, str(out),
            {"csv", "json", "md", "m3u8", "rekordbox", "png", "html"},
            png_fn=viz.render_png, html_fn=report_html.render,
        )
    )


def _path(written, suffix):
    return next(p for p in written if p.endswith(suffix))


def test_all_formats_written(written):
    suffixes = (".csv", ".json", ".md", ".m3u8", ".rekordbox.xml", ".png", ".html")
    assert len(written) == len(suffixes)
    for s in suffixes:
        assert any(p.endswith(s) for p in written)


def test_csv_row_count_and_monotonic_timestamps(written, result):
    df = pd.read_csv(_path(written, ".csv"))
    assert len(df) == len(result.tracklist)
    assert df["start_s"].is_monotonic_increasing


def test_json_carries_report_and_tracklist(written, result):
    payload = json.load(open(_path(written, ".json")))
    assert payload["report"] == set_report(result)
    assert len(payload["tracklist"]) == len(result.tracklist)
    assert "rationale" in payload["tracklist"][1]


def test_m3u8_structure(written, result):
    lines = open(_path(written, ".m3u8")).read().splitlines()
    assert lines[0] == "#EXTM3U"
    assert sum(1 for l in lines if l.startswith("#EXTINF:")) == len(result.tracklist)


def test_rekordbox_xml_parses(written, result):
    doc = xml.dom.minidom.parse(_path(written, ".rekordbox.xml"))
    tracks = doc.getElementsByTagName("TRACK")
    # collection entries + playlist keys
    assert len(tracks) == 2 * len(result.tracklist)
    assert doc.getElementsByTagName("COLLECTION")[0].getAttribute("Entries") == str(
        len(result.tracklist)
    )


def test_rationales_are_nonempty(result):
    df = add_rationales(result)
    assert (df["rationale"].iloc[1:] != "").all()
    assert df["start_hms"].iloc[0] == "0:00:00"
