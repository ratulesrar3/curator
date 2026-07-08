import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import config, library, msd_adapter


# ---------------------------------------------------------------- tag join
@pytest.fixture()
def tags_db(tmp_path):
    """Tiny lastfm_tags.db replica: tids / tags / tid_tag(val 0-100)."""
    path = tmp_path / "tags.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE tags (tag TEXT)")
    con.execute("CREATE TABLE tids (tid TEXT)")
    con.execute("CREATE TABLE tid_tag (tid INT, tag INT, val FLOAT)")
    tags = ["Drum n Bass", "House", "techno", "seen live", "Hip-Hop"]
    tids = ["TRAAA", "TRBBB", "TRCCC", "TRDDD"]
    con.executemany("INSERT INTO tags VALUES (?)", [(t,) for t in tags])
    con.executemany("INSERT INTO tids VALUES (?)", [(t,) for t in tids])
    rows = [
        ("TRAAA", "Drum n Bass", 100.0),   # cased alias -> drum-and-bass
        ("TRAAA", "seen live", 100.0),     # unmapped tag ignored
        ("TRBBB", "House", 40.0),          # below the val floor -> dropped
        ("TRCCC", "House", 80.0),
        ("TRCCC", "techno", 60.0),         # weaker than house -> house wins
        ("TRDDD", "Hip-Hop", 90.0),
    ]
    for tid, tag, val in rows:
        con.execute(
            "INSERT INTO tid_tag SELECT tids.ROWID, tags.ROWID, ? "
            "FROM tids, tags WHERE tids.tid = ? AND tags.tag = ?",
            (val, tid, tag),
        )
    con.commit()
    con.close()
    return str(path)


def test_tag_rows_case_insensitive_and_val_floor(tags_db):
    rows = msd_adapter.tag_rows(tags_db)
    assert set(rows["track_id"]) == {"TRAAA", "TRCCC", "TRDDD"}
    assert "seen live" not in set(rows["tag"])
    assert (rows["val"] >= config.MSD_TAG_VAL_MIN).all()


def test_resolve_genres_winner_and_tie_drop(tags_db):
    rows = msd_adapter.tag_rows(tags_db)
    tie = pd.DataFrame(
        [{"track_id": "TRTIE", "tag": "house", "val": 80.0},
         {"track_id": "TRTIE", "tag": "techno", "val": 80.0}]
    )
    resolved = msd_adapter.resolve_genres(pd.concat([rows, tie], ignore_index=True))
    got = dict(zip(resolved["track_id"], resolved["genre"]))
    assert got == {"TRAAA": "drum-and-bass", "TRCCC": "house", "TRDDD": "hip hop"}
    assert "TRTIE" not in got  # exact tie across genres is ambiguous


def test_tag_map_targets_are_valid_genres():
    assert set(msd_adapter.TAG_MAP.values()) == set(config.GENRES)
    assert all(t == t.lower() for t in msd_adapter.TAG_MAP)


# ---------------------------------------------------------------- energy proxy
def test_energy_proxy_recovers_linear_fit(tmp_path):
    loud = np.linspace(-30, 0, 200)
    df = pd.DataFrame({"loudness": loud, "energy": 0.025 * loud + 0.9})
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    proxy = msd_adapter.fit_energy_proxy(str(path))
    assert proxy["slope"] == pytest.approx(0.025, abs=1e-9)
    assert proxy["intercept"] == pytest.approx(0.9, abs=1e-9)
    assert proxy["r2"] == pytest.approx(1.0, abs=1e-9)


def test_energy_proxy_application_monotone_and_clipped():
    proxy = {"slope": 0.03, "intercept": 1.0}
    loud = np.array([-60.0, -20.0, -5.0, 5.0])
    e = msd_adapter.apply_energy_proxy(loud, proxy)
    assert (np.diff(e) >= 0).all()
    assert e.min() >= 0.0 and e.max() <= 1.0
    assert e[0] == 0.0 and e[-1] == 1.0  # clipping engaged at both ends


# ---------------------------------------------------------------- assembly
def _feats():
    rows = [
        # track_id, name, artist, tempo, key, key_conf, mode, loud, dur_s, genre
        ("TR1", "Alpha", "A1", 126.0, 9, 0.9, 0, -6.0, 240.0, "house"),
        ("TR2", "Beta", "A2", 128.0, 4, 0.8, 1, -5.0, 300.0, "techno"),
        ("TR3", "Gamma", "A3", 174.0, 7, 0.7, 0, -4.0, 260.0, "drum-and-bass"),
        ("TR4", "Delta", "A4", 0.0, 2, 0.9, 1, -7.0, 250.0, "house"),      # bad tempo
        ("TR5", "Epsilon", "A5", 122.0, 3, 0.05, 1, -8.0, 230.0, "house"),  # low key conf
        ("TR6", "Zeta", "A6", 124.0, 5, 0.9, 0, -9.0, 30.0, "techno"),      # too short
        ("TR7", "Alpha", "a1 ", 127.0, 6, 0.9, 1, -6.5, 245.0, "house"),    # dup of TR1 after lower+strip
        ("TR8", "Alpha", "A1", 129.0, 8, 0.9, 1, -6.2, 250.0, "house"),     # exact dup of TR1
    ]
    return pd.DataFrame(
        rows,
        columns=["track_id", "track_name", "artists", "tempo", "key",
                 "key_confidence", "mode", "loudness", "duration", "genre"],
    )


def test_assemble_schema_filters_and_passthrough_nans():
    proxy = {"slope": 0.03, "intercept": 0.95, "r2": 0.6, "n": 100}
    df, adjacency, funnel = msd_adapter.assemble(_feats(), proxy)

    assert list(df.columns) == library.KEEP_COLS
    # TR4 (tempo), TR5 (key conf), TR6 (duration), TR7+TR8 (dups of TR1) drop
    assert set(df["track_id"]) == {"TR1", "TR2", "TR3"}
    assert funnel["tagged & matched"] == 8

    assert df["camelot"].map(lambda c: c[-1] in "AB").all()
    assert df["valence"].isna().all() and df["popularity"].isna().all()
    assert (df["energy"].between(0, 1)).all()

    dnb = df[df["genre"] == "drum-and-bass"].iloc[0]
    assert not dnb["halftime"] and dnb["effective_bpm"] == dnb["bpm"]


def test_assemble_absent_genres_stay_finite_and_untargeted():
    proxy = {"slope": 0.03, "intercept": 0.95, "r2": 0.6, "n": 100}
    feats = _feats()
    feats = feats[feats["genre"].isin(["house", "techno"])]
    df, adjacency, _ = msd_adapter.assemble(feats, proxy)

    present = {"house", "techno"}
    assert set(df["genre"]) == present
    assert set(df["bridge_to"]) <= present          # never bridge to an empty genre
    assert np.isfinite(adjacency.to_numpy()).all()  # scorer needs finite distances
