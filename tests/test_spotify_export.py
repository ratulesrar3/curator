import csv
import json

import pytest

from src import config, spotify_export as sx

ID22 = "6AtTHvvQzF9r02iZK9rH1K"   # 22-char base-62 Spotify id
ID22B = "7i08AhQcrdD4GLlr2Pmamg"
MSD = "TRMMMRX128F93187D9"        # Million Song Dataset id (18 chars)


def track(track_id=None, name="Song", artists="Artist"):
    return {"track_id": track_id, "track_name": name, "artists": artists}


def item(uri):
    return {"uri": uri}


class FakeSpotify:
    """Captures calls like tests/test_agent.py's FakeTransport. `search_results`
    is a flat list returned for any query, or a callable q -> items."""

    def __init__(self, search_results=None):
        self._search = search_results
        self.searches = []      # captured queries, in order
        self.added = []         # captured item lists per playlist_add_items call
        self.created = None     # captured create kwargs
        self.me_calls = 0

    def me(self):
        self.me_calls += 1
        return {"id": "u1"}

    def search(self, q, type="track", limit=5, market=None):
        self.searches.append(q)
        if callable(self._search):
            items = self._search(q)
        elif self._search:
            items = self._search
        else:
            items = []
        return {"tracks": {"items": items}}

    def current_user_playlist_create(self, name, public=True, collaborative=False,
                                     description=""):
        self.created = {"name": name, "public": public, "description": description}
        return {"id": "pl1", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}

    def playlist_add_items(self, playlist_id, items, position=None):
        self.added.append(list(items))


# ---------------------------------------------------------------- id shape
def test_is_spotify_id():
    assert sx.is_spotify_id(ID22)
    assert not sx.is_spotify_id(MSD)
    assert not sx.is_spotify_id("")
    assert not sx.is_spotify_id(None)
    assert not sx.is_spotify_id(ID22 + "x")   # 23
    assert not sx.is_spotify_id(ID22[:-1])    # 21


# ---------------------------------------------------------------- credentials
def test_resolve_client_id_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "envid123")
    assert sx.resolve_client_id(tmp_path / "none.md") == "envid123"


def test_resolve_client_id_from_secrets(monkeypatch, tmp_path):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    s = tmp_path / "secrets.md"
    s.write_text("# secrets\n\nSPOTIFY_CLIENT_ID=abc123def456ghi789jkl\n")
    assert sx.resolve_client_id(s) == "abc123def456ghi789jkl"


def test_resolve_client_id_none(monkeypatch, tmp_path):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    assert sx.resolve_client_id(tmp_path / "absent.md") is None


# ---------------------------------------------------------------- load_set
def test_load_set_json(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"tracklist": [track(ID22)], "report": {"vibe": "v", "seed": 42}}))
    tracks, report = sx.load_set(str(p))
    assert len(tracks) == 1 and report["vibe"] == "v"


def test_load_set_csv(tmp_path):
    p = tmp_path / "s.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "track_name", "artists"])
        w.writeheader()
        w.writerow(track(ID22))
    tracks, report = sx.load_set(str(p))
    assert tracks[0]["track_id"] == ID22 and report == {}


def test_load_set_bad_suffix(tmp_path):
    with pytest.raises(sx.SpotifyExportError):
        sx.load_set(str(tmp_path / "s.txt"))


# ---------------------------------------------------------------- helpers
def test_primary_artist():
    assert sx.primary_artist({"artists": "A;B"}) == "A"
    assert sx.primary_artist({"artists": "A, B"}) == "A"
    assert sx.primary_artist({"artists": "  Solo  "}) == "Solo"


def test_classify_splits_by_id_shape():
    direct, search = sx.classify([track(ID22), track(MSD), track(None)])
    assert len(direct) == 1 and len(search) == 2


# ---------------------------------------------------------------- resolve_uris
def test_resolve_uris_direct_never_searches():
    sp = FakeSpotify()
    uris, misses = sx.resolve_uris(sp, [track(ID22), track(ID22B)], "US")
    assert uris == [f"spotify:track:{ID22}", f"spotify:track:{ID22B}"]
    assert misses == []
    assert sp.searches == []          # zero API calls for direct IDs


def test_resolve_uris_search_preserves_order():
    found = "spotify:track:FOUNDVIASEARCH00000000"
    sp = FakeSpotify(search_results=[item(found)])
    tracks = [track(ID22), track(MSD, name="Mid"), track(ID22B)]   # search is the middle one
    uris, misses = sx.resolve_uris(sp, tracks, "US")
    assert uris == [f"spotify:track:{ID22}", found, f"spotify:track:{ID22B}"]
    assert misses == [] and len(sp.searches) == 1


def test_resolve_uris_miss_when_no_results():
    sp = FakeSpotify(search_results=[])
    uris, misses = sx.resolve_uris(sp, [track(ID22), track(MSD, name="Ghost")], "US")
    assert uris == [f"spotify:track:{ID22}"]
    assert len(misses) == 1 and misses[0]["track_name"] == "Ghost"


def test_search_uri_falls_back_to_loose_query():
    found = "spotify:track:LOOSEQUERYHIT000000000"
    sp = FakeSpotify(search_results=lambda q: [item(found)] if q.startswith("Song ") else [])
    uri = sx.search_uri(sp, track(MSD, name="Song", artists="Artist"), "US")
    assert uri == found
    assert sp.searches == ["track:Song artist:Artist", "Song Artist"]  # field then loose


# ---------------------------------------------------------------- create_playlist
def test_create_playlist_chunks_and_reports():
    tracks = [track(ID22) for _ in range(150)]
    sp = FakeSpotify()
    res = sx.create_playlist(sp, tracks, {"template": "peak_time", "duration": "2:00:00",
                                          "seed": 42}, name="My Set", public=False, market="US")
    assert sp.me_calls == 1
    assert sp.created["name"] == "My Set" and sp.created["public"] is False
    assert "peak_time" in sp.created["description"] and "seed 42" in sp.created["description"]
    assert [len(c) for c in sp.added] == [config.SPOTIFY_ADD_CHUNK, 50]
    assert res["added"] == 150 and res["total"] == 150 and res["misses"] == []
    assert res["url"].endswith("/playlist/pl1")


def test_create_playlist_raises_when_nothing_resolves():
    sp = FakeSpotify(search_results=[])
    with pytest.raises(sx.SpotifyExportError):
        sx.create_playlist(sp, [track(MSD, name="Ghost")], {}, "n", False, "US")


# ---------------------------------------------------------------- spotipy interface lock
def test_get_client_builds_offline():
    pytest.importorskip("spotipy")
    client = sx.get_client("dummyclientid")   # construct only; no token fetch, no browser
    for method in ("me", "search", "current_user_playlist_create", "playlist_add_items"):
        assert hasattr(client, method)
