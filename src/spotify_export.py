"""
Turn a curator set export into a Spotify playlist.

A set from the default Spotify library already carries real Spotify track IDs,
so those map straight to `spotify:track:` URIs with no search. Sets from the
MSD supplement carry Million Song Dataset IDs (TR...) and are resolved by
searching Spotify for artist + title (best-effort: the top hit).

Auth is Authorization Code with PKCE - a public client, so only a client ID is
needed, no secret. The client ID is resolved from SPOTIFY_CLIENT_ID (env) or a
labeled line in the gitignored secrets.md, mirroring pplx.resolve_api_key. The
OAuth token is cached in the gitignored .cache-spotify. See
docs/spotify_export.md for the developer.spotify.com setup.

The CLI wrapper is scripts/spotify_playlist.py; the logic lives here so it can
be unit-tested with a fake Spotify client (see tests/test_spotify_export.py).
"""

import csv
import json
import os
import re
from pathlib import Path

from . import config

ROOT = Path(__file__).resolve().parent.parent
SECRETS_PATH = ROOT / "secrets.md"
CACHE_PATH = str(ROOT / ".cache-spotify")
SPOTIFY_ID_RE = re.compile(r"^[0-9A-Za-z]{22}$")   # base-62 Spotify track id


class SpotifyExportError(RuntimeError):
    """Unrecoverable problem building the playlist (bad input, no credentials)."""


def resolve_client_id(secrets_path=SECRETS_PATH) -> str | None:
    """Env SPOTIFY_CLIENT_ID, else a labeled line in the gitignored secrets.md."""
    cid = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    if cid:
        return cid
    p = Path(secrets_path)
    if p.exists():
        m = re.search(r"SPOTIFY_CLIENT_ID\s*[=:]\s*([0-9A-Za-z]{20,})", p.read_text(), re.I)
        if m:
            return m.group(1)
    return None


def load_set(path: str) -> tuple[list[dict], dict]:
    """Read a curator set export; returns (tracklist rows, report metadata)."""
    p = Path(path)
    if p.suffix == ".json":
        payload = json.loads(p.read_text())
        return payload.get("tracklist", []), payload.get("report", {})
    if p.suffix == ".csv":
        with open(p, newline="") as f:
            return list(csv.DictReader(f)), {}
    raise SpotifyExportError(f"unsupported set file '{p.suffix}' - use .json or .csv")


def is_spotify_id(track_id) -> bool:
    return bool(track_id) and SPOTIFY_ID_RE.match(str(track_id)) is not None


def primary_artist(track: dict) -> str:
    return re.split(r"[;,]", str(track.get("artists", "")))[0].strip()


def classify(tracks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (resolvable by direct Spotify ID, needing live search)."""
    direct = [t for t in tracks if is_spotify_id(t.get("track_id"))]
    search = [t for t in tracks if not is_spotify_id(t.get("track_id"))]
    return direct, search


def search_uri(sp, track: dict, market: str) -> str | None:
    """Best-effort match for a non-Spotify-ID track: field query, then loose."""
    name, artist = str(track.get("track_name", "")).strip(), primary_artist(track)
    for q in (f"track:{name} artist:{artist}", f"{name} {artist}"):
        try:
            items = sp.search(q=q, type="track", limit=config.SPOTIFY_SEARCH_LIMIT,
                              market=market)["tracks"]["items"]
        except Exception:
            items = []
        if items:
            return items[0]["uri"]
    return None


def resolve_uris(sp, tracks: list[dict], market: str) -> tuple[list[str], list[dict]]:
    """Resolve every track to a URI in set order. Direct IDs never call `sp`;
    the rest go through search. Returns (uris in order, unresolved tracks)."""
    uris, misses = [], []
    for t in tracks:
        tid = t.get("track_id")
        uri = f"spotify:track:{tid}" if is_spotify_id(tid) else search_uri(sp, t, market)
        (uris.append(uri) if uri else misses.append(t))
    return uris, misses


def add_in_chunks(sp, playlist_id: str, uris: list[str]) -> None:
    for i in range(0, len(uris), config.SPOTIFY_ADD_CHUNK):
        sp.playlist_add_items(playlist_id, uris[i:i + config.SPOTIFY_ADD_CHUNK])


def get_client(client_id: str):
    """spotipy client on the PKCE flow. Lazy import so --dry-run and the unit
    tests never need spotipy installed."""
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyPKCE
        from spotipy.cache_handler import CacheFileHandler
    except ImportError as e:
        raise SpotifyExportError("spotipy is not installed - run: pip install spotipy") from e
    auth = SpotifyPKCE(client_id=client_id, redirect_uri=config.SPOTIFY_REDIRECT_URI,
                       scope=config.SPOTIFY_SCOPE, open_browser=True,
                       cache_handler=CacheFileHandler(CACHE_PATH))
    return spotipy.Spotify(auth_manager=auth)


def create_playlist(sp, tracks: list[dict], report: dict, name: str,
                    public: bool, market: str) -> dict:
    """Resolve the set and create the playlist in curator's running order.
    Returns {url, added, total, misses}."""
    sp.me()  # forces the OAuth consent up front and validates the token
    uris, misses = resolve_uris(sp, tracks, market)
    if not uris:
        raise SpotifyExportError("resolved 0 tracks - nothing to add")

    desc = (f"Sequenced by curator · {report.get('template', '?')}, "
            f"{report.get('duration', '?')}, seed {report.get('seed', '?')}")
    playlist = sp.current_user_playlist_create(name=name, public=public, description=desc)
    add_in_chunks(sp, playlist["id"], uris)
    return {
        "url": playlist["external_urls"]["spotify"],
        "added": len(uris),
        "total": len(tracks),
        "misses": misses,
    }
