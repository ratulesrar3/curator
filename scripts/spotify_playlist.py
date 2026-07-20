"""
Build a Spotify playlist from a curator set file.

    python scripts/spotify_playlist.py output/<set>.json --name "Sunday slow jams"
    python scripts/spotify_playlist.py output/<set>.json --dry-run   # no auth, no writes

A set exported from the default Spotify library already carries real Spotify
track IDs, so those map straight to `spotify:track:` URIs with no search. Sets
built from the MSD supplement carry MSD IDs (TR...) and are resolved by
searching Spotify for artist + title (best-effort). The playlist is the curator
running order; Spotify can't reproduce beatmatched transitions (turn on its
client-side Crossfade for a blend).

Auth is Authorization Code with PKCE - only a client ID, no secret. Register
`http://127.0.0.1:8888/callback` on your app at developer.spotify.com and set
SPOTIFY_CLIENT_ID (env, or a `SPOTIFY_CLIENT_ID=...` line in the gitignored
secrets.md). See docs/spotify_export.md. The logic lives in src/spotify_export.py.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import spotify_export as sx  # noqa: E402

PREVIEW = 8  # rows shown per group in the dry-run report


def _label(track: dict) -> str:
    return f"{track.get('artists', '?')} — {track.get('track_name', '?')}"


def print_dry_run(tracks: list[dict]) -> None:
    direct, search = sx.classify(tracks)
    print(f"tracks: {len(tracks)}  |  direct Spotify IDs: {len(direct)}  "
          f"|  need live search: {len(search)}")
    for t in direct[:PREVIEW]:
        print(f"  direct  spotify:track:{t['track_id']}  {_label(t)}")
    if len(direct) > PREVIEW:
        print(f"  … +{len(direct) - PREVIEW} more direct")
    for t in search[:PREVIEW]:
        print(f"  search  (id={t.get('track_id') or '—'})  {_label(t)}")
    if len(search) > PREVIEW:
        print(f"  … +{len(search) - PREVIEW} more needing search")
    chunks = (len(tracks) + sx.config.SPOTIFY_ADD_CHUNK - 1) // sx.config.SPOTIFY_ADD_CHUNK
    print(f"\ndry run: {len(direct)}/{len(tracks)} resolvable offline with zero API "
          f"calls; {len(search)} would be searched live. Adding the set would take "
          f"{chunks} request(s) of ≤{sx.config.SPOTIFY_ADD_CHUNK} tracks. "
          "No auth performed, nothing created.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a Spotify playlist from a curator set.")
    ap.add_argument("set_file", help="a curator set export (output/<set>.json or .csv)")
    ap.add_argument("--name", help="playlist name (default: the set's vibe)")
    ap.add_argument("--public", action="store_true", help="make the playlist public")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and report only — no auth, no playlist created")
    ap.add_argument("--market", default="US", help="market for search matching (default US)")
    args = ap.parse_args(argv)

    try:
        tracks, report = sx.load_set(args.set_file)
    except sx.SpotifyExportError as e:
        sys.exit(str(e))
    if not tracks:
        sys.exit("no tracks in set file")
    print(f"set: {args.set_file}  ({report.get('vibe', '?')})")

    if args.dry_run:
        print_dry_run(tracks)
        return 0

    client_id = sx.resolve_client_id()
    if not client_id:
        sys.exit("no SPOTIFY_CLIENT_ID (env var or secrets.md) — see docs/spotify_export.md")
    name = args.name or report.get("vibe") or Path(args.set_file).stem
    try:
        res = sx.create_playlist(sx.get_client(client_id), tracks, report,
                                 name, args.public, args.market)
    except sx.SpotifyExportError as e:
        sys.exit(str(e))

    print(f"created: {res['url']}")
    print(f"added {res['added']}/{res['total']} tracks"
          + (f"; {len(res['misses'])} not found on Spotify:" if res["misses"] else ""))
    for t in res["misses"]:
        print(f"  miss  {_label(t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
