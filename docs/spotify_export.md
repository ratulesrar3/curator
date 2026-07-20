# Turning a curator set into a Spotify playlist

**Spike verdict: it works, and for sets built on the default library it's
lossless.** Those sets already carry real Spotify track IDs, so every track
maps straight to a `spotify:track:` URI with no matching guesswork. Sets built
from the MSD supplement carry Million Song Dataset IDs instead and fall back to
search-by-artist/title — best-effort, lower fidelity. What a playlist *can't*
capture is the mix itself: Spotify plays the tracks in curator's running order
but can't beatmatch the transitions (turn on Spotify's client-side **Crossfade**
for a blend).

```bash
python scripts/spotify_playlist.py output/dark_warehouse_peak_hour_techno.json --dry-run
python scripts/spotify_playlist.py output/dark_warehouse_peak_hour_techno.json --name "Warehouse"
```

## How resolution works

A curator set export now includes each track's `track_id` (threaded through
`src/sequencer.py`). The script (`scripts/spotify_playlist.py`) classifies each:

- **22-char base-62 ID** → a Spotify ID → `spotify:track:{id}` directly, **zero
  API calls**. This is every track from the default Spotify library.
- **Anything else** (an MSD `TR…` ID, or blank) → `search` for `track:<name>
  artist:<artist>`, take the top hit. Approximate: it can pick a different
  release, a remaster, or miss entirely. Misses are reported, not fatal.

`--dry-run` does the classification and prints a match report **without
authenticating or creating anything** — the direct path is fully offline.

## One-time setup

1. Create an app at <https://developer.spotify.com/dashboard>.
2. **Redirect URI** — add exactly `http://127.0.0.1:8888/callback`. Since the
   2025 OAuth migration, Spotify **rejects `localhost`**; you must use the
   loopback IP literal. HTTP is allowed for loopback.
3. Copy the app's **Client ID**. Under PKCE the client is *public* — there is
   **no client secret to store**.
4. Provide the ID one of two ways:
   - `export SPOTIFY_CLIENT_ID=<your-client-id>`, or
   - add a line `SPOTIFY_CLIENT_ID=<your-client-id>` to the gitignored
     `secrets.md`.
5. `pip install spotipy` (already in `requirements.txt`).

First non-`--dry-run` run opens a browser for the Spotify consent screen
(scopes: `playlist-modify-private playlist-modify-public`). The resulting token
is cached in the gitignored `.cache-spotify` and refreshed automatically.

## Auth model

Authorization Code with **PKCE** (Proof Key for Code Exchange). Creating a
playlist is a user-scoped action, so the app-only Client Credentials flow can't
do it — it needs a signed-in user. PKCE is the current recommended flow for a
public client (no secret); `spotipy`'s `SpotifyPKCE` runs the code-verifier
handshake and the loopback callback server for us.

## Limitations & fidelity

- **MSD sets**: era-limited catalog + fuzzy search, so expect misses and the
  occasional wrong version. The default-library path has neither problem.
- **No transitions**: a playlist is an ordered list; curator's harmonic and
  tempo work between tracks isn't expressible in Spotify. Crossfade is the
  closest client-side approximation.
- **No file import**: Spotify has no native "import a CSV/M3U" feature, which is
  why this uses the Web API. If you'd rather not register an app, a third-party
  importer (Soundiiz, TuneMyMusic) can ingest curator's existing `.m3u8`/`.csv`
  exports — lossy in the same search-matching way.

## Credential hygiene

The Client ID is not a secret under PKCE, but the OAuth token is: `.cache-spotify`
holds access/refresh tokens and is gitignored. Never commit `secrets.md` or the
cache. The script reads the ID from the environment or `secrets.md` only — it is
never hardcoded.

## Sources

- OAuth migration — PKCE required, `localhost` removed:
  <https://developer.spotify.com/blog/2025-10-14-reminder-oauth-migration-27-nov-2025>
- Redirect URI rules:
  <https://developer.spotify.com/documentation/web-api/concepts/redirect_uri>
- Authorization Code with PKCE:
  <https://developer.spotify.com/documentation/web-api/tutorials/code-pkce-flow>
- spotipy API reference: <https://spotipy.readthedocs.io/en/latest/>
