"""
Minimal Perplexity API client.

One OpenAI-shaped endpoint (chat/completions) called with `requests` - no
SDK dependency. The API has NO native tool/function calling (verified
against the official OpenAPI spec); what it does have is `response_format`
json_schema structured outputs and built-in web search with citations, and
the agent layer is built on exactly those two capabilities.

The key is resolved from the PERPLEXITY_API_KEY env var, falling back to
the gitignored secrets.md. It is never logged and never included in errors.
"""

import json
import os
import re
import time

import requests

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRETS_PATH = os.path.join(ROOT, "secrets.md")
RETRY_STATUSES = {429, 500, 502, 503, 504}


class PplxError(RuntimeError):
    """API-level failure (network, HTTP error, or non-JSON structured reply)."""


def resolve_api_key(secrets_path: str = SECRETS_PATH) -> str | None:
    key = os.environ.get("PERPLEXITY_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(secrets_path):
        with open(secrets_path) as f:
            m = re.search(r"pplx-[A-Za-z0-9]+", f.read())
        if m:
            return m.group(0)
    return None


class PerplexityClient:
    """`transport` is requests.post-shaped and injectable for tests."""

    def __init__(self, api_key: str | None = None, model: str = config.PPLX_MODEL,
                 transport=None, secrets_path: str = SECRETS_PATH):
        self.api_key = api_key if api_key is not None else resolve_api_key(secrets_path)
        self.model = model
        self.transport = transport or requests.post

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict], schema: dict | None = None,
             search: bool = False, model: str | None = None,
             max_tokens: int = config.AGENT_MAX_TOKENS,
             temperature: float = 0.2) -> tuple[dict | str, list[dict]]:
        """Returns (parsed JSON dict if schema else text, citations).

        Citations are [{"title", "url"}, ...], normalized across the API's
        `citations` (bare URLs) and `search_results` response fields.
        """
        if not self.available:
            raise PplxError("no API key (set PERPLEXITY_API_KEY or secrets.md)")

        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
        if not search:
            payload["disable_search"] = True

        data = self._post(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise PplxError(f"malformed API response: {e}") from e

        citations = _citations(data)
        if schema is None:
            return content, citations
        try:
            return json.loads(content), citations
        except json.JSONDecodeError as e:
            raise PplxError(f"model returned invalid JSON: {e}") from e

    def _post(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last = "no attempt made"
        for attempt in range(config.PPLX_RETRIES + 1):
            if attempt:
                time.sleep(1.5 ** attempt)
            try:
                resp = self.transport(config.PPLX_URL, headers=headers,
                                      json=payload, timeout=config.PPLX_TIMEOUT_S)
            except requests.RequestException as e:
                last = f"network error: {type(e).__name__}"
                continue
            if resp.status_code == 200:
                return resp.json()
            last = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code not in RETRY_STATUSES:
                break
        raise PplxError(last)


def _citations(data: dict) -> list[dict]:
    out, seen = [], set()
    for r in data.get("search_results") or []:
        url = r.get("url")
        if url and url not in seen:
            seen.add(url)
            out.append({"title": r.get("title") or url, "url": url})
    for url in data.get("citations") or []:
        if url and url not in seen:
            seen.add(url)
            out.append({"title": url, "url": url})
    return out
