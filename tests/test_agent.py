import json

import pytest

from src import agent, config, exporters, pplx


# ---------------------------------------------------------------- fakes
class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)[:400]

    def json(self):
        return self._payload


def reply(content, search_results=None):
    """API-shaped body; dict content is serialized like a structured output."""
    if not isinstance(content, str):
        content = json.dumps(content)
    return FakeResp({
        "choices": [{"message": {"content": content}}],
        "search_results": search_results or [],
    })


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []          # captured request payloads, in order

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        return self.replies.pop(0)


def make_client(replies):
    t = FakeTransport(replies)
    return pplx.PerplexityClient(api_key="pplx-test", transport=t), t


PLAN = {"template": "peak_time", "boosts": {"techno": 1.5}, "rationale": "warehouse brief"}
ACCEPT = {"verdict": "accept", **PLAN}


# ---------------------------------------------------------------- happy path
def test_happy_path_research_plan_accept(mini_library, mini_adjacency):
    client, t = make_client([
        reply({"summary": "Dark industrial techno rooms."},
              search_results=[{"title": "RA guide", "url": "https://ra.co/x"}]),
        reply(PLAN),
        reply(ACCEPT),
    ])
    result, notes = agent.run("dark warehouse", 0.3, mini_library, mini_adjacency,
                              seed=3, client=client)

    assert len(t.calls) == 3
    assert result.template.name == "peak_time"
    assert notes["fallback"] is None
    assert notes["vibe_reading"] == "Dark industrial techno rooms."
    assert notes["revisions"] == 0
    assert notes["citations"] == [{"title": "RA guide", "url": "https://ra.co/x"}]
    assert result.agent_notes is notes

    # research call searches; plan and critique must not
    assert "disable_search" not in t.calls[0]
    assert t.calls[1]["disable_search"] is True
    assert t.calls[2]["disable_search"] is True
    # every call is schema-constrained
    assert all(c["response_format"]["type"] == "json_schema" for c in t.calls)


def test_revise_loop_is_bounded_and_applied(mini_library, mini_adjacency):
    revise = {"verdict": "revise", "template": "warmup",
              "boosts": {"house": 1.8}, "rationale": "too hot for the brief"}
    client, t = make_client([reply(PLAN), reply(revise), reply(ACCEPT)])
    result, notes = agent.run("easy opener", 0.3, mini_library, mini_adjacency,
                              seed=3, search=False, client=client)

    assert len(t.calls) == 3            # plan + 2 critiques (revise, accept)
    assert result.template.name == "warmup"
    assert notes["revisions"] == 1
    assert notes["plan_rationale"] == "too hot for the brief"
    assert notes["boosts"] == {"house": 1.8}


def test_boosts_clamped_to_whitelist(mini_library, mini_adjacency):
    wild = {"template": "peak_time",
            "boosts": {"techno": 99.0, "house": -3.0}, "rationale": "extreme"}
    client, _ = make_client([reply(wild), reply({"verdict": "accept", **wild})])
    _, notes = agent.run("x", 0.3, mini_library, mini_adjacency,
                         seed=3, search=False, client=client)
    assert notes["boosts"]["techno"] == config.AGENT_BOOST_MAX
    assert notes["boosts"]["house"] == config.AGENT_BOOST_MIN


def test_pinned_template_wins(mini_library, mini_adjacency):
    client, _ = make_client([reply(PLAN), reply(ACCEPT)])
    result, _ = agent.run("dark warehouse", 0.3, mini_library, mini_adjacency,
                          seed=3, search=False, pinned_template="closing",
                          client=client)
    assert result.template.name == "closing"


# ---------------------------------------------------------------- fallbacks
def test_invalid_plan_json_falls_back_to_parse_vibe(mini_library, mini_adjacency):
    client, _ = make_client([reply("this is not json")])
    result, notes = agent.run("dark warehouse peak techno", 0.3,
                              mini_library, mini_adjacency, seed=3,
                              search=False, client=client)
    assert "plan call failed" in notes["fallback"]
    assert result.template.name == "peak_time"      # parse_vibe's choice
    assert len(result.tracklist) > 0


def test_no_key_falls_back_without_any_call(mini_library, mini_adjacency):
    def explode(*a, **k):
        raise AssertionError("transport must not be called without a key")
    client = pplx.PerplexityClient(api_key="", transport=explode)
    result, notes = agent.run("sunrise closing", 0.3, mini_library,
                              mini_adjacency, seed=3, client=client)
    assert "no API key" in notes["fallback"]
    assert result.template.name == "closing"


def test_critique_failure_keeps_current_set(mini_library, mini_adjacency):
    client, _ = make_client([reply(PLAN), FakeResp({"error": "nope"}, status=400)])
    result, notes = agent.run("x", 0.3, mini_library, mini_adjacency,
                              seed=3, search=False, client=client)
    assert notes["fallback"] is None                # plan survived
    assert result.template.name == "peak_time"
    assert notes["revisions"] == 0


# ---------------------------------------------------------------- plumbing
def test_citations_reach_json_export(mini_library, mini_adjacency, tmp_path):
    client, _ = make_client([
        reply({"summary": "Notes."},
              search_results=[{"title": "Source", "url": "https://e.com/1"}]),
        reply(PLAN),
        reply(ACCEPT),
    ])
    result, _ = agent.run("v", 0.3, mini_library, mini_adjacency,
                          seed=3, client=client)
    path = tmp_path / "set.json"
    from src.explain import add_rationales
    exporters.export_json(result, add_rationales(result), str(path))
    payload = json.loads(path.read_text())
    assert payload["agent"]["citations"] == [{"title": "Source", "url": "https://e.com/1"}]


def test_client_retries_on_429(monkeypatch, mini_library, mini_adjacency):
    monkeypatch.setattr(pplx.time, "sleep", lambda s: None)
    transport = FakeTransport([FakeResp({"e": 1}, status=429), reply(PLAN)])
    client = pplx.PerplexityClient(api_key="pplx-test", transport=transport)
    data, _ = client.chat([{"role": "user", "content": "x"}],
                          schema=agent.PLAN_SCHEMA)
    assert data["template"] == "peak_time"
    assert len(transport.calls) == 2


def test_client_no_retry_on_400():
    transport = FakeTransport([FakeResp({"error": "bad schema"}, status=400)])
    client = pplx.PerplexityClient(api_key="pplx-test", transport=transport)
    with pytest.raises(pplx.PplxError, match="HTTP 400"):
        client.chat([{"role": "user", "content": "x"}])
    assert len(transport.calls) == 1
