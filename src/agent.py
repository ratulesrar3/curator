"""
Perplexity-backed set-planning agent over the same functions the CLI calls.

The Perplexity API has no native tool calling, so this is not a free-form
tool loop: it is a staged, bounded pipeline where every LLM reply is
constrained to a JSON schema and every levered value is clamped to a
whitelist before it touches the sequencer.

    research (web search ON, citations kept)
      -> plan   (search off: template + genre boosts)
      -> build_set (deterministic)
      -> critique against set_report metrics (accept / revise, <= 2 loops)
      -> agent notes + citations threaded into the exports

Every failure mode (no key, HTTP error, invalid JSON) falls back to the
deterministic arcs.parse_vibe path - an agent run never fails to produce
a set. Search-grounded vibe interpretation is the reason Perplexity is the
provider here: "closing set at Robert Johnson" gets researched, not guessed.
"""

import json
import re

import pandas as pd

from . import arcs, config
from .explain import set_report
from .pplx import PerplexityClient, PplxError
from .sequencer import SetResult, build_set

RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "2-3 sentences: what this vibe implies musically "
                           "(energy shape, genres, era, mood).",
        },
    },
    "required": ["summary"],
    "additionalProperties": False,
}

_PLAN_PROPS = {
    "template": {"type": "string", "enum": sorted(arcs.TEMPLATES)},
    "boosts": {
        "type": "object",
        "properties": {g: {"type": "number"} for g in config.GENRES},
        "additionalProperties": False,
    },
    "rationale": {"type": "string",
                  "description": "1-3 sentences; this is printed on the set report"},
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": _PLAN_PROPS,
    "required": ["template", "boosts", "rationale"],
    "additionalProperties": False,
}

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["accept", "revise"]},
                   **_PLAN_PROPS},
    "required": ["verdict", "template", "boosts", "rationale"],
    "additionalProperties": False,
}


def _catalog() -> str:
    lines = []
    for name in sorted(arcs.TEMPLATES):
        t = arcs.TEMPLATES[name]
        shape = " -> ".join(f"{p.name} {p.e0:.2f}-{p.e1:.2f}" for p in t.phases)
        lines.append(f"- {name}: {t.description} [{shape}]")
    return "\n".join(lines)


def _system_prompt(pinned_template: str | None) -> str:
    pin = (f"\nThe user has pinned the template to '{pinned_template}'; "
           "always return that template and only tune boosts." if pinned_template else "")
    return (
        "You plan long-form DJ sets for curator, a sequencer that follows an "
        "energy arc template and per-genre boost multipliers.\n\n"
        f"Templates (phase energy targets on a 0-1 scale):\n{_catalog()}\n\n"
        f"Genres: {', '.join(config.GENRES)}.\n"
        f"Boosts multiply a genre's presence; use {config.AGENT_BOOST_MIN} to "
        f"{config.AGENT_BOOST_MAX}, 1.0 = neutral. Only boost genres the vibe "
        "actually calls for.\nRespond with JSON matching the schema - no prose."
        + pin
    )


def _strip_markup(text: str | None) -> str | None:
    """Drop sonar's inline citation markers ([3][8]) and bold asterisks -
    the sources ride separately as structured citations."""
    if not text:
        return text
    return re.sub(r"\[\d+\]|\*\*", "", text).strip()


def _clamp_boosts(raw: dict | None) -> dict[str, float]:
    boosts = {g: 1.0 for g in config.GENRES}
    for g, v in (raw or {}).items():
        if g in boosts:
            try:
                boosts[g] = min(max(float(v), config.AGENT_BOOST_MIN),
                                config.AGENT_BOOST_MAX)
            except (TypeError, ValueError):
                pass
    return boosts


def _fallback(reason: str, vibe: str, hours: float, lib: pd.DataFrame,
              adjacency: pd.DataFrame, seed: int,
              pinned_template: str | None) -> tuple[SetResult, dict]:
    template, boosts = arcs.parse_vibe(vibe)
    if pinned_template:
        template = arcs.TEMPLATES[pinned_template]
    result = build_set(lib, adjacency, template, hours=hours,
                       boosts=boosts, vibe=vibe, seed=seed)
    notes = {"fallback": reason, "model": None, "vibe_reading": None,
             "plan_rationale": None, "revisions": 0, "citations": []}
    result.agent_notes = notes
    return result, notes


def run(vibe: str, hours: float, lib: pd.DataFrame, adjacency: pd.DataFrame,
        seed: int = 42, pinned_template: str | None = None,
        search: bool = True, model: str = config.PPLX_MODEL,
        client: PerplexityClient | None = None) -> tuple[SetResult, dict]:
    """Plan, build, and self-critique a set. Returns (SetResult, notes);
    notes ride on result.agent_notes into every exporter."""
    client = client or PerplexityClient(model=model)
    if not client.available:
        return _fallback("no API key - used deterministic vibe parser",
                         vibe, hours, lib, adjacency, seed, pinned_template)

    system = {"role": "system", "content": _system_prompt(pinned_template)}
    citations: list[dict] = []
    reading = None

    # ---- 1. research: web-grounded vibe interpretation ------------------
    if search and vibe.strip():
        try:
            data, cites = client.chat(
                [{"role": "system",
                  "content": "You are a music researcher. Answer in JSON "
                             "matching the schema - no prose."},
                 {"role": "user",
                  "content": "Interpret this DJ-set brief. If it names venues, "
                             "cities, events, or scenes, research what music is "
                             f"played there.\n\nBrief: {vibe!r}"}],
                schema=RESEARCH_SCHEMA, search=True, model=model)
            reading = _strip_markup(data.get("summary"))
            citations = cites
        except PplxError:
            reading = None  # research is optional garnish; planning continues

    # ---- 2. plan ---------------------------------------------------------
    context = f"\n\nResearch notes: {reading}" if reading else ""
    try:
        plan, _ = client.chat(
            [system, {"role": "user",
                      "content": f"Plan a {hours:g}-hour set.\nVibe: "
                                 f"{vibe or '(none given)'}{context}"}],
            schema=PLAN_SCHEMA, search=False, model=model)
    except PplxError as e:
        return _fallback(f"plan call failed ({e}) - used deterministic vibe "
                         "parser", vibe, hours, lib, adjacency, seed,
                         pinned_template)

    template_name = pinned_template or plan["template"]
    boosts = _clamp_boosts(plan.get("boosts"))
    result = build_set(lib, adjacency, arcs.TEMPLATES[template_name],
                       hours=hours, boosts=boosts, vibe=vibe, seed=seed)

    # ---- 3. critique / revise -------------------------------------------
    revisions = 0
    rationale = plan.get("rationale", "")
    for _ in range(config.AGENT_MAX_REVISIONS):
        report = set_report(result)
        metrics = {k: report[k] for k in
                   ("template", "n_tracks", "duration", "harmonic_pct",
                    "mean_abs_bpm_delta_pct", "arc_rmse", "genre_counts",
                    "genre_crossings", "relaxed_transitions")}
        try:
            critique, _ = client.chat(
                [system, {"role": "user", "content": (
                    f"Vibe: {vibe or '(none given)'}{context}\n"
                    f"Current plan: template={template_name}, "
                    f"boosts={json.dumps(boosts)}\n"
                    f"Sequencer metrics: {json.dumps(metrics)}\n\n"
                    "Accept unless something is clearly off (arc_rmse > 0.10, "
                    "harmonic_pct < 75, relaxed_transitions > 2, or a genre "
                    "mix that contradicts the vibe). If revising, return the "
                    "adjusted template/boosts.")}],
                schema=CRITIQUE_SCHEMA, search=False, model=model)
        except PplxError:
            break  # keep the set we have; note stays on the last good plan
        if critique.get("verdict") != "revise":
            break
        revisions += 1
        template_name = pinned_template or critique["template"]
        boosts = _clamp_boosts(critique.get("boosts"))
        rationale = critique.get("rationale", rationale)
        result = build_set(lib, adjacency, arcs.TEMPLATES[template_name],
                           hours=hours, boosts=boosts, vibe=vibe, seed=seed)

    notes = {"fallback": None, "model": model, "vibe_reading": reading,
             "plan_rationale": rationale, "revisions": revisions,
             "boosts": {g: v for g, v in boosts.items() if v != 1.0},
             "citations": citations}
    result.agent_notes = notes
    return result, notes
