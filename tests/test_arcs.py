import pytest

from src import arcs, config


def test_all_templates_cover_unit_interval_contiguously():
    for template in arcs.TEMPLATES.values():
        phases = template.phases
        assert phases[0].t0 == 0.0
        assert phases[-1].t1 == 1.0
        for prev, nxt in zip(phases, phases[1:]):
            assert prev.t1 == pytest.approx(nxt.t0)


def test_targets_stay_in_bounds():
    for template in arcs.TEMPLATES.values():
        for i in range(0, 101):
            energy, weights = template.target(i / 100)
            assert 0.0 <= energy <= 1.0
            assert set(weights) == set(config.GENRES)
            assert all(w > 0 for w in weights.values())


def test_energy_interpolates_within_phase():
    t = arcs.TEMPLATES["warmup"]
    e_start, _ = t.target(0.0)
    e_mid, _ = t.target(0.25)
    e_end, _ = t.target(0.499)
    assert e_start < e_mid < e_end            # ease_in rises 0.35 -> 0.48


def test_parse_vibe_word_boundaries():
    # "warehouse" must not trigger the "house" keyword
    template, boosts = arcs.parse_vibe("dark warehouse peak hour techno")
    assert template.name == "peak_time"
    assert boosts["techno"] > 1.0
    assert boosts["house"] == 1.0


def test_parse_vibe_sunrise_leans_house():
    template, boosts = arcs.parse_vibe("sunrise melodic closing set")
    assert template.name == "closing"
    assert boosts["house"] > 1.0


def test_parse_vibe_unknown_defaults_to_journey():
    template, boosts = arcs.parse_vibe("xyzzy quux")
    assert template.name == "full_journey"
    assert all(v == 1.0 for v in boosts.values())
