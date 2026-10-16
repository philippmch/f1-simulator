"""Race tyre constraints must reach each isolated opening-policy projection."""

import struct
from copy import deepcopy
from dataclasses import asdict
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TireCompound
from f1sim.simulation import opening_strategy as opening
from f1sim.simulation import race
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


def inputs(*, wet=False):
    simulator = RaceSimulator(np.random.default_rng(42))
    return (Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"),
            Track(id="T", name="T", country="T", total_laps=6, base_lap_time=90),
            Weather(track_wetness=.5, rain_intensity=.5) if wet else Weather(),
            TeamStrategyArchetype.BALANCED, simulator.strategy_tuning, simulator.strategy_profiles)


def bits(value):
    if isinstance(value, float):
        return struct.pack(">d", value).hex()
    if isinstance(value, (tuple, list)):
        return [bits(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return bits(asdict(value))
    if isinstance(value, dict):
        return {key: bits(item) for key, item in value.items()}
    return value


# Captured from the pre-propagation implementation with these exact inputs.
# A small tolerance permits platform math differences in the fixed reference;
# all disabled forms must still match every score bit within the same runtime.
DISABLED_SCORE_TIMES = {
    "dry": (("soft", "4081f3740a54d629"), ("medium", "4081f3740a54d629"),
            ("hard", "4081f6b17af8ad34")),
    "weather": (("intermediate", "408342f053ba9239"), ("wet", "408353e8db3b2499")),
    "inventory": (("I1", "408342f053ba9239"), ("I2", "408349d29d713f88")),
}


@pytest.mark.parametrize("mode", ["dry", "weather", "inventory"])
def test_disabled_opening_scores_preserve_pre_rules_costs_and_opt_out_bits(mode):
    arguments = inputs(wet=mode != "dry")
    if mode == "inventory":
        arguments += ([dict(id="I1", compound="intermediate"),
                       dict(id="I2", compound="intermediate", age=2)],)
    name = {"dry": "dry_opening_policy_costs", "weather": "opening_policy_costs",
            "inventory": "inventory_opening_policy_costs"}[mode]
    default = getattr(opening, name)(*arguments)
    assert [identifier for identifier, _ in default] == [
        identifier for identifier, _ in DISABLED_SCORE_TIMES[mode]]
    for (_, score), (_, mean_time) in zip(default, DISABLED_SCORE_TIMES[mode], strict=True):
        expected_time, = struct.unpack(">d", bytes.fromhex(mean_time))
        assert score.mean_time == pytest.approx(expected_time, rel=0, abs=1.e-12)
        assert score.negative_mean_laps == -6
    for options in ({"race_tire_rules": None}, {"race_tire_rules": {}},
                    {"race_tire_rules": {"minimum_sets": 1}}):
        actual = getattr(opening, name)(*arguments, **options)
        assert bits(actual) == bits(default)


@pytest.mark.parametrize("mode", ["dry", "weather", "inventory"])
def test_rules_are_canonical_separate_cache_keys_and_do_not_alias_inputs(monkeypatch, mode):
    arguments = inputs(wet=mode != "dry")
    if mode == "inventory":
        arguments += ([dict(id="I1", compound="intermediate")],)
    public = {"dry": "dry_opening_policy_costs", "weather": "opening_policy_costs",
              "inventory": "inventory_opening_policy_costs"}[mode]
    cache = {"dry": "_cached_dry_policy_costs", "weather": "_cached_policy_costs",
             "inventory": "_cached_inventory_policy_costs"}[mode]
    calls = []
    monkeypatch.setattr(opening, cache, lambda *a, **kw: calls.append(kw) or ())
    rules = {"minimum_sets": 3, "max_set_laps": 2}
    before = deepcopy(arguments)
    getattr(opening, public)(*arguments, race_tire_rules=rules)
    rules["max_set_laps"] = 3
    getattr(opening, public)(*arguments, race_tire_rules=rules)
    getattr(opening, public)(*arguments, race_tire_rules={})
    assert calls[0]["race_tire_rules_json"] == '{"max_set_laps": 2, "minimum_sets": 3}'
    assert calls[1]["race_tire_rules_json"] == '{"max_set_laps": 3, "minimum_sets": 3}'
    assert "race_tire_rules_json" not in calls[2]
    assert arguments == before


@pytest.mark.parametrize("mode", ["dry", "weather", "inventory"])
def test_invalid_rules_rejected_before_any_policy_projection(monkeypatch, mode):
    arguments = inputs(wet=mode != "dry")
    if mode == "inventory":
        arguments += ([dict(id="I1", compound="intermediate")],)
    public = {"dry": "dry_opening_policy_costs", "weather": "opening_policy_costs",
              "inventory": "inventory_opening_policy_costs"}[mode]
    monkeypatch.setattr(opening, "_policy_path_outcome", lambda *a, **kw: pytest.fail("projected"))
    with pytest.raises(ValueError, match="race_tire_rules"):
        getattr(opening, public)(*arguments, race_tire_rules={"max_set_laps": True})


@pytest.mark.parametrize("finite", [False, True])
def test_manual_projection_initializes_and_advances_actual_rule_ledger(monkeypatch, finite):
    captured = []
    original = RaceSimulator._race_tire_usage_fields

    def inspect(state):
        value = original(state)
        captured.append(deepcopy(value))
        return value

    monkeypatch.setattr(RaceSimulator, "_race_tire_usage_fields", staticmethod(inspect))
    options = {}
    if finite:
        options = {"tire_inventory": [dict(id=f"I{i}", compound="intermediate")
                                       for i in range(1, 4)], "opening_set_id": "I1"}
    outcome = opening._policy_path_outcome(
        *inputs(wet=True), TireCompound.INTERMEDIATE, 0,
        race_tire_rules={"max_set_laps": 2, "minimum_sets": 3}, pit_plan=[], **options)
    assert outcome[0] == 6 and outcome[1] != inf
    usage = captured[-1]["race_tire_usage"]
    assert usage["compliant"] and len(usage["used_set_ids"]) == 3
    assert sorted(usage["completed_race_laps_by_set"].values()) == [2, 2, 2]


def test_impossible_manual_projection_is_not_given_a_legal_finish():
    result = opening._policy_path_outcome(
        *inputs(wet=True), TireCompound.INTERMEDIATE, 0,
        race_tire_rules={"minimum_sets": 7}, pit_plan=[])
    assert result[0] == 6 and result[1] == inf


@pytest.mark.parametrize("wet", [False, True])
def test_native_unlimited_forwarding_preserves_rng_contract(monkeypatch, wet):
    arguments = inputs(wet=wet)
    driver, car, track, weather, style, _, _ = arguments
    simulator = RaceSimulator(np.random.default_rng(17), race_tire_rules={"minimum_sets": 3})
    calls = []
    name = "opening_policy_costs" if wet else "dry_opening_policy_costs"
    candidate = TireCompound.INTERMEDIATE if wet else TireCompound.SOFT
    scores = ((candidate, opening.OpeningPolicyScore(-6, 1)),)
    if not wet:
        scores = tuple((c, opening.OpeningPolicyScore(-6, i+1))
                       for i, c in enumerate(opening.SLICKS))
    monkeypatch.setattr(race, name, lambda *a, **kw: calls.append(kw) or scores)
    expected = np.random.default_rng(17)
    if not wet:
        expected.choice([c.value for c in opening.SLICKS], p=[1., 0., 0.])
    simulator._choose_starting_compound(style, track, weather, driver, car)
    assert calls[0]["race_tire_rules"] == {"minimum_sets": 3}
    assert simulator.rng.bit_generator.state == expected.bit_generator.state


def test_native_automatic_finite_selector_forwards_rules(monkeypatch):
    driver, car, track, weather, style, _, _ = inputs(wet=True)
    simulator = RaceSimulator(np.random.default_rng(17), race_tire_rules={"minimum_sets": 3})
    records = [dict(id="I1", compound="intermediate")]
    calls = []
    monkeypatch.setattr(opening, "inventory_opening_policy_costs", lambda *a, **kw:
                        calls.append(kw) or (("I1", opening.OpeningPolicyScore(-6, 1)),))
    simulator._inventory_opening_set(driver, car, track, weather, style, records)
    assert calls[0]["race_tire_rules"] == {"minimum_sets": 3}
