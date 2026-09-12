"""Equivalent physical opening sets share work without losing inventory."""

from copy import deepcopy
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation import opening_strategy
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


@pytest.mark.parametrize("wetness,rain", [(0., 0.), (.18, .35), (.3, 0.)])
def test_equivalent_scores_match_independent_per_identity_paths(monkeypatch, wetness, rain):
    driver = Driver(id="projection", name="projection", team_id="projection")
    car = Car(team_id="projection", team_name="projection")
    track = Track(id="t", name="T", country="T", total_laps=6, base_lap_time=90)
    weather = Weather(track_wetness=wetness, rain_intensity=rain)
    simulator = RaceSimulator(np.random.default_rng(1))
    strategy = TeamStrategyArchetype.BALANCED
    args = (driver, car, track, weather, strategy, simulator.strategy_tuning,
            simulator.strategy_profiles)
    records = [{"id": str(i), "compound": compound, "age": age}
               for i, (compound, age) in enumerate([
                   ("soft", 5), ("soft", 5), ("hard", 0), ("hard", 0),
                   ("intermediate", 4), ("intermediate", 4), ("soft", 6),
               ])]
    before = deepcopy(records)
    eligible = [item for item in records
                if weather.tire_mismatch(TireCompound(item["compound"])) != "critical"]
    original = opening_strategy._policy_path_outcome
    expected = []
    for item in eligible:
        laps, time = original(*args, TireCompound(item["compound"]), 0,
                              tire_inventory=records, opening_set_id=item["id"])
        expected.append((item["id"], opening_strategy.OpeningPolicyScore(
            -laps if time != inf else inf, time)))
    calls = []

    def observe(*args, **kwargs):
        calls.append(kwargs["opening_set_id"])
        assert kwargs["tire_inventory"] == before
        return original(*args, **kwargs)

    opening_strategy._cached_inventory_policy_costs.cache_clear()
    monkeypatch.setattr(opening_strategy, "_policy_path_outcome", observe)
    result = opening_strategy.inventory_opening_policy_costs(*args, records)
    assert result == tuple(expected)
    assert len(calls) == len({(item["compound"], item["age"]) for item in eligible})
    assert len(calls) < len(eligible)
    assert records == before


def test_equivalent_opening_tie_keeps_first_identity_and_all_sets(monkeypatch):
    simulator = RaceSimulator(np.random.default_rng(1))
    records = [{"id": name, "compound": "wet", "age": 4} for name in ("Z", "A", "B")]
    original = opening_strategy._policy_path_outcome
    calls = []

    def observe(*args, **kwargs):
        calls.append(kwargs["opening_set_id"])
        return original(*args, **kwargs)

    opening_strategy._cached_inventory_policy_costs.cache_clear()
    monkeypatch.setattr(opening_strategy, "_policy_path_outcome", observe)
    pool, selected = simulator._inventory_opening_set(
        Driver(id="d", name="D", team_id="t"), Car(team_id="t", team_name="T"),
        Track(id="t", name="T", country="T", total_laps=4, base_lap_time=90),
        Weather(track_wetness=.8, rain_intensity=.8), TeamStrategyArchetype.BALANCED, records,
    )
    assert calls == ["Z"]
    assert selected.id == "Z"
    assert list(pool.sets) == ["Z", "A", "B"]
    pool.fit(selected.id)
    assert [item.id for item in pool.replacements()] == ["A", "B"]
