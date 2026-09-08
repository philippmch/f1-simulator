"""Slick opening scores follow the executed policy and actual finish deadline."""

import copy
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation import opening_strategy as opening
from f1sim.simulation import race_timing
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


def fixture(laps=12, base=90):
    simulator = RaceSimulator(np.random.default_rng(42))
    return (Driver(id="a", name="A", team_id="a"), Car(team_id="a", team_name="A"),
            Track(id="t", name="T", country="T", total_laps=laps, base_lap_time=base,
                  pit_lane_delta=22), Weather(), TeamStrategyArchetype.BALANCED,
            simulator.strategy_tuning, simulator.strategy_profiles), simulator


@pytest.mark.parametrize("laps,base", [(12, 90), (90, 110)])
def test_scores_match_direct_completed_policy_and_preserve_inputs(laps, base):
    args, simulator = fixture(laps, base)
    before = copy.deepcopy((args, simulator.rng.bit_generator.state))
    scores = opening.dry_opening_policy_costs(*args)
    for compound, score in scores:
        distance, elapsed = opening._policy_path_outcome(*args, compound, 0)
        assert score == opening.OpeningPolicyScore(-distance if elapsed != inf else inf, elapsed)
        if laps == 90:
            assert distance == 65
    assert (args, simulator.rng.bit_generator.state) == before
    if laps == 90:
        assert min(scores, key=lambda item: item[1])[0] == TireCompound.HARD
        assert dict(scores)[TireCompound.MEDIUM].mean_time == pytest.approx(7424.44058160)


@pytest.mark.parametrize("rain,wetness,seeds", [(0, 0, (0,)), (0, .079, (0,)),
                                                (0, .08, opening.REACTION_SEEDS),
                                                (.1, 0, opening.REACTION_SEEDS)])
def test_reaction_samples_and_distance_first_ranking(monkeypatch, rain, wetness, seeds):
    args, _ = fixture()
    args[3].rain_intensity = rain
    args[3].track_wetness = wetness
    calls = []

    def outcome(*values):
        compound, seed = values[-2:]
        calls.append((compound, seed))
        return {TireCompound.SOFT: (12, 1200), TireCompound.MEDIUM: (11, 1100),
                TireCompound.HARD: (12, inf)}[compound]

    monkeypatch.setattr(opening, "_policy_path_outcome", outcome)
    opening._cached_dry_policy_costs.cache_clear()
    scores = opening.dry_opening_policy_costs(*args)
    assert calls == [(compound, seed) for compound in opening.SLICKS for seed in seeds]
    assert min(scores, key=lambda item: item[1])[0] == TireCompound.SOFT
    assert dict(scores)[TireCompound.HARD] == opening.OpeningPolicyScore(inf, inf)


def test_cache_normalizes_identity_and_invalidates_physics(monkeypatch):
    args, _ = fixture()
    opening._cached_dry_policy_costs.cache_clear()
    baseline = opening.dry_opening_policy_costs(*args)
    args[0].id = args[0].name = args[0].team_id = "other"
    args[0].current_tire_laps = 99
    args[0].dnf = True
    args[1].team_id = args[1].team_name = "other"
    assert opening.dry_opening_policy_costs(*args) is baseline
    assert opening._cached_dry_policy_costs.cache_info().hits == 1
    mutations = [lambda: setattr(args[0], "skill_rating", .9),
                 lambda: setattr(args[1], "base_pace", .9),
                 lambda: setattr(args[2], "total_laps", 13),
                 lambda: setattr(args[3], "track_temperature", 24),
                 lambda: args[5].update(pit_prob_min=args[5]["pit_prob_min"] + .001),
                 lambda: args[6]["balanced"].update(long_stint_threshold=25),
                 lambda: monkeypatch.setitem(TIRE_COMPOUNDS, TireCompound.SOFT,
                     TIRE_COMPOUNDS[TireCompound.SOFT].model_copy(update={"initial_grip": 1.06})),
                 lambda: monkeypatch.setattr(race_timing, "RACING_TIME_LIMIT_SECONDS", 900)]
    for expected_misses, mutate in enumerate(mutations, 2):
        mutate()
        opening.dry_opening_policy_costs(*args)
        assert opening._cached_dry_policy_costs.cache_info().misses == expected_misses
    assert opening._cached_dry_policy_costs.cache_info().maxsize == 128


