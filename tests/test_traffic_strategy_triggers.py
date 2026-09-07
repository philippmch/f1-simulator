"""Strategy changes react to traffic rather than clean air or neutralisation."""

import copy

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.race import DriverRaceState, RaceSimulator, TeamStrategyArchetype


def make_track(laps=60):
    return Track(id="test", name="Test", country="Test", total_laps=laps, base_lap_time=90)


def make_state(strategy=TeamStrategyArchetype.CONSERVATIVE, position=8):
    return DriverRaceState(
        driver=Driver(id="test", name="Test", team_id="test"),
        car=Car(team_id="test", team_name="Test"),
        position=position,
        strategy_archetype=strategy,
        pit_plan_options=[[15, 40], [30]],
        active_pit_plan_index=1,
    )


@pytest.mark.parametrize("gap,expected", [(None, False), (-1, False), (0, True),
                                        (0.5, True), (1.99, True), (2, False), (8, False)])
def test_conservative_switch_requires_close_traffic(gap, expected):
    sim = RaceSimulator()
    assert sim._should_switch_conservative_to_balanced(
        make_state(), 30, make_track(), gap,
    ) is expected


@pytest.mark.parametrize("gap,expected", [(0.5, True), (1, True), (1.5, False), (4, False)])
def test_tuned_gap_is_a_maximum_not_a_minimum(gap, expected):
    sim = RaceSimulator(strategy_tuning={"conservative_switch_gap": 1.0})
    assert sim._should_switch_conservative_to_balanced(
        make_state(), 30, make_track(), gap,
    ) is expected


@pytest.mark.parametrize("position,lap,expected", [(6, 30, False), (7, 24, False),
                                                  (7, 25, True)])
def test_switch_retains_position_and_progress_guards(position, lap, expected):
    sim = RaceSimulator()
    assert sim._should_switch_conservative_to_balanced(
        make_state(position=position), lap, make_track(), 0.5,
    ) is expected


@pytest.mark.parametrize("flag", ["safety_car_active", "vsc_active", "red_flag_active",
                                 "sc_restart_lap_number", "red_flag_restart_lap_number"])
def test_bunching_does_not_switch_profile_or_fallback_plan(flag):
    sim = RaceSimulator()
    setattr(sim.event_manager, flag, 40 if flag.endswith("number") else True)
    assert not sim._should_switch_conservative_to_balanced(
        make_state(), 40, make_track(), 0.1,
    )
    state = make_state(TeamStrategyArchetype.BALANCED)
    assert sim._select_active_pit_plan(
        state, Weather(), 40, 0.1, track=make_track(),
    ) == [30]
    assert state.active_pit_plan_index == 1


@pytest.mark.parametrize("laps", [30, 60, 78])
def test_fallback_traffic_reaction_uses_actual_race_midpoint(laps):
    sim = RaceSimulator()
    state = make_state(TeamStrategyArchetype.BALANCED)
    track = make_track(laps)
    assert sim._select_active_pit_plan(
        state, Weather(), laps // 2, 0.5, track=track,
    ) == [30]
    assert sim._select_active_pit_plan(
        state, Weather(), laps // 2 + 1, 0.5, track=track,
    ) == [15, 40]


@pytest.mark.parametrize("gap", [None, 2, 8])
def test_clean_air_does_not_replace_fallback_plan(gap):
    sim = RaceSimulator()
    state = make_state(TeamStrategyArchetype.BALANCED)
    assert sim._select_active_pit_plan(state, Weather(), 40, gap, track=make_track()) == [30]


def test_wet_fallback_has_priority_over_close_traffic():
    sim = RaceSimulator()
    state = make_state(TeamStrategyArchetype.BALANCED)
    state.active_pit_plan_index = 0
    assert sim._select_active_pit_plan(
        state, Weather(track_wetness=0.4), 40, 0.5, track=make_track(),
    ) == [30]


def test_restart_exclusion_expires_without_consuming_rng():
    sim = RaceSimulator(rng=np.random.default_rng(4))
    sim.event_manager.sc_restart_lap_number = 30
    before = copy.deepcopy(sim.rng.bit_generator.state)
    state = make_state()
    assert not sim._should_switch_conservative_to_balanced(state, 30, make_track(), 0.5)
    assert sim._should_switch_conservative_to_balanced(state, 31, make_track(), 0.5)
    assert sim.rng.bit_generator.state == before
