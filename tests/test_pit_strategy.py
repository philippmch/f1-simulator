import copy
from itertools import combinations, product
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import (
    SLICKS,
    expected_stationary_time,
    plan_dry_stop,
)
from f1sim.simulation.race import DriverRaceState, RaceSimulator, TeamStrategyArchetype


def _track() -> Track:
    return Track(
        id="test",
        name="Test Track",
        country="Nowhere",
        total_laps=60,
        base_lap_time=90.0,
        pit_lane_delta=22.0,
        overtake_difficulty=0.8,
        tire_stress=0.7,
        safety_car_probability=0.3,
    )


def _state(
    driver_id: str,
    position: int,
    total_time: float,
    tire: TireCompound = TireCompound.MEDIUM,
) -> DriverRaceState:
    driver = Driver(id=driver_id, name=driver_id, team_id=f"team_{driver_id}")
    car = Car(team_id=f"team_{driver_id}", team_name=f"Team {driver_id}")
    return DriverRaceState(
        driver=driver,
        car=car,
        position=position,
        total_time=total_time,
        current_tire=TIRE_COMPOUNDS[tire].model_copy(deep=True),
    )


def test_late_race_stop_on_hard_switches_to_soft() -> None:
    sim = RaceSimulator(rng=np.random.default_rng(1))
    state = _state("VER", position=3, total_time=1000.0, tire=TireCompound.HARD)
    weather = Weather(condition=WeatherCondition.DRY, track_wetness=0.0)

    sim._execute_pit_stop(state, _track(), weather, current_lap=52)

    assert state.current_tire.compound == TireCompound.SOFT


def test_large_gap_does_not_make_an_unnecessary_stop_free() -> None:
    sim = RaceSimulator(rng=np.random.default_rng(7))
    track = _track()
    weather = Weather(condition=WeatherCondition.DRY, track_wetness=0.0)

    state = _state("VER", position=2, total_time=1000.0, tire=TireCompound.MEDIUM)
    state.tire_laps = 0
    state.pit_stops = 1
    state.tire_compound_history = ["hard", "medium"]

    ahead = _state("NOR", position=1, total_time=999.2, tire=TireCompound.MEDIUM)
    # Keeping track position does not eliminate the stop's elapsed time cost.
    behind = _state("HAM", position=3, total_time=1024.0, tire=TireCompound.MEDIUM)

    should_pit = sim._should_pit(
        state,
        [ahead, state, behind],
        track,
        lap=22,
        pit_window_open=True,
        weather=weather,
    )

    assert should_pit is False


def _plan(state, track, remaining, stops=1, factor=1.0, modifier=1.0):
    return plan_dry_stop(
        state.driver, state.car, track, state.current_tire, state.tire_laps,
        remaining, stops, RaceSimulator._used_slick_compounds(state),
        RaceSimulator._has_used_wet_compound(state), factor,
        current_lap_time_modifier=modifier,
    )


def _enumerated_actions(state, track, laps, budget, factor, modifier=1.0):
    """Independent oracle: enumerate complete schedules and compound paths."""
    result = {True: (inf, set()), False: (inf, set())}
    for stops in range(budget + 1):
        for schedule in combinations(range(laps), stops):
            for path in product(SLICKS, repeat=stops):
                used = RaceSimulator._used_slick_compounds(state).copy()
                tire, age, cost, valid = state.current_tire, state.tire_laps, 0.0, True
                for lap in range(laps):
                    if lap in schedule:
                        compound = path[schedule.index(lap)]
                        if len(used) < 2 and compound in used:
                            valid = False
                            break
                        used.add(compound)
                        tire, age = TIRE_COMPOUNDS[compound], 0
                        cost += expected_stationary_time(state.car)
                        cost += track.pit_lane_delta * (factor if lap == 0 else 1)
                    cost += LapSimulator.tire_pace_contribution(
                        state.driver, state.car, track, tire, age
                    ) * (modifier if lap == 0 else 1.0)
                    age += 1
                if not valid or len(used) < 2:
                    continue
                now = bool(schedule and schedule[0] == 0)
                compound = path[0] if now else None
                previous, choices = result[now]
                if cost < previous - 1e-9:
                    result[now] = (cost, {compound})
                elif abs(cost - previous) < 1e-9:
                    choices.add(compound)
    return result


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
@pytest.mark.parametrize("compound", SLICKS)
@pytest.mark.parametrize("compliant", [False, True])
@pytest.mark.parametrize("factor", [1.0, 0.55, 0.75])
def test_solver_matches_exhaustive_schedules_and_compounds(budget, compound, compliant, factor):
    state = _state("A", 1, 0.0, compound)
    state.tire_laps = 28
    # Custom tyre physics must be used for old-set laps, not the stock compound.
    state.current_tire.degradation_rate = 0.037
    if compliant:
        state.tire_compound_history = ["soft", "medium", "hard"]
    track = _track()
    track.total_laps = 5
    track.pit_lane_delta = 0.5
    actual = _plan(state, track, 5, budget, factor)
    expected = _enumerated_actions(state, track, 5, budget, factor)
    assert actual.pit_now_cost == pytest.approx(expected[True][0])
    assert actual.wait_cost == pytest.approx(expected[False][0])
    if actual.compound is not None:
        assert actual.compound in expected[True][1]


def test_optional_stop_declined_when_pit_cost_exceeds_tyre_gain():
    state = _state("A", 1, 0.0)
    state.tire_laps = 22
    state.pit_stops = 1
    state.tire_compound_history = ["medium", "hard"]
    state.planned_pit_laps = [21, 42]
    sim = RaceSimulator(np.random.default_rng(3))
    assert not sim._should_pit(state, [state], _track(), 42, False, Weather())
    assert _plan(state, _track(), 19).pit_now_cost > _plan(state, _track(), 19).wait_cost


@pytest.mark.parametrize("modifier,factor", [(1.4, 0.55), (1.2, 0.75)])
@pytest.mark.parametrize("budget", [0, 1, 2, 3])
@pytest.mark.parametrize("compliant", [False, True])
def test_neutralized_current_lap_matches_exhaustive_green_future(
    modifier, factor, budget, compliant,
):
    state = _state("A", 1, 0)
    state.tire_laps = 25
    state.current_tire.degradation_rate = 0.037
    if compliant:
        state.tire_compound_history = ["medium", "hard"]
    track = _track()
    track.total_laps = 5
    track.pit_lane_delta = 0.5
    green_before = _plan(state, track, 5, budget)
    actual = _plan(state, track, 5, budget, factor, modifier)
    expected = _enumerated_actions(state, track, 5, budget, factor, modifier)
    assert actual.pit_now_cost == pytest.approx(expected[True][0])
    assert actual.wait_cost == pytest.approx(expected[False][0])
    if actual.compound is not None:
        assert actual.compound in expected[True][1]
    assert _plan(state, track, 5, budget) == green_before


def test_current_lap_modifier_reranks_first_compound_before_selection():
    state = _state("A", 1, 0)
    state.tire_laps = 20
    state.tire_compound_history = ["medium", "hard"]
    track = _track()
    track.tire_stress = 0.3
    green = _plan(state, track, 22)
    neutralized = _plan(state, track, 22, modifier=1.4)
    expected = _enumerated_actions(state, track, 22, 1, 1.0, 1.4)
    assert green.compound == TireCompound.MEDIUM
    assert neutralized.compound == TireCompound.SOFT
    assert neutralized.compound in expected[True][1]
    assert neutralized.pit_now_cost == pytest.approx(expected[True][0])
    assert neutralized.wait_cost == pytest.approx(expected[False][0])


@pytest.mark.parametrize("flag,modifier", [(None, 1.0), ("safety_car_active", 1.4),
                                          ("vsc_active", 1.2)])
def test_race_forwards_running_modifier_without_randomness(monkeypatch, flag, modifier):
    import f1sim.simulation.race as race_module

    simulator = RaceSimulator(np.random.default_rng(42))
    state = _state("A", 1, 0)
    state.tire_laps = 20
    state.tire_compound_history = ["medium", "hard"]
    if flag:
        setattr(simulator.event_manager, flag, True)
    forwarded = []

    def capture(*args, **kwargs):
        forwarded.append(kwargs["current_lap_time_modifier"])
        return plan_dry_stop(*args, **kwargs)

    monkeypatch.setattr(race_module, "plan_dry_stop", capture)
    before = copy.deepcopy(simulator.rng.bit_generator.state)
    simulator._should_pit(state, [state], _track(), 30, bool(flag), Weather())
    assert forwarded == [modifier]
    assert simulator.rng.bit_generator.state == before


def test_discounted_safety_car_stop_is_worthwhile_in_last_five_laps():
    state = _state("A", 1, 0.0, TireCompound.HARD)
    state.tire_laps = 50
    state.pit_stops = 1
    state.tire_compound_history = ["medium", "hard"]
    state.car.tire_degradation_factor = 1.5
    track = _track()
    track.pit_lane_delta = 19
    track.tire_stress = 0.8
    sim = RaceSimulator(np.random.default_rng(1))
    assert not sim._should_pit(state, [state], track, 55, False, Weather())
    sim.event_manager.safety_car_active = True
    state.car.tire_degradation_factor = 1.0
    assert not sim._should_pit(state, [state], track, 55, True, Weather())
    state.car.tire_degradation_factor = 1.5
    assert sim._should_pit(state, [state], track, 55, True, Weather())
    assert state.dry_pit_proposal == (55, TireCompound.SOFT)
    # Even an adversarial legacy stint chooser cannot replace the DP action.
    sim._choose_compound_for_next_stint = lambda *args: TireCompound.HARD
    sim._execute_pit_stop(state, track, Weather(), 55)
    assert state.current_tire.compound == TireCompound.SOFT
    assert state.dry_pit_proposal is None


def test_mandatory_rule_overrides_exhausted_budget_and_expensive_stop():
    state = _state("A", 1, 0.0)
    state.pit_stops = 5
    track = _track()
    track.pit_lane_delta = 60
    sim = RaceSimulator()
    assert sim._should_pit(state, [state], track, 59, False, Weather())
    sim._execute_pit_stop(state, track, Weather(), 59)
    assert len(sim._used_slick_compounds(state)) == 2


def test_weather_priority_overrides_dry_proposal_and_exhausted_budget():
    state = _state("A", 1, 0.0)
    state.pit_stops = 5
    sim = RaceSimulator()
    weather = Weather(condition=WeatherCondition.HEAVY_RAIN, track_wetness=0.9,
                      rain_intensity=0.9)
    assert sim._should_pit(state, [state], _track(), 60, False, weather)
    state.dry_pit_proposal = (60, TireCompound.SOFT)
    sim._execute_pit_stop(state, _track(), weather, 60)
    assert state.current_tire.compound == TireCompound.WET


def test_mandatory_extra_stop_is_optimized_before_the_final_safeguard():
    state = _state("A", 1, 0.0)
    state.pit_stops = 1
    state.tire_laps = 21
    state.tire_compound_history = ["medium", "medium"]
    track = _track()
    track.total_laps = 30
    track.tire_stress = 1.0
    track.pit_lane_delta = 1
    sim = RaceSimulator()
    assert sim._should_pit(state, [state], track, 16, False, Weather())
    sim._execute_pit_stop(state, track, Weather(), 16)
    assert len(sim._used_slick_compounds(state)) == 2


def test_projection_is_rng_free_and_service_expectation_matches_sampled_execution():
    state = _state("A", 1, 0.0)
    state.car.pit_stop_avg = 1.5
    state.car.pit_stop_std = 1.0
    rng = np.random.default_rng(42)
    before = copy.deepcopy(rng.bit_generator.state)
    sim = RaceSimulator(rng)
    sim._should_pit(state, [state], _track(), 30, False, Weather())
    sim._should_pit(state, [state], _track(), 30, False, Weather())
    assert rng.bit_generator.state == before
    # Monte Carlo integration of the independent execution distribution,
    # including floor clipping after adding the occasional slow-stop delay.
    normals = rng.normal(state.car.pit_stop_avg, state.car.pit_stop_std, 500_000)
    delays = (rng.random(500_000) < 0.05) * rng.uniform(2, 8, 500_000)
    assert expected_stationary_time(state.car) == pytest.approx(
        np.maximum(1.8, normals + delays).mean(), abs=0.01
    )


def test_bounded_timing_preference_cannot_overrule_material_cost():
    from f1sim.simulation.pit_strategy import DryPitDecision

    assert not DryPitDecision(11, 10, TireCompound.SOFT).should_pit(100)
    assert DryPitDecision(10.05, 10, TireCompound.SOFT).should_pit(0.1)
    assert not DryPitDecision(9.95, 10, TireCompound.SOFT).should_pit(-0.1)


@pytest.mark.parametrize("stress", [0.3, 0.9])
def test_full_race_matches_best_exhaustive_legal_one_stop(stress):
    class MeanPace:
        def normal(self, mean, _std):
            return mean

    def run(forced_lap=None, forced_compound=None):
        state = _state("A", 1, 0.0)
        track = _track()
        track.total_laps = 30
        track.tire_stress = stress
        sim = RaceSimulator(np.random.default_rng(42))
        sim.event_manager.process_lap = lambda **kwargs: []
        sim.lap_simulator.rng = MeanPace()
        sim.lap_simulator.calculate_pit_stop_time = (
            lambda *args: expected_stationary_time(state.car)
        )
        sim._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
        if forced_lap is not None:
            sim._should_pit = lambda state, states, track, lap, *args, **kwargs: lap == forced_lap
            sim._choose_distinct_dry_compound = lambda *args: forced_compound
        return sim.simulate_race(
            [state.driver], {state.car.team_id: state.car}, track, Weather(change_probability=0),
            [state.driver.id], starting_tires={state.driver.id: TireCompound.MEDIUM},
        )[0]

    selected = run()
    alternatives = [run(lap, compound).total_time for lap in range(6, 31)
                    for compound in (TireCompound.SOFT, TireCompound.HARD)]
    assert selected.total_time == pytest.approx(min(alternatives), abs=1e-8)
    assert selected == run()
    assert selected.pit_stops == 1


def test_fresh_table_cache_is_reused_after_stops_and_tracks_physics(monkeypatch):
    from f1sim.simulation.pit_strategy import _fresh_tables

    _fresh_tables.cache_clear()
    state = _state("A", 1, 0.0)
    track = _track()
    decision = _plan(state, track, 25, 3)
    _plan(state, track, 24, 2)
    _plan(state, track, 23, 1)
    assert _fresh_tables.cache_info().misses == 1
    assert _fresh_tables.cache_info().maxsize == 64
    replacement = TIRE_COMPOUNDS[TireCompound.HARD].model_copy(deep=True)
    replacement.degradation_rate = 0.0
    monkeypatch.setitem(TIRE_COMPOUNDS, TireCompound.HARD, replacement)
    updated = _plan(state, track, 25, 3)
    assert _fresh_tables.cache_info().misses == 2
    assert updated.pit_now_cost < decision.pit_now_cost
