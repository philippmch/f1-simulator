"""Two-hour racing-clock expiry announces the following lap as the finish."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator


def run_controlled(
    monkeypatch, pace=1800, scheduled=10, *, flags=(), retire=None, mandatory=False,
    neutralization=None,
):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="t", name="T", country="T", total_laps=scheduled, base_lap_time=90)
    simulator = RaceSimulator(np.random.default_rng(42))
    calls, planning, weather_calls, fits = [], [], [], []
    original_should_pit = simulator._should_pit

    def should_pit(state, states, plan_track, lap, *args, **kwargs):
        planning.append((lap, plan_track.total_laps))
        if mandatory and lap == 5:
            state.pit_stops = 3  # Mandatory correction survives exhausted elective budget.
            return original_should_pit(state, states, plan_track, lap, *args, **kwargs)
        return False

    def lap_time(**kwargs):
        calls.append((kwargs["lap_number"], kwargs["total_laps"], kwargs["track"].total_laps))
        return pace

    def events(lap, drivers, **kwargs):
        output = [RaceEvent(EventType.RED_FLAG, lap)] if lap in flags else []
        if neutralization is not None:
            field = ("safety_car_active" if neutralization == EventType.SAFETY_CAR
                     else "vsc_active")
            setattr(simulator.event_manager, field, lap == 1)
            if lap == 1:
                output.append(RaceEvent(neutralization, lap))
        if lap == retire:
            drivers[0].dnf = True
            output.append(RaceEvent(EventType.MECHANICAL_FAILURE, lap, ["A"]))
        return output

    def evolve(weather, rng):
        weather_calls.append(1)
        rng.random()
        return weather.project_surface()

    original_fit = simulator._fit_red_flag_tires

    def fit(states, weather, plan_track, lap, **kwargs):
        fits.append((lap, plan_track.total_laps))
        assert kwargs.get("physical_total_laps", plan_track.total_laps) == scheduled
        return original_fit(states, weather, plan_track, lap, **kwargs)

    monkeypatch.setattr(simulator, "_should_pit", should_pit)
    monkeypatch.setattr(simulator, "_fit_red_flag_tires", fit)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lap_time)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda *args: 3)
    monkeypatch.setattr(simulator.event_manager, "process_lap", events)
    monkeypatch.setattr(Weather, "evolve", evolve)
    (result,) = simulator.simulate_race(
        [driver], {"A": car}, track, Weather(), ["A"], starting_tires={"A": TireCompound.SOFT}
    )
    return result, simulator, track, calls, planning, weather_calls, fits


@pytest.mark.parametrize(
    "pace,scheduled,completed,limited",
    [
        (1800, 10, 5, True),
        (1900, 10, 5, True),
        (1799, 10, 6, True),
        (1800, 4, 4, False),
        (1800, 5, 5, False),
        (90, 10, 10, False),
    ],
)
def test_time_limit_threshold_and_scheduled_cap(monkeypatch, pace, scheduled, completed, limited):
    result, simulator, track, calls, planning, weather, _ = run_controlled(
        monkeypatch,
        pace,
        scheduled,
    )
    assert result.laps_completed == completed
    assert result.race_time_limited is limited
    assert len(calls) == completed and len(weather) == completed - 1
    assert all(total == original == scheduled for _, total, original in calls)
    assert track.total_laps == scheduled
    if limited:
        assert planning[-1] == (completed, completed)
        assert planning[-2][1] == scheduled
    expected_rng = np.random.default_rng(42)
    expected_rng.random(completed - 1)
    assert simulator.rng.random() == expected_rng.random()
    assert result.classified


def test_announced_final_lap_forces_distinct_compound_even_after_budget(monkeypatch):
    result, _, _, _, _, _, _ = run_controlled(monkeypatch, mandatory=True)
    assert result.laps_completed == 5 and result.race_time_limited
    assert result.pit_laps == [5]
    assert len(set(result.strategy)) == 2


def test_red_flag_restart_uses_announced_horizon_and_finish_has_no_refit(monkeypatch):
    result, _, _, _, _, weather, fits = run_controlled(monkeypatch, flags=(4, 5))
    assert result.laps_completed == 5
    assert fits == [(4, 5)]
    assert len(weather) == 4
    assert len(result.strategy) == 2


def test_all_retired_after_announcement_has_no_timed_finisher(monkeypatch):
    result, _, _, _, _, weather, _ = run_controlled(monkeypatch, retire=5)
    assert result.laps_completed == 4
    assert not result.race_time_limited
    assert result.points_awarded == 0
    assert len(weather) == 4


@pytest.mark.parametrize("flags,expected", [((2, 4), 0), ((3, 4), 19)])
def test_points_require_two_consecutive_green_laps(monkeypatch, flags, expected):
    result, *_ = run_controlled(monkeypatch, flags=flags)
    # Five of ten laps gives the 50%-75% winner award, but only after two
    # consecutive green laps. A red-flag deployment breaks the streak.
    assert result.points_awarded == expected


@pytest.mark.parametrize("event_type", [EventType.SAFETY_CAR, EventType.VIRTUAL_SAFETY_CAR])
@pytest.mark.parametrize("laps,expected", [(3, 0), (4, 25)])
def test_lap_that_starts_neutralized_cannot_count_as_green(monkeypatch, event_type, laps, expected):
    result, *_ = run_controlled(
        monkeypatch, pace=90, scheduled=laps, neutralization=event_type,
    )
    assert result.points_awarded == expected


def test_all_retired_keeps_existing_early_stop(monkeypatch):
    result, _, _, calls, _, weather, fits = run_controlled(monkeypatch, retire=4, flags=(4,))
    assert result.laps_completed == 3
    assert len(calls) == 4 and len(weather) == 3
    assert fits == [] and not result.race_time_limited


def test_classification_uses_time_limited_winner_distance(monkeypatch):
    drivers = [Driver(id=name, name=name, team_id=name) for name in ("A", "B")]
    cars = {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers}
    track = Track(id="t", name="T", country="T", total_laps=10, base_lap_time=90)
    simulator = RaceSimulator(np.random.default_rng(42))
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator, "_process_overtakes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lambda **kwargs: 1800)

    def events(lap, drivers, **kwargs):
        if lap == 5:
            next(d for d in drivers if d.id == "B").dnf = True
            return [RaceEvent(EventType.MECHANICAL_FAILURE, lap, ["B"])]
        return []

    monkeypatch.setattr(simulator.event_manager, "process_lap", events)
    results = simulator.simulate_race(
        drivers,
        cars,
        track,
        Weather(change_probability=0),
        ["A", "B"],
        starting_tires={d.id: TireCompound.INTERMEDIATE for d in drivers},
    )
    assert [r.laps_completed for r in results] == [5, 4]
    assert all(r.classified and r.race_time_limited for r in results)


def test_isolated_opening_projection_uses_same_finish_rule_and_original_fuel(monkeypatch, request):
    from f1sim.simulation.lap import LapSimulator
    from f1sim.simulation.opening_strategy import _policy_path_cost
    from f1sim.simulation.race import TeamStrategyArchetype
    from f1sim.simulation.rain_strategy import _fresh_future, _plan, _running_row

    for cached in (_fresh_future, _plan, _running_row):
        cached.cache_clear()
        request.addfinalizer(cached.cache_clear)

    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="t", name="T", country="T", total_laps=10, base_lap_time=90)
    weather = Weather(track_wetness=0.3, rain_intensity=0.3)
    simulator = RaceSimulator(np.random.default_rng(42))
    calls = []

    def running(self, *args, **kwargs):
        # The rain planner also evaluates isolated future laps. Count only
        # the opening policy's committed running on the original track.
        assert args[6] == 10
        if args[2] is track:
            calls.append((args[6], args[2].total_laps))
        return 1800

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", running)
    cost = _policy_path_cost(
        driver,
        car,
        track,
        weather,
        TeamStrategyArchetype.BALANCED,
        simulator.strategy_tuning,
        simulator.strategy_profiles,
        TireCompound.INTERMEDIATE,
        0,
    )
    assert cost == 9000
    assert calls == [(10, 10)] * 5
