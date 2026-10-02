"""Prescribed atmosphere uses global cadence and existing surface physics."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator
from f1sim.simulation.surface_projection import projected_surfaces, suffix_weather_intervals
from f1sim.simulation.weather_schedule import (
    WeatherForecastContext,
    validate_weather_schedule,
)

SCHEDULE = [{"lap": 2, "rain_intensity": .8, "condition": "heavy_rain"},
            {"lap": 4, "rain_intensity": 0}]


@pytest.mark.parametrize("value", [
    {}, (), [{"lap": 1, "rain_intensity": .1}], [{"lap": True, "rain_intensity": .1}],
    [{"lap": 2.0, "rain_intensity": .1}], [{"lap": "2", "rain_intensity": .1}],
    [{"lap": 2, "rain_intensity": True}], [{"lap": 2, "rain_intensity": "0.2"}],
    [{"lap": 2, "rain_intensity": float("nan")}],
    [{"lap": 2, "rain_intensity": float("inf")}],
    [{"lap": 2, "rain_intensity": -.1}], [{"lap": 2, "rain_intensity": 1.1}],
    [{"lap": 2}], [{"rain_intensity": .1}],
    [{"lap": 2, "rain_intensity": .1, "other": 1}],
    [{"lap": 2, "rain_intensity": .1, "condition": "rain"}],
    [{"lap": 2, "rain_intensity": .1}, {"lap": 2, "rain_intensity": .2}],
    [{"lap": 3, "rain_intensity": .1}, {"lap": 2, "rain_intensity": .2}],
    [{"lap": 5, "rain_intensity": .1}], [None],
])
def test_strict_schedule_validation(value):
    with pytest.raises(ValueError, match="weather_schedule"):
        validate_weather_schedule(value, total_laps=4)


def test_validation_isolates_and_canonicalizes_input():
    source = [{"lap": 2, "rain_intensity": 1, "condition": WeatherCondition.HEAVY_RAIN}]
    result = validate_weather_schedule(source)
    assert result == [{"lap": 2, "rain_intensity": 1., "condition": "heavy_rain"}]
    assert type(result[0]["rain_intensity"]) is float
    result[0]["lap"] = 8
    assert source[0]["lap"] == 2
    context = WeatherForecastContext.from_schedule(source)
    source[0]["rain_intensity"] = 0
    assert context.schedule[0][1] == 1
    with pytest.raises(FrozenInstanceError):
        context.leading_lap = 3
    assert validate_weather_schedule(None) == validate_weather_schedule([]) == []


def test_hand_stepped_surface_and_suffix_rebase():
    weather = Weather(track_wetness=.1)
    context = WeatherForecastContext.from_schedule(SCHEDULE)
    path = projected_surfaces(weather, 5, (0, 1, 2, 3, 5), forecast_context=context)
    assert [surface.rain_intensity for surface in path] == [0, .8, .8, 0, 0]
    assert [surface.track_wetness for surface in path] == pytest.approx(
        [.1, .24, .352, .322, .262])
    assert path[0].condition == WeatherCondition.DRY
    assert all(surface.condition == WeatherCondition.HEAVY_RAIN for surface in path[1:])
    from f1sim.simulation.surface_projection import normalize_weather_intervals

    cadence = normalize_weather_intervals(5, (0, 1, 2, 3, 5), forecast_context=context)
    suffix = suffix_weather_intervals(cadence, 2, path[2])
    assert suffix.context.leading_lap == 3
    assert projected_surfaces(path[2], 3, suffix) == path[2:]
    assert weather.track_wetness == .1 and weather.rain_intensity == 0


def test_dry_equilibrium_keeps_future_external_updates():
    context = WeatherForecastContext.from_schedule([{"lap": 4, "rain_intensity": 1}])
    path = projected_surfaces(Weather(), 3, (0, 3, 5), forecast_context=context)
    assert [surface.track_wetness for surface in path] == pytest.approx([0, .2, .488])


def race_inputs(laps=5):
    return ([Driver(id="A", name="A", team_id="A")],
            {"A": Car(team_id="A", team_name="A")},
            Track(id="T", name="T", country="T", total_laps=laps, base_lap_time=90),
            Weather(track_wetness=.1, change_probability=1))


@pytest.mark.parametrize("chronological", [False, True])
def test_execution_schedule_and_reuse_reset(monkeypatch, chronological):
    simulator = RaceSimulator(np.random.default_rng(7))
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **kw: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(simulator, "_should_pit", lambda *a, **kw: False)
    monkeypatch.setattr(Weather, "evolve", lambda *a: pytest.fail("random atmosphere sampled"))
    run = ChronologicalRace(simulator).run if chronological else simulator.simulate_race
    source = deepcopy(SCHEDULE)
    run(*race_inputs(), ["A"], starting_tires={"A": TireCompound.SOFT},
        weather_schedule=source)
    assert [row["lap"] for row in simulator.weather_history] == [1, 2, 3, 4, 5]
    assert [row["track_wetness"] for row in simulator.weather_history] == pytest.approx(
        [.1, .24, .352, .322, .292])
    assert source == SCHEDULE
    run(*race_inputs(laps=1), ["A"], starting_tires={"A": TireCompound.SOFT})
    assert simulator.weather_forecast_context is None


@pytest.mark.parametrize("chronological", [False, True])
def test_empty_schedule_matches_default_race_and_rng(chronological):
    def execute(options):
        simulator = RaceSimulator(np.random.default_rng(19))
        run = ChronologicalRace(simulator).run if chronological else simulator.simulate_race
        results = run(*race_inputs(), ["A"], **options)
        return results, simulator.weather_history, simulator.rng.bit_generator.state

    assert execute({}) == execute({"weather_schedule": []})


@pytest.mark.parametrize("chronological", [False, True])
def test_invalid_schedule_precedes_mutation_and_rng(chronological):
    args = race_inputs()
    args[0][0].current_tire_laps = 7
    simulator = RaceSimulator(np.random.default_rng(19))
    before = deepcopy(simulator.rng.bit_generator.state)
    run = ChronologicalRace(simulator).run if chronological else simulator.simulate_race
    with pytest.raises(ValueError, match="weather_schedule"):
        run(*args, ["A"], weather_schedule=[{"lap": 6, "rain_intensity": 1}])
    assert args[0][0].current_tire_laps == 7
    assert simulator.rng.bit_generator.state == before


@pytest.mark.parametrize("chronological", [False, True])
@pytest.mark.parametrize("restart,retire", [(False, False), (True, False), (False, True)])
def test_shared_cadence_with_lapped_car_restart_and_leader_handoff(
    monkeypatch, chronological, restart, retire,
):
    simulator = RaceSimulator(np.random.default_rng(5))
    engine = ChronologicalRace(simulator, red_flag_pause_seconds=600)
    drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
    cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
    track = Track(id="T", name="T", country="T", total_laps=6, base_lap_time=90)
    snapshots = []

    def physics(driver, car, track, tire, weather, lap_number, total_laps, **kwargs):
        context = simulator.weather_forecast_context
        snapshots.append((driver.id, lap_number, context.leading_lap,
                          weather.model_copy(deep=True)))
        return 90 if driver.id == "A" else 250

    def failure(driver, car, track, lap, weather):
        if retire and driver.id == "A" and lap == 3:
            driver.dnf = True
            driver.dnf_reason = "Scripted retirement"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, ["A"])
        return None

    control = simulator.event_manager
    if restart:
        control.set_forced_red_flag(2)
    monkeypatch.setattr(control, "_check_mechanical_failure", failure)
    monkeypatch.setattr(control, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *a: None)
    monkeypatch.setattr(control, "_check_red_flag_conditions", lambda *a: None)
    monkeypatch.setattr(control, "_check_severe_weather_red_flag", lambda *a: None)
    monkeypatch.setattr(simulator, "_should_pit", lambda *a, **kw: False)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", physics)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **kw: (True, False))
    run = engine.run if chronological else simulator.simulate_race
    run(drivers, cars, track, Weather(track_wetness=.1), ["A", "B"],
        starting_tires={key: TireCompound.SOFT for key in "AB"}, weather_schedule=SCHEDULE)
    history = simulator.weather_history
    assert [row["lap"] for row in history] == list(range(1, len(history) + 1))
    expected_wetness = [.1, .24, .352]
    expected_wetness.extend(max(0, .352 - .03 * n) for n in range(1, len(history) - 2))
    assert [row["track_wetness"] for row in history] == pytest.approx(expected_wetness)
    for driver, own_lap, shared_lap, surface in snapshots:
        assert surface.track_wetness == pytest.approx(history[shared_lap - 1]["track_wetness"])
    if restart:
        assert (engine.suspensions if chronological else simulator.suspensions)
    if chronological and not restart:
        assert any(driver == "B" and shared > own for driver, own, shared, _ in snapshots)
    if retire:
        assert drivers[0].dnf


@pytest.mark.parametrize("schedule,lap", [([], 0), ([], True), ([], 1.0), ([], "1"),
                                          ([[2, .5, None]], 1), (((2, .5, "rain"),), 1),
                                          (((True, .5, None),), 1)])
def test_forecast_context_rejects_mutable_or_malformed_values(schedule, lap):
    with pytest.raises(ValueError, match="forecast context"):
        WeatherForecastContext(schedule, lap)


@pytest.mark.parametrize("updates", [-1, True, 1.0, "1"])
def test_context_rejects_invalid_update_counts(updates):
    with pytest.raises(ValueError, match="forecast updates"):
        WeatherForecastContext(()).advanced(updates)


def test_context_method_patch_bypasses_shared_surface_cache(monkeypatch):
    context = WeatherForecastContext.from_schedule(SCHEDULE)
    baseline = projected_surfaces(Weather(), 3, forecast_context=context)
    original = WeatherForecastContext.project_next

    def altered(context, weather):
        value = original(context, weather)
        value.track_wetness += .01
        return value

    monkeypatch.setattr(WeatherForecastContext, "project_next", altered)
    changed = projected_surfaces(Weather(), 3, forecast_context=context)
    assert changed[1].track_wetness == pytest.approx(baseline[1].track_wetness + .01)
    assert changed[2].track_wetness != baseline[2].track_wetness
