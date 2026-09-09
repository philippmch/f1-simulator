"""Storms and manual suspension can supersede an active neutralization."""

import numpy as np
import pytest

from f1sim.models import Track, Weather
from f1sim.simulation.events import EventManager, EventType


def manager(mode, remaining, seed=2):
    control = EventManager(np.random.default_rng(seed))
    setattr(control, "safety_car_active" if mode == "sc" else "vsc_active", True)
    setattr(control, "safety_car_laps_remaining" if mode == "sc" else "vsc_laps_remaining",
            remaining)
    return control


def process(control, lap, weather):
    track = Track(id="t", name="T", country="T", total_laps=30, base_lap_time=90)
    return control.process_lap(lap, [], {}, track, weather)


@pytest.mark.parametrize("mode", ["sc", "vsc"])
@pytest.mark.parametrize("remaining", [1, 4])
@pytest.mark.parametrize("forced", [False, True])
def test_suspension_supersedes_neutralization_including_ending_lap(mode, remaining, forced):
    control = manager(mode, remaining)
    weather = Weather(track_wetness=0.9, rain_intensity=0.9)
    if forced:
        control.set_forced_red_flag(7)
    events = process(control, 7, weather)
    assert [event.event_type for event in events] == [EventType.RED_FLAG]
    assert events == control.events
    assert events[0].lap == 7
    assert control.red_flag_deployments == 1
    assert control.red_flag_active
    assert not control.safety_car_active and not control.vsc_active
    assert control.safety_car_laps_remaining == control.vsc_laps_remaining == 0
    assert not control.sc_just_ended and not control.sc_restart_lap
    assert control.sc_restart_lap_number is None
    expected = np.random.default_rng(2)
    if not forced:
        expected.random()  # One storm decision; no background or SC redraw.
    assert control.rng.random() == expected.random()
    control.end_red_flag()
    assert control.red_flag_restart_lap_number == 8


@pytest.mark.parametrize("mode", ["sc", "vsc"])
def test_declined_storm_is_not_rerolled_until_conditions_clear(mode):
    control = manager(mode, 20, seed=0)  # First storm draw declines.
    severe = Weather(track_wetness=0.9, rain_intensity=0.9)
    assert process(control, 1, severe) == []
    assert process(control, 2, severe) == []
    assert process(control, 3, Weather(track_wetness=0.7, rain_intensity=0.6)) == []
    assert [event.event_type for event in process(control, 4, severe)] == [EventType.RED_FLAG]
    expected = np.random.default_rng(0)
    expected.random(2)
    assert control.rng.random() == expected.random()


@pytest.mark.parametrize("mode", ["sc", "vsc"])
def test_nonsevere_neutralized_lap_has_no_new_hazard_draws(mode):
    control = manager(mode, 1)
    assert process(control, 4, Weather()) == []
    assert control.rng.random() == np.random.default_rng(2).random()


def test_live_escalation_fits_once_and_restarts_without_paid_stop(monkeypatch):
    from f1sim.models import Car, Driver, TireCompound
    from f1sim.simulation.race import RaceSimulator

    simulator = RaceSimulator(np.random.default_rng(2))
    control = simulator.event_manager
    control.set_forced_safety_car(1)
    monkeypatch.setattr(control, "_check_mechanical_failure", lambda *args: None)
    monkeypatch.setattr(control, "_check_random_incident", lambda *args, **kwargs: None)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *args: None)
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lambda **kwargs: 90)
    severe = Weather(track_wetness=0.96, rain_intensity=0.9)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: severe.model_copy())
    fits = []
    original_fit = simulator._fit_red_flag_tires

    def fit(states, weather, track, lap):
        fits.append(lap)
        return original_fit(states, weather, track, lap)

    monkeypatch.setattr(simulator, "_fit_red_flag_tires", fit)
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="t", name="T", country="T", total_laps=3, base_lap_time=90)
    (result,) = simulator.simulate_race(
        [driver], {"A": car}, track, Weather(), ["A"],
        starting_tires={"A": TireCompound.INTERMEDIATE},
    )
    assert [event.event_type for event in control.events] == [
        EventType.SAFETY_CAR, EventType.RED_FLAG,
    ]
    assert fits == [2]
    assert control.red_flag_restart_lap_number == 3
    assert control.sc_restart_lap_number is None
    assert result.pit_stops == 0 and result.pit_laps == []
    assert result.strategy == [TireCompound.INTERMEDIATE, TireCompound.WET]
    assert result.laps_completed == 3
