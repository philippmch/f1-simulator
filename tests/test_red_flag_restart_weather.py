"""Free red-flag refits use the weather actually consumed by the restart lap."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator


def controlled_race(monkeypatch, initial, start, *, forced=None, worsening=False,
                    flag_lap=1, retire=False, puncture=False):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="t", name="T", country="T", total_laps=2, base_lap_time=90)
    simulator = RaceSimulator(np.random.default_rng(42))
    observed, weather_calls, refits = [], [], []
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    actual_time = simulator.lap_simulator.calculate_lap_time

    def lap_time(**kwargs):
        observed.append((kwargs["tire"].compound, kwargs["weather"].track_wetness))
        return actual_time(**kwargs, sample_variation=False)

    def evolve(weather, rng):
        weather_calls.append(weather.track_wetness)
        # The existing single weather reaction draw, with no condition change.
        rng.random()
        return Weather(track_wetness=0.9, rain_intensity=0.9) if worsening else (
            weather.project_surface()
        )

    def events(lap, drivers, **kwargs):
        if lap != flag_lap:
            return []
        output = [RaceEvent(EventType.RED_FLAG, lap)]
        if puncture:
            output.append(RaceEvent(EventType.PUNCTURE, lap, ["A"], forces_pit_stop=True))
        if retire:
            drivers[0].dnf = True
            output.append(RaceEvent(EventType.MECHANICAL_FAILURE, lap, ["A"]))
        return output

    original_choose = simulator._choose_red_flag_tire

    def choose(state, weather, track, lap):
        assert not simulator.event_manager.red_flag_active
        refits.append(weather.track_wetness)
        return forced if forced is not None else original_choose(state, weather, track, lap)

    monkeypatch.setattr(simulator, "_choose_red_flag_tire", choose)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lap_time)
    monkeypatch.setattr(Weather, "evolve", evolve)
    monkeypatch.setattr(simulator.event_manager, "process_lap", events)
    result, = simulator.simulate_race([driver], {"A": car}, track, initial, ["A"],
                                      starting_tires={"A": start})
    return result, simulator, observed, weather_calls, refits


@pytest.mark.parametrize("wetness,start,expected", [
    (0.21, TireCompound.INTERMEDIATE, TireCompound.SOFT),
    (0.71, TireCompound.WET, TireCompound.INTERMEDIATE),
])
def test_drying_restart_matches_correct_free_set_and_preserves_weather_draw(
    monkeypatch, wetness, start, expected,
):
    initial = Weather(track_wetness=wetness, change_probability=0)
    result, sim, ran, calls, refits = controlled_race(monkeypatch, initial, start, puncture=True)
    assert ran == [(start, wetness), (expected, pytest.approx(wetness - 0.03))]
    assert calls == [wetness]
    assert refits == [pytest.approx(wetness - 0.03)]
    assert result.pit_stops == 0 and result.pit_laps == []
    assert result.strategy == [start.value, expected.value]
    expected_rng = np.random.default_rng(42)
    expected_rng.random()
    assert sim.rng.random() == expected_rng.random()
    forced, _, _, _, _ = controlled_race(monkeypatch, initial, start, forced=expected)
    assert result.total_time == pytest.approx(forced.total_time)
    old, _, _, _, _ = controlled_race(monkeypatch, initial, start, forced=start)
    assert result.total_time < old.total_time


def test_worsening_restart_observes_new_rain_before_free_refit(monkeypatch):
    result, _, ran, calls, refits = controlled_race(
        monkeypatch, Weather(track_wetness=0.3), TireCompound.INTERMEDIATE, worsening=True,
    )
    assert ran[-1] == (TireCompound.WET, 0.9)
    assert refits == [0.9] and len(calls) == 1
    assert result.pit_stops == 0


@pytest.mark.parametrize("flag_lap,retire", [(2, False), (1, True)])
def test_no_restart_means_no_free_fit_or_extra_weather(monkeypatch, flag_lap, retire):
    result, _, _, calls, refits = controlled_race(
        monkeypatch, Weather(), TireCompound.SOFT, flag_lap=flag_lap, retire=retire,
    )
    assert refits == []
    assert len(calls) == (0 if retire else 1)
    assert result.strategy == ["soft"]
