"""A drying track does not justify buying an intermediate set just before slicks."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_drying_race_skips_redundant_intermediate_stop(monkeypatch, engine):
    def run(legacy_bypass):
        elapsed = 0

        def evolve(weather, rng):
            nonlocal elapsed
            elapsed += 1
            if elapsed == 14:
                return weather.model_copy(update={"rain_intensity": 0})
            return weather.project_surface()

        monkeypatch.setattr(Weather, "evolve", evolve)
        driver = Driver(id="A", name="A", team_id="A")
        car = Car(team_id="A", team_name="A")
        track = Track(id="t", name="T", country="T", total_laps=50, base_lap_time=90)
        weather = Weather(condition=WeatherCondition.LIGHT_RAIN, rain_intensity=.3,
                          track_wetness=.3, change_probability=0)
        sim = RaceSimulator(np.random.default_rng(0))
        sim.event_manager.process_lap = lambda *a, **k: []
        sim.event_manager._check_mechanical_failure = lambda *a: None
        sim.event_manager._check_random_incident = lambda *a: None
        actual_lap = sim.lap_simulator.calculate_lap_time

        def mean_lap(*args, **kwargs):
            kwargs["sample_variation"] = False
            return actual_lap(*args, **kwargs)

        sim.lap_simulator.calculate_lap_time = mean_lap
        sim.lap_simulator.calculate_pit_stop_time = expected_stationary_time
        sim._infer_team_strategy = lambda *a: TeamStrategyArchetype.BALANCED
        sim._plan_pit_lap_options = lambda *a: [[17]]
        if legacy_bypass:
            current_check = sim._weather_stop_can_pay

            def old_check(state, track, weather, lap, *args, **kwargs):
                surface = weather
                for _ in range(track.total_laps - lap + 1):
                    if surface.tire_mismatch(state.current_tire.compound) == "critical":
                        return True
                    surface = surface.project_surface()
                return current_check(state, track, weather, lap, *args, **kwargs)

            sim._weather_stop_can_pay = old_check
        execute = sim.simulate_race if engine == "standard" else ChronologicalRace(sim).run
        return execute([driver], {"A": car}, track, weather, ["A"],
                       starting_tires={"A": TireCompound.INTERMEDIATE})[0]

    corrected, previous = run(False), run(True)
    assert corrected.laps_completed == previous.laps_completed == 50
    assert corrected.total_time < previous.total_time
    assert corrected.pit_stops == previous.pit_stops - 1
    assert len(corrected.pit_laps) == 1 and corrected.pit_laps[0] > 17
    assert previous.strategy[:2] == ["intermediate", "intermediate"]
    assert corrected.strategy[0] == "intermediate"
    assert corrected.strategy[1] in {"soft", "medium", "hard"}
