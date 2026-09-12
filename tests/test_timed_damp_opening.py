"""A changed deadline must select the best executed damp opening policy."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation import opening_strategy, race_timing
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_warmed_damp_scores_follow_new_deadline_in_actual_execution(monkeypatch, engine):
    driver = Driver(id="a", name="A", team_id="a")
    car = Car(team_id="a", team_name="A")
    track = Track(id="t", name="T", country="T", total_laps=10, base_lap_time=90)
    weather = Weather(track_wetness=.1, rain_intensity=.1, change_probability=0)
    warm = RaceSimulator(np.random.default_rng(0))
    opening_strategy._cached_policy_costs.cache_clear()
    opening_strategy.opening_policy_costs(
        driver, car, track, weather, TeamStrategyArchetype.BALANCED,
        warm.strategy_tuning, warm.strategy_profiles,
    )
    monkeypatch.setattr(race_timing, "RACING_TIME_LIMIT_SECONDS", 200.)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())

    def run(compound=None):
        simulator = RaceSimulator(np.random.default_rng(0))
        simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
        simulator.event_manager.process_lap = lambda *args, **kwargs: []
        simulator.event_manager._check_mechanical_failure = lambda *args, **kwargs: None
        simulator.event_manager._check_random_incident = lambda *args, **kwargs: None
        running = simulator.lap_simulator.calculate_lap_time
        fuel_distances = []

        def mean_lap(*args, **kwargs):
            fuel_distances.append(kwargs.get("total_laps", args[6] if args else None))
            return running(*args, **kwargs, sample_variation=False)

        simulator.lap_simulator.calculate_lap_time = mean_lap
        simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
        execute = (simulator.simulate_race if engine == "standard"
                   else ChronologicalRace(simulator).run)
        result, = execute(
            [driver.model_copy(deep=True)], {"a": car.model_copy(deep=True)},
            track, weather, ["a"], starting_tires={"a": compound} if compound else None,
        )
        assert set(fuel_distances) == {10}
        return result

    selected = run()
    alternatives = [run(compound) for compound in opening_strategy.OPENING_CANDIDATES]
    best = min(alternatives, key=lambda row: (-row.laps_completed, row.total_time))
    assert selected.race_time_limited and selected.laps_completed == 4
    assert selected.strategy == best.strategy == ["intermediate"]
    assert selected.total_time == pytest.approx(best.total_time, rel=0., abs=1e-9)
    # Soft also completes four laps, but pays for a dry compound correction.
    assert alternatives[1].laps_completed == selected.laps_completed
    assert alternatives[1].total_time - selected.total_time > 9.8
    assert all(row.laps_completed == 3 for row in alternatives[2:])
