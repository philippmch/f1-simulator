"""Opening selection is judged by complete, timed execution in both engines."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import SLICKS, expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_timed_dry_opening_matches_best_executed_alternative(monkeypatch, engine):
    def run(compound=None):
        driver = Driver(id="d", name="Driver", team_id="t")
        car = Car(team_id="t", team_name="Team")
        track = Track(id="t", name="Track", country="Test", total_laps=90,
                      base_lap_time=110, pit_lane_delta=22)
        simulator = RaceSimulator(np.random.default_rng(7))
        monkeypatch.setattr(simulator, "_infer_team_strategy",
                            lambda *args: TeamStrategyArchetype.BALANCED)
        monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *args, **kwargs: [])
        monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure",
                            lambda *args, **kwargs: None)
        monkeypatch.setattr(simulator.event_manager, "_check_random_incident",
                            lambda *args, **kwargs: None)
        running = simulator.lap_simulator.calculate_lap_time
        monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                            lambda *args, **kwargs: running(
                                *args, **kwargs, sample_variation=False,
                            ))
        monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                            expected_stationary_time)
        monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())
        execute = (simulator.simulate_race if engine == "standard"
                   else ChronologicalRace(simulator).run)
        result, = execute([driver], {"t": car}, track, Weather(), ["d"],
                          starting_tires={"d": compound} if compound else None)
        return result

    alternatives = {compound: run(compound) for compound in SLICKS}
    selected = run()
    assert selected.strategy[0] == TireCompound.HARD.value
    assert selected.laps_completed == 65 and selected.race_time_limited
    assert all(result.laps_completed == 65 for result in alternatives.values())
    assert selected.total_time == pytest.approx(
        min(result.total_time for result in alternatives.values()), abs=1e-8,
    )
    assert alternatives[TireCompound.MEDIUM].total_time - selected.total_time > 9.8
