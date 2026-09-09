"""Execution selection reaches real workers without changing seeded inputs."""

from dataclasses import asdict

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner, _run_single_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.events import EventManager
from f1sim.simulation.execution import RACE_ENGINES, validate_race_engine
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.overtaking import OvertakingModel
from f1sim.simulation.race import RaceSimulator


def inputs(laps=5):
    drivers = [Driver(id=str(i), name=str(i), team_id=str(i)) for i in range(2)]
    cars = {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers}
    track = Track(id="t", name="T", country="T", total_laps=laps, base_lap_time=90)
    return drivers, cars, track, Weather(change_probability=0)


def serialized(values):
    drivers, cars, track, weather = values
    return ([d.model_dump() for d in drivers],
            {k: c.model_dump() for k, c in cars.items()},
            track.model_dump(), weather.model_dump())


@pytest.mark.parametrize("value", [None, True, 1, [], {}, "", "STANDARD", "other"])
def test_invalid_engine_rejected_at_construction(value):
    with pytest.raises(ValueError, match="race_engine"):
        MonteCarloRunner(*inputs(), race_engine=value)
    with pytest.raises(ValueError, match="race_engine"):
        validate_race_engine(value)


def test_default_standard_preserves_legacy_worker_results():
    values = inputs()
    default = MonteCarloRunner(*values, seed=42).run(1, parallel=False)
    explicit = MonteCarloRunner(*values, seed=42, race_engine="standard").run(
        1, parallel=False,
    )
    legacy_race, legacy_quali, _ = _run_single_simulation((*serialized(values), 42))
    assert default == explicit
    shared = MonteCarloRunner(*values, seed=42, rng_policy="shared_v1").run(
        1, parallel=False,
    )
    assert shared.race_results == [legacy_race]
    assert shared.qualifying_results == [legacy_quali]
    assert default.race_engine == "standard"
    assert RACE_ENGINES == ("standard", "chronological")


def test_chronological_real_process_pool_matches_sequential_and_preserves_inputs():
    values = inputs()
    before = serialized(values)
    runner = MonteCarloRunner(*values, seed=71, race_engine="chronological")
    sequential = runner.run(3, parallel=False)
    parallel = runner.run(3, parallel=True, max_workers=2)
    repeated = runner.run(3, parallel=False)
    assert sequential == repeated
    parallel_data = asdict(parallel)
    parallel_data.update(parallel=False, max_workers=None)
    assert asdict(sequential) == parallel_data
    assert sequential.race_engine == "chronological"
    assert serialized(values) == before
    standard = MonteCarloRunner(*values, seed=71).run(3, parallel=False)
    assert standard.qualifying_results == sequential.qualifying_results


def test_selected_worker_executes_lapped_finish_instead_of_full_distance(monkeypatch):
    values = inputs(laps=10)
    before = serialized(values)
    monkeypatch.setattr(RaceSimulator, "_should_pit", lambda *a, **kw: False)
    monkeypatch.setattr(OvertakingModel, "attempt_overtake", lambda *a, **kw: (True, False))
    for method in ("_check_mechanical_failure", "_check_random_incident",
                   "_deploy_safety_measure"):
        monkeypatch.setattr(EventManager, method, lambda *a, **kw: None)

    def constant_lap(self, driver, *args, **kwargs):
        return 90 if driver.id == "0" else 110

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", constant_lap)
    chronological, quali, _ = _run_single_simulation((*before, 42, "chronological"))
    standard, standard_quali, _ = _run_single_simulation((*before, 42, "standard"))
    assert quali == standard_quali
    chrono_by_id = {row.driver_id: row for row in chronological}
    assert chrono_by_id["0"].laps_completed == 10
    assert chrono_by_id["1"].laps_completed == 9
    assert chrono_by_id["1"].total_time == 990
    assert {row.laps_completed for row in standard} == {10}
    assert serialized(values) == before
