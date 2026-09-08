"""Replay saved inputs offline using the installed model and dependencies.

Exact reproduction requires matching model and dependency versions. Saved runtime
provenance is informational; replay never installs or executes saved code.
"""

import json
from pathlib import Path

from f1sim.analysis.montecarlo import MonteCarloRunner, SimulationResults
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.execution import validate_race_engine


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def replay_saved_simulation(
    path: str | Path, simulation: int = 1, scenario: str | None = None,
) -> SimulationResults:
    """Replay a one-based trial from exported statistics, without provider access.

    Runtime provenance does not change the installed implementation. Exact results
    require the same model and dependency versions as the original run.
    """
    runner, count = _load_saved_runner(path, scenario)
    index = _integer(simulation, "simulation", 1)
    if index > count:
        raise ValueError(f"simulation must be between 1 and {count}")
    runner.base_seed += index - 1
    return runner.run(1, parallel=False)


def _load_saved_runner(
    path: str | Path, scenario: str | None = None,
) -> tuple[MonteCarloRunner, int]:
    """Validate saved models and metadata without running a simulation."""
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise ValueError("Saved statistics must be a JSON object")
    if "scenarios" in saved:
        scenarios = saved["scenarios"]
        if not isinstance(scenarios, dict) or not scenarios:
            raise ValueError("Saved scenarios must be a nonempty object")
        if scenario is None:
            if len(scenarios) != 1:
                raise ValueError("Multiple saved scenarios; select one with --scenario")
            scenario = next(iter(scenarios))
        if not isinstance(scenario, str) or scenario not in scenarios:
            raise ValueError("Unknown saved scenario")
        saved = scenarios[scenario]
        if not isinstance(saved, dict):
            raise ValueError("Saved scenario must be an object")
        metadata = saved
    else:
        if scenario is not None:
            raise ValueError("This statistics file has no named scenarios")
        metadata = saved.get("metadata")
    inputs = saved.get("simulation_inputs")
    if inputs is None:
        raise ValueError("Saved statistics have no simulation inputs; legacy exports cannot replay")
    if not isinstance(inputs, dict):
        raise ValueError("simulation_inputs must be an object")
    version = inputs.get("schema_version")
    if type(version) is not int or version != 1:
        raise ValueError("Unsupported simulation input schema_version; expected 1")
    if not isinstance(metadata, dict):
        raise ValueError("Saved metadata must be an object")
    seed = _integer(metadata.get("seed"), "seed", 0)
    count = _integer(metadata.get("num_simulations"), "num_simulations", 1)
    engine = validate_race_engine(metadata.get("race_engine"))
    raw_drivers, raw_cars = inputs.get("drivers"), inputs.get("cars")
    if not isinstance(raw_drivers, list):
        raise ValueError("Saved drivers must be a list")
    if not all(isinstance(row, dict) for row in raw_drivers):
        raise ValueError("Each saved driver must be an object")
    if not isinstance(raw_cars, dict):
        raise ValueError("Saved cars must be an object")
    if not all(isinstance(row, dict) for row in raw_cars.values()):
        raise ValueError("Each saved car must be an object")
    for name in ("track", "weather", "runtime"):
        if not isinstance(inputs.get(name), dict):
            raise ValueError(f"Saved {name} must be an object")
    drivers = [Driver.model_validate(row) for row in raw_drivers]
    cars = {key: Car.model_validate(row) for key, row in raw_cars.items()}
    return MonteCarloRunner(
        drivers, cars, Track.model_validate(inputs["track"]),
        Weather.model_validate(inputs["weather"]), seed=seed,
        race_engine=engine,
        starting_tires=inputs.get("starting_tires"),
    ), count
