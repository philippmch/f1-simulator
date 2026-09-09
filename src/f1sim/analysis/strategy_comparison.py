"""Compare strategy and engine variants against saved inputs and seeds offline."""

from collections.abc import Iterable
from numbers import Integral
from pathlib import Path

from f1sim.analysis.montecarlo import MonteCarloRunner, SimulationResults
from f1sim.analysis.replay import _load_saved_runner
from f1sim.models.tire import TireCompound
from f1sim.simulation.execution import parse_starting_tire_spec
from f1sim.simulation.randomness import validate_rng_policy


def compare_saved_race_engines(
    path: str | Path,
    engines: Iterable[str] = ("standard", "chronological"),
    *,
    scenario: str | None = None,
    num_simulations: int = 100,
    parallel: bool = False,
    max_workers: int | None = None,
    rng_policy: str | None = None,
) -> dict[str, SimulationResults]:
    """Run ordered engine variants using identical saved models and seed ranges.

    Uses installed simulation code without fetching live inputs. Matching seeds
    preserve qualifying inputs but do not guarantee matched random race events.
    The saved RNG policy is retained unless explicitly overridden for all variants.
    """
    if rng_policy is not None:
        rng_policy = validate_rng_policy(rng_policy)
    for name, value in (("num_simulations", num_simulations), ("max_workers", max_workers)):
        if name == "max_workers" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be greater than 0 (integer required)")
    message = "engines must be a nonempty iterable of distinct engine labels"
    if isinstance(engines, (str, bytes)):
        raise ValueError(message)
    try:
        labels = list(engines)
    except TypeError as exc:
        raise ValueError(message) from exc
    if not labels or any(
        not isinstance(label, str) or label not in ("standard", "chronological")
        for label in labels
    ):
        raise ValueError("engines must contain standard or chronological")
    if len(set(labels)) != len(labels):
        raise ValueError("engines must be distinct")
    runner, _ = _load_saved_runner(path, scenario)
    results = {}
    for label in labels:
        variant = MonteCarloRunner(
            [driver.model_copy(deep=True) for driver in runner.drivers],
            {key: car.model_copy(deep=True) for key, car in runner.cars.items()},
            runner.track.model_copy(deep=True), runner.weather.model_copy(deep=True),
            seed=runner.base_seed, race_engine=label,
            starting_tires=runner.starting_tires.copy(),
            starting_tire_ages=runner.starting_tire_ages.copy(),
            rng_policy=runner.rng_policy if rng_policy is None else rng_policy,
        )
        results[label] = variant.run(
            int(num_simulations), parallel=parallel,
            max_workers=None if max_workers is None else int(max_workers),
        )
    return results


def compare_saved_starting_tires(
    path: str | Path,
    driver_id: str,
    compounds: Iterable[str | TireCompound] = ("automatic", "soft", "medium", "hard"),
    *,
    scenario: str | None = None,
    num_simulations: int = 100,
    parallel: bool = False,
    max_workers: int | None = None,
    rng_policy: str | None = None,
) -> dict[str, SimulationResults]:
    """Run ordered variants, retaining all other drivers' saved tyre overrides.

    Each variant starts at the saved base seed; subsequent adaptive race choices
    remain enabled. Matching seeds do not guarantee matched random race events.
    The saved RNG policy is retained unless explicitly overridden for all variants.
    """
    if rng_policy is not None:
        rng_policy = validate_rng_policy(rng_policy)
    for name, value in (("num_simulations", num_simulations), ("max_workers", max_workers)):
        if name == "max_workers" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be greater than 0 (integer required)")
    if isinstance(compounds, (str, bytes)):
        raise ValueError("compounds must be a nonempty iterable of distinct compound labels")
    try:
        labels = list(compounds)
    except TypeError as exc:
        message = "compounds must be a nonempty iterable of distinct compound labels"
        raise ValueError(message) from exc
    if not labels or any(not isinstance(label, str) for label in labels):
        raise ValueError("compounds must contain valid tyre labels or automatic")
    for label in labels:
        if label != "automatic":
            parse_starting_tire_spec(label)
    labels = [str(label.value) if isinstance(label, TireCompound) else label for label in labels]
    if len(set(labels)) != len(labels):
        raise ValueError("compounds must be distinct")
    runner, _ = _load_saved_runner(path, scenario)
    target = next((driver for driver in runner.drivers if driver.id == driver_id), None)
    if target is None:
        raise ValueError(f"Unknown driver ID: {driver_id}")
    if target.team_id not in runner.cars:
        raise ValueError(f"No saved car available for driver ID: {driver_id}")
    results = {}
    for label in labels:
        overrides = runner.starting_tires.copy()
        ages = runner.starting_tire_ages.copy()
        if label == "automatic":
            overrides.pop(driver_id, None)
            ages.pop(driver_id, None)
        else:
            compound, age = parse_starting_tire_spec(label)
            overrides[driver_id] = compound
            ages.pop(driver_id, None)
            if "@" in label:
                ages[driver_id] = age
        variant = MonteCarloRunner(
            [driver.model_copy(deep=True) for driver in runner.drivers],
            {key: car.model_copy(deep=True) for key, car in runner.cars.items()},
            runner.track.model_copy(deep=True), runner.weather.model_copy(deep=True),
            seed=runner.base_seed, race_engine=runner.race_engine, starting_tires=overrides,
            starting_tire_ages=ages,
            rng_policy=runner.rng_policy if rng_policy is None else rng_policy,
        )
        results[label] = variant.run(
            int(num_simulations), parallel=parallel,
            max_workers=None if max_workers is None else int(max_workers),
        )
    return results
