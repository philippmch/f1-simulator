"""Compare opening tyres against the same saved inputs and seed range offline."""

from collections.abc import Iterable
from numbers import Integral
from pathlib import Path

from f1sim.analysis.montecarlo import MonteCarloRunner, SimulationResults
from f1sim.analysis.replay import _load_saved_runner
from f1sim.models.tire import TireCompound


def compare_saved_starting_tires(
    path: str | Path,
    driver_id: str,
    compounds: Iterable[str | TireCompound] = ("automatic", "soft", "medium", "hard"),
    *,
    scenario: str | None = None,
    num_simulations: int = 100,
    parallel: bool = False,
    max_workers: int | None = None,
) -> dict[str, SimulationResults]:
    """Run ordered variants, retaining all other drivers' saved tyre overrides.

    Each variant starts at the saved base seed; subsequent adaptive race choices
    remain enabled. Matching seeds do not guarantee matched random race events.
    """
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
    valid = {"automatic", *(compound.value for compound in TireCompound)}
    if not labels or any(not isinstance(label, str) or label not in valid for label in labels):
        raise ValueError("compounds must contain valid tyre labels or automatic")
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
        if label == "automatic":
            overrides.pop(driver_id, None)
        else:
            overrides[driver_id] = label
        variant = MonteCarloRunner(
            [driver.model_copy(deep=True) for driver in runner.drivers],
            {key: car.model_copy(deep=True) for key, car in runner.cars.items()},
            runner.track.model_copy(deep=True), runner.weather.model_copy(deep=True),
            seed=runner.base_seed, race_engine=runner.race_engine, starting_tires=overrides,
        )
        results[label] = variant.run(
            int(num_simulations), parallel=parallel,
            max_workers=None if max_workers is None else int(max_workers),
        )
    return results
