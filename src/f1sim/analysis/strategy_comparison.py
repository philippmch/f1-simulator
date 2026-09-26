"""Compare strategy and engine variants against saved inputs and seeds offline."""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from numbers import Integral
from pathlib import Path

from f1sim.analysis.montecarlo import MonteCarloRunner, SimulationResults
from f1sim.analysis.replay import _load_saved_runner
from f1sim.models.tire import TireCompound
from f1sim.simulation.execution import parse_starting_tire_spec
from f1sim.simulation.randomness import validate_rng_policy


def _validate_pit_plans(
    value,
    driver_ids,
    *,
    total_laps,
    tire_inventory,
):
    """Validate one plan mapping using the shared race-engine contract."""
    if value is None:
        return None
    from f1sim.simulation.pit_plans import validate_pit_plans

    return validate_pit_plans(
        value,
        driver_ids=driver_ids,
        total_laps=total_laps,
        tire_inventory=tire_inventory,
    )


def _validate_comparison_bounds(num_simulations, max_workers):
    for name, value in (("num_simulations", num_simulations), ("max_workers", max_workers)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be greater than 0 (integer required)")


def _runner_variant(runner: MonteCarloRunner, **overrides) -> MonteCarloRunner:
    """Deep-copy saved runner inputs before running an offline variant."""
    values = {
        "seed": runner.base_seed,
        "race_engine": runner.race_engine,
        "starting_tires": deepcopy(runner.starting_tires),
        "starting_tire_ages": deepcopy(runner.starting_tire_ages),
        "rng_policy": runner.rng_policy,
        "tire_inventory": deepcopy(runner.tire_inventory),
        "pit_plans": deepcopy(getattr(runner, "pit_plans", None)),
        "tire_warmup": deepcopy(runner.tire_warmup),
    }
    values.update(overrides)
    return MonteCarloRunner(
        [driver.model_copy(deep=True) for driver in runner.drivers],
        {key: car.model_copy(deep=True) for key, car in runner.cars.items()},
        runner.track.model_copy(deep=True), runner.weather.model_copy(deep=True),
        **values,
    )


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
            tire_inventory=deepcopy(runner.tire_inventory),
            pit_plans=deepcopy(getattr(runner, "pit_plans", None)),
            tire_warmup=deepcopy(runner.tire_warmup),
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
    # Validate every opening against the saved pool before spending work on trials.
    variant_runners = {}
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
            tire_inventory=deepcopy(runner.tire_inventory),
            pit_plans=deepcopy(getattr(runner, "pit_plans", None)),
            tire_warmup=deepcopy(runner.tire_warmup),
        )
        variant_runners[label] = variant

    results = {}
    for label, variant in variant_runners.items():
        results[label] = variant.run(
            int(num_simulations), parallel=parallel,
            max_workers=None if max_workers is None else int(max_workers),
        )
    return results


def compare_saved_pit_plans(
    path: str | Path,
    driver_id: str,
    plans: Mapping[str, list[dict] | None],
    *,
    scenario: str | None = None,
    num_simulations: int = 100,
    parallel: bool = False,
    max_workers: int | None = None,
    rng_policy: str | None = None,
) -> dict[str, SimulationResults]:
    """Compare named custom pit-plan variants from one saved input bundle.

    ``None`` removes the target driver's saved plan and restores automatic
    strategy.  An empty list is retained as an explicit no-elective-stop plan.
    Every variant is validated and constructed before the first trial starts,
    so malformed alternatives cannot leave a partially run comparison.
    """
    _validate_comparison_bounds(num_simulations, max_workers)
    if rng_policy is not None:
        rng_policy = validate_rng_policy(rng_policy)
    if not isinstance(plans, Mapping) or not plans:
        raise ValueError("plans must be a nonempty mapping of labels to lists or null")
    if len(plans) > 10:
        raise ValueError("plans must contain at most 10 variants")
    labels = list(plans.keys())
    for label in labels:
        if not isinstance(label, str) or not label.strip() or len(label) > 80:
            raise ValueError("plan labels must be nonempty strings of at most 80 characters")
        value = plans[label]
        if value is not None and not isinstance(value, list):
            raise ValueError(f"plan variant {label!r} must be a list or null")

    runner, _ = _load_saved_runner(path, scenario)
    drivers = list(runner.drivers)
    driver_ids = [driver.id for driver in drivers]
    target = next((driver for driver in drivers if driver.id == driver_id), None)
    if target is None:
        raise ValueError(f"Unknown driver ID: {driver_id}")
    if target.team_id not in runner.cars:
        raise ValueError(f"No saved car available for driver ID: {driver_id}")

    source_plans = deepcopy(getattr(runner, "pit_plans", None) or {})
    variant_runners: dict[str, MonteCarloRunner] = {}
    for label, requested in plans.items():
        candidate = deepcopy(source_plans)
        if requested is None:
            candidate.pop(driver_id, None)
        else:
            candidate[driver_id] = deepcopy(requested)
        canonical = _validate_pit_plans(
            candidate or None,
            driver_ids,
            total_laps=runner.track.total_laps,
            tire_inventory=runner.tire_inventory,
        )
        variant_runners[label] = _runner_variant(
            runner,
            rng_policy=runner.rng_policy if rng_policy is None else rng_policy,
            pit_plans=canonical,
        )

    results: dict[str, SimulationResults] = {}
    for label, variant in variant_runners.items():
        results[label] = variant.run(
            int(num_simulations),
            parallel=parallel,
            max_workers=None if max_workers is None else int(max_workers),
        )
    return results
