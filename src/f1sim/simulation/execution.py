"""Shared public names for selecting race execution engines."""

from collections.abc import Iterable

from f1sim.models.tire import TireCompound

RACE_ENGINES = ("standard", "chronological")


def validate_starting_tires(
    value: object, driver_ids: Iterable[str] | None = None,
) -> dict[str, str]:
    """Copy explicit opening compounds, optionally checking the entrant roster."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("starting_tires must be an object mapping driver IDs to compounds")
    known = set(driver_ids) if driver_ids is not None else None
    result = {}
    for driver_id, compound in value.items():
        if not isinstance(driver_id, str) or not driver_id.strip():
            raise ValueError("starting_tires driver IDs must be nonempty strings")
        if known is not None and driver_id not in known:
            raise ValueError(f"Unknown starting_tires driver ID: {driver_id}")
        if not isinstance(compound, str):
            raise ValueError("starting_tires compounds must be strings")
        try:
            result[driver_id] = TireCompound(compound).value
        except ValueError as exc:
            choices = ", ".join(item.value for item in TireCompound)
            raise ValueError(f"starting_tires compounds must be one of: {choices}") from exc
    return result


def validate_race_engine(value: str) -> str:
    """Reject unknown engines rather than silently changing execution semantics."""
    if not isinstance(value, str) or value not in RACE_ENGINES:
        raise ValueError(f"race_engine must be one of: {', '.join(RACE_ENGINES)}")
    return value


MAX_STARTING_TIRE_AGE = 1000


def validate_starting_tire_ages(value, starting_tires, driver_ids=None):
    """Copy prior wear; every aged set requires an explicit opening compound."""
    compounds = validate_starting_tires(starting_tires, driver_ids)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("starting_tire_ages must be an object")
    result = {}
    for driver, age in value.items():
        if not isinstance(driver, str) or driver not in compounds:
            raise ValueError("starting_tire_ages requires an explicit starting_tires driver")
        if type(age) is not int or not 0 <= age <= MAX_STARTING_TIRE_AGE:
            raise ValueError(
                f"starting_tire_ages must be integers from 0 to {MAX_STARTING_TIRE_AGE}"
            )
        result[driver] = age
    return result


def parse_starting_tire_spec(value):
    """Parse a canonical compound or compound@prior-wear-laps."""
    if not isinstance(value, str):
        raise ValueError("Starting tyre specification must be a string")
    pieces = value.split("@")
    if len(pieces) > 2 or (len(pieces) == 2 and (
        not pieces[1] or not pieces[1].isascii() or not pieces[1].isdigit()
    )):
        raise ValueError("Expected compound or compound@nonnegative-integer")
    compound = validate_starting_tires({"driver": pieces[0]})["driver"]
    age = int(pieces[1]) if len(pieces) == 2 else 0
    validate_starting_tire_ages({"driver": age}, {"driver": compound})
    return compound, age
