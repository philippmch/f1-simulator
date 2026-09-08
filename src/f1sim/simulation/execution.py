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
