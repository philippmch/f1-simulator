"""Shared entrant identity checks at simulation boundaries."""

from collections.abc import Iterable


def validate_unique_ids(ids: Iterable[str], source: str) -> None:
    """Reject repeated exact IDs before simulation state or randomness changes."""
    seen: set[str] = set()
    for driver_id in ids:
        if driver_id in seen:
            raise ValueError(f"Duplicate driver ID {driver_id!r} in {source}")
        seen.add(driver_id)
