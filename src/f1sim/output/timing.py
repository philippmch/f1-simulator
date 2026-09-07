"""Presentation of unavailable qualifying timing values."""

from math import isfinite


def finite_time(value: float | None) -> float | None:
    """Keep finite timing values, including zero; expose no-time sentinels as None."""
    return value if value is not None and isfinite(value) else None


def csv_time(value: float | None) -> str:
    time = finite_time(value)
    return f"{time:.3f}" if time is not None else ""
