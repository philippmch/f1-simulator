"""Presentation of timing values and race-distance gaps."""

from math import isfinite
from numbers import Integral


def finite_time(value: float | None) -> float | None:
    """Keep finite timing values, including zero; expose no-time sentinels as None."""
    return value if value is not None and isfinite(value) else None


def csv_time(value: float | None) -> str:
    time = finite_time(value)
    return f"{time:.3f}" if time is not None else ""


def format_lap_deficit(completed: int | None, leader_completed: int | None) -> str | None:
    """Use a distance gap only when both lap counts establish a positive deficit."""
    counts = (completed, leader_completed)
    if any(isinstance(count, bool) or not isinstance(count, Integral) or count < 0
           for count in counts):
        return None
    deficit = leader_completed - completed
    if deficit <= 0:
        return None
    return f"+{deficit} {'lap' if deficit == 1 else 'laps'}"
