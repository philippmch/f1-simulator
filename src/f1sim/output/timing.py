"""Presentation of timing values, race-distance gaps and suspension pauses."""

from collections.abc import Iterable
from math import isfinite
from numbers import Integral
from typing import Any

from f1sim.simulation.race import get_race_suspension_seconds


def finite_time(value: float | None) -> float | None:
    """Keep finite timing values, including zero; expose no-time sentinels as None."""
    return value if value is not None and isfinite(value) else None


def finite_nonnegative_seconds(value: Any) -> float | None:
    """Return a finite non-negative duration, preserving a meaningful zero.

    Suspension duration is optional in saved/duck-typed results.  This helper
    keeps malformed legacy values out of presentation and JSON while treating
    ``0`` as a recorded value rather than as missing data.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return value if isfinite(value) and value >= 0 else None


def race_suspension_seconds(race_rows: Iterable[Any]) -> float | None:
    """Read one shared completed suspension duration from race rows.

    New race results expose ``f1sim.simulation.race.get_race_suspension_seconds``.
    That helper owns strict shared-value validation and legacy row handling;
    this wrapper keeps all output surfaces on the same public contract.
    """
    try:
        rows = list(race_rows) if race_rows is not None else []
    except (TypeError, ValueError):
        return None
    return finite_nonnegative_seconds(get_race_suspension_seconds(rows))


def suspension_statistics(results: Any) -> dict[str, int | float | None]:
    """Return aggregate suspension observations with an explicit denominator.

    ``SimulationResults.get_suspension_statistics`` is preferred when present.
    Older duck-typed aggregate objects have no trustworthy race denominator,
    so they remain explicitly unknown instead of being inferred from rows.
    """
    method = getattr(results, "get_suspension_statistics", None)
    if callable(method):
        try:
            summary = method()
        except (AttributeError, TypeError, ValueError):
            summary = None
        if isinstance(summary, dict):
            recorded = summary.get("recorded_races")
            with_suspension = summary.get("races_with_recorded_suspension")
            mean = finite_nonnegative_seconds(
                summary.get("mean_completed_suspension_seconds")
            )
            if (isinstance(recorded, Integral) and not isinstance(recorded, bool)
                    and recorded >= 0
                    and isinstance(with_suspension, Integral)
                    and not isinstance(with_suspension, bool)
                    and 0 <= with_suspension <= recorded):
                return {
                    "recorded_races": int(recorded),
                    "races_with_recorded_suspension": int(with_suspension),
                    "mean_completed_suspension_seconds": mean,
                }

    return {
        "recorded_races": 0,
        "races_with_recorded_suspension": 0,
        "mean_completed_suspension_seconds": None,
    }


def format_seconds(value: float | None, *, precision: int = 3) -> str:
    """Format a known duration in seconds, leaving unknown values explicit."""
    seconds = finite_nonnegative_seconds(value)
    return f"{seconds:.{precision}f} s" if seconds is not None else "Not recorded"


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
