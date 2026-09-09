"""Traffic observations at the instant a strategy decision is made."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real


def normalize_current_traffic_gaps(gaps):
    """Optional stay/rejoin gaps for candidate-specific first-lap physics."""
    if gaps is None:
        return None
    if not isinstance(gaps, (tuple, list)) or len(gaps) != 2:
        raise ValueError("current_traffic_gaps must contain stay and rejoin gaps")
    if any(value is not None and (
        isinstance(value, bool) or not isinstance(value, Real)
        or not isfinite(value) or value < 0
    ) for value in gaps):
        raise ValueError("Traffic gaps must be finite nonnegative numbers or None")
    return tuple(None if value is None else float(value) for value in gaps)


@dataclass(frozen=True)
class StrategyTrafficSnapshot:
    """Physical gaps and incremental pit-rejoin pace cost, all in seconds.

    A missing gap means no observed car in that direction. The rejoin cost
    excludes the weather multiplier and may be negative when stopping escapes
    traffic. Callers include any expected queue delay in their projection.
    current_traffic_gaps retains the stay/rejoin observations for pricing each
    candidate through lap physics; None retains the legacy scalar-cost contract.
    """

    gap_ahead: float | None
    gap_behind: float | None
    rejoin_traffic_cost: float
    current_traffic_gaps: tuple[float | None, float | None] | None = None
