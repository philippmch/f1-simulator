"""Traffic observations at the instant a strategy decision is made."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyTrafficSnapshot:
    """Physical gaps and incremental pit-rejoin pace cost, all in seconds.

    A missing gap means no observed car in that direction. The rejoin cost
    excludes the weather multiplier and may be negative when stopping escapes
    traffic. Callers include any expected queue delay in their projection.
    """

    gap_ahead: float | None
    gap_behind: float | None
    rejoin_traffic_cost: float
