"""Race points from completed distance and explicit result awards."""

from numbers import Integral
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from f1sim.simulation.race import RaceResult


POINTS_SYSTEM = dict(enumerate((25, 18, 15, 12, 10, 8, 6, 4, 2, 1), start=1))
_DISTANCE_BANDS = (
    (25, (6, 4, 3, 2, 1)),
    (50, (13, 10, 8, 6, 5, 4, 3, 2, 1)),
    (75, (19, 14, 12, 10, 8, 6, 4, 3, 2, 1)),
)


def points_for_classification(
    position: int,
    classified: bool,
    winner_laps: int | None,
    scheduled_laps: int,
    has_two_green_laps: bool,
) -> int:
    """Award distance-band points using the original scheduled race length.

    The green-lap flag represents two consecutive completed leader laps without
    a safety car or virtual safety car. An absent eligible winner earns no points.
    Integer comparisons preserve exact percentage boundaries.
    """
    if (
        not classified
        or position < 1
        or winner_laps is None
        or winner_laps < 2
        or scheduled_laps <= 0
        or not has_two_green_laps
    ):
        return 0
    for percentage, awards in _DISTANCE_BANDS:
        if winner_laps * 100 < scheduled_laps * percentage:
            return awards[position - 1] if position <= len(awards) else 0
    return POINTS_SYSTEM.get(position, 0)


def points_for_result(result: "RaceResult") -> int:
    """Use an explicit award, retaining classification-based legacy scoring."""
    award = getattr(result, "points_awarded", None)
    if isinstance(award, Integral) and not isinstance(award, bool) and award >= 0:
        return int(award)

    from f1sim.simulation.race import result_is_classified

    return POINTS_SYSTEM.get(result.position, 0) if result_is_classified(result) else 0
