"""Race points from completed distance and explicit result awards."""

from dataclasses import dataclass
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
RACE_POINTS_POLICY = "race_distance_points_2026_v1"
POINTS_REASON_LABELS = {
    "no_winner": "No finishing winner",
    "fewer_than_two_laps": "Fewer than two complete leader laps",
    "no_green_pair": "Two consecutive complete green laps were not recorded",
    "not_classified": "Not classified",
    "outside_points_positions": "Outside this race's points positions",
    "full_distance": "Full points schedule",
    "reduced_distance": "Reduced points schedule",
}


@dataclass(frozen=True)
class RacePointsContext:
    """Recorded race-wide scoring inputs, independent of a car's own distance."""

    scheduled_laps: int
    winner_laps: int | None
    has_two_green_laps: bool
    policy: str = RACE_POINTS_POLICY


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


def _recorded_context(value) -> RacePointsContext | None:
    if isinstance(value, RacePointsContext):
        scheduled, winner, green, policy = (
            value.scheduled_laps, value.winner_laps, value.has_two_green_laps, value.policy,
        )
    elif isinstance(value, dict):
        if "winner_laps" not in value:
            return None
        scheduled, winner, green, policy = (
            value.get("scheduled_laps"), value.get("winner_laps"),
            value.get("has_two_green_laps"), value.get("policy"),
        )
    else:
        return None
    if (not isinstance(policy, str) or policy != RACE_POINTS_POLICY
            or not isinstance(green, bool)
            or not isinstance(scheduled, Integral) or isinstance(scheduled, bool)
            or scheduled <= 0):
        return None
    if winner is not None and (
        not isinstance(winner, Integral) or isinstance(winner, bool)
        or not 1 <= winner <= scheduled
    ):
        return None
    return RacePointsContext(int(scheduled), int(winner) if winner is not None else None, green)


def scoring_context_summary(value) -> dict | None:
    """Describe recorded inputs; absent or malformed evidence stays unknown."""
    context = _recorded_context(value)
    if context is None:
        return None
    winner, scheduled = context.winner_laps, context.scheduled_laps
    reason = ("no_winner" if winner is None else "fewer_than_two_laps" if winner < 2
              else "no_green_pair" if not context.has_two_green_laps else None)
    band = None
    if winner is not None:
        band = next((label for percentage, label in (
            (25, "under_25_percent"), (50, "25_to_50_percent"), (75, "50_to_75_percent"),
        ) if winner * 100 < scheduled * percentage), "75_percent_or_more")
    points = points_for_classification(1, True, winner, scheduled, context.has_two_green_laps)
    if reason is not None:
        description = f"No points: {POINTS_REASON_LABELS[reason].lower()}."
    else:
        kind = "Full" if band == "75_percent_or_more" else "Reduced"
        description = (f"{kind} points schedule ({points} for first place): "
                       f"winner completed {winner}/{scheduled} laps; "
                       "two consecutive complete green laps recorded.")
    return {
        "policy": context.policy, "scheduled_laps": scheduled, "winner_laps": winner,
        "has_two_green_laps": context.has_two_green_laps, "distance_band": band,
        "points_eligible": reason is None, "ineligibility_reason": reason,
        "winner_points": points, "description": description,
    }


def _points_reason(result, context: RacePointsContext) -> str | None:
    position = getattr(result, "position", None)
    classified = getattr(result, "classified", None)
    award = getattr(result, "points_awarded", None)
    if (not isinstance(position, Integral) or isinstance(position, bool) or position < 1
            or not isinstance(classified, bool)
            or not isinstance(award, Integral) or isinstance(award, bool) or award < 0):
        return None
    finished = getattr(getattr(result, "status", None), "value", None) == "finished"
    if ((context.winner_laps is None and (classified or finished))
            or (context.winner_laps is not None and position == 1 and not finished)):
        return None
    expected = points_for_classification(
        position, classified, context.winner_laps, context.scheduled_laps,
        context.has_two_green_laps,
    )
    if award != expected:
        return None
    summary = scoring_context_summary(context)
    if summary["ineligibility_reason"] == "no_winner":
        return "no_winner"
    if not classified:
        return "not_classified"
    if summary["ineligibility_reason"] is not None:
        return summary["ineligibility_reason"]
    if expected == 0:
        return "outside_points_positions"
    return ("full_distance" if summary["distance_band"] == "75_percent_or_more"
            else "reduced_distance")


def points_reason_for_result(result) -> str | None:
    """Explain an award only when complete recorded inputs agree with it."""
    context = _recorded_context(getattr(result, "race_points_context", None))
    return _points_reason(result, context) if context is not None else None


def race_scoring_context(results, recorded=None) -> dict | None:
    """Read one race's consistent evidence, including an explicit empty-grid outcome."""
    try:
        rows = list(results)
    except (TypeError, ValueError):
        return None
    raw = recorded if recorded is not None else (
        getattr(rows[0], "race_points_context", None) if rows else None
    )
    context = _recorded_context(raw)
    if context is None:
        return None
    if not rows:
        return scoring_context_summary(context) if context.winner_laps is None else None
    for row in rows:
        row_context = getattr(row, "race_points_context", None)
        if ((row_context is not None or recorded is None)
                and _recorded_context(row_context) != context):
            return None
        if _points_reason(row, context) is None:
            return None
    if context.winner_laps is not None and not any(
        row.position == 1 and row.classified is True
        and isinstance(laps := getattr(row, "laps_completed", None), Integral)
        and not isinstance(laps, bool) and laps == context.winner_laps for row in rows
    ):
        return None
    return scoring_context_summary(context)
