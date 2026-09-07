"""Monotonic lap-resolution catch-up behind a safety car."""

from math import isfinite


def safety_car_running_time(
    free_running: float,
    nominal_running: float,
    gap_to_ahead: float | None,
    target_gap: float = 1.0,
) -> float:
    """Close excess queue gap through future running, without rewriting clocks.

    The leader (no ahead gap) keeps its nominal SC pace. A follower's nominal
    time is the common queue pace, bounded below by its own free running time.
    Followers may recover excess gap, at most that available pace difference.
    One second is the mean of the existing 0.8–1.2-second queue-gap model.
    Call only for a full safety car: VSC preserves gaps instead of bunching.
    """
    if not (isfinite(free_running) and isfinite(nominal_running)
            and 0 < free_running <= nominal_running):
        raise ValueError("running times must be finite, positive and nominal >= free")
    if not isfinite(target_gap) or target_gap < 0:
        raise ValueError("target gap must be finite and nonnegative")
    if gap_to_ahead is None:
        return nominal_running
    if not isfinite(gap_to_ahead) or gap_to_ahead < 0:
        raise ValueError("ahead gap must be finite and nonnegative")
    recoverable = nominal_running - free_running
    return nominal_running - min(recoverable, max(0.0, gap_to_ahead - target_gap))
