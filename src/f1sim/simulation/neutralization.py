"""Monotonic lap-resolution catch-up behind a safety car."""

from math import isfinite

from f1sim.models._native import register_forecast_helpers


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


def safety_car_running_times(observations, modifier):
    """Resolve an ordered queue of (identifier, entry clock, free running).

    Entry clocks already include physical pit loss. Fitting sensitivity and
    later incident/position reconciliation remain outside this running step.
    The same arithmetic serves actual shared laps and immutable finish fields.
    """
    observations = tuple(observations)
    if not observations:
        return {}
    queue_pace = observations[0][2] * modifier
    result, ahead_crossing = {}, None
    for identifier, entry, free in observations:
        nominal = max(free, queue_pace)
        gap = (None if ahead_crossing is None else
               max(0.0, entry + nominal - ahead_crossing))
        running = safety_car_running_time(free, nominal, gap)
        if ahead_crossing is not None:
            running = max(running, ahead_crossing - entry)
        result[identifier] = running
        ahead_crossing = entry + running
    return result


register_forecast_helpers(globals(), ("safety_car_running_time",))
