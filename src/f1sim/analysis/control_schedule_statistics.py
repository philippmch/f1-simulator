"""Complete per-trial evidence for assumed SC/VSC announcements."""

from f1sim.simulation.control_schedule import (
    CONTROL_OUTCOME_REASONS,
    validate_control_schedule,
    validate_control_schedule_history,
    validate_control_schedule_snapshot,
)


def control_schedule_statistics(snapshot, histories, recorded_races, expected_races):
    """Exclude a whole incomplete/malformed trial instead of reporting partial success."""
    summary = {
        "source": "not_recorded", "requested_schedule": None,
        "recorded_races": recorded_races,
        "unrecorded_races": max(0, expected_races - recorded_races),
        "valid_history_races": 0, "missing_history_races": 0, "invalid_history_races": 0,
        "unexpected_histories": 0,
        **dict.fromkeys(CONTROL_OUTCOME_REASONS, 0), "entries": [],
    }
    if not isinstance(snapshot, dict):
        return summary
    if (type(snapshot.get("schema_version")) is not int
            or snapshot["schema_version"] not in range(1, 16)):
        summary["source"] = "invalid"
        return summary
    try:
        track = snapshot.get("track")
        total_laps = track.get("total_laps") if isinstance(track, dict) else None
        schedule = validate_control_schedule(
            snapshot.get("control_schedule"), total_laps=total_laps,
        )
        validate_control_schedule_snapshot(snapshot, schedule)
    except (TypeError, ValueError):
        summary["source"] = "invalid"
        return summary
    if schedule is None:
        summary["source"] = "automatic"
        return summary
    summary.update(source="controlled", requested_schedule=schedule)
    summary["entries"] = [entry | dict.fromkeys(CONTROL_OUTCOME_REASONS, 0) for entry in schedule]
    if histories is None:
        histories = []
    if not isinstance(histories, list):
        summary["invalid_history_races"] = recorded_races
        return summary
    summary["unexpected_histories"] = max(0, len(histories) - recorded_races)
    for index in range(recorded_races):
        history = histories[index] if index < len(histories) else None
        if history is None:
            summary["missing_history_races"] += 1
            continue
        try:
            rows = validate_control_schedule_history(history, schedule)
        except (TypeError, ValueError):
            summary["invalid_history_races"] += 1
            continue
        summary["valid_history_races"] += 1
        for entry, row in zip(summary["entries"], rows, strict=True):
            entry[row["status"]] += 1
            summary[row["status"]] += 1
    return summary
