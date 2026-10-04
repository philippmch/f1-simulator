"""Assumed SC/VSC announcements, observed only at their leading crossing."""

import re

CONTROL_SCHEDULE_POLICY = "observed_control_schedule_v1"
CONTROL_CODES = {"sc": "safety_car", "vsc": "vsc"}
_ENTRY_KEYS = {"lap", "control", "duration_laps"}
CONTROL_OUTCOME_REASONS = {
    "applied": {"scheduled_announcement"},
    "suppressed": {"red_flag", "no_survivors", "existing_neutralization"},
    "not_reached": {"race_ended_before_request"},
}


def validate_control_schedule(value, *, total_laps=None):
    """Preserve None versus an explicit scenario with no SC/VSC deployments."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 20:
        raise ValueError("control_schedule must be a list with at most 20 entries")
    result, previous_end = [], 0
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
            raise ValueError(
                "control_schedule entries require exactly lap, control and duration_laps",
            )
        lap, control, duration = entry["lap"], entry["control"], entry["duration_laps"]
        if type(lap) is not int or not 1 <= lap <= 1000:
            raise ValueError("control_schedule laps must be integers from 1 through 1000")
        if type(duration) is not int or not 1 <= duration <= 6:
            raise ValueError("control_schedule duration_laps must be integers from 1 through 6")
        if not isinstance(control, str) or control not in CONTROL_CODES.values():
            raise ValueError("control_schedule control must be safety_car or vsc")
        if lap <= previous_end:
            raise ValueError(
                "control_schedule announcements must follow the previous clearance lap",
            )
        if total_laps is not None and lap > total_laps:
            raise ValueError("control_schedule laps must not exceed total_laps")
        result.append({"lap": lap, "control": control, "duration_laps": duration})
        previous_end = lap + duration
    return result


def parse_control_schedule_spec(text):
    """Read `12:sc:4,26:vsc:2`, or explicit `none`, without coercing numbers."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("control schedule expects lap:sc/vsc:duration,... or none")
    text = text.strip()
    if text == "none":
        return []
    entries = []
    for token in text.split(","):
        match = re.fullmatch(r"([0-9]+):(sc|vsc):([0-9]+)", token.strip())
        if match is None:
            raise ValueError("control schedule expects lap:sc/vsc:duration,... or none")
        entries.append({"lap": int(match[1]), "control": CONTROL_CODES[match[2]],
                        "duration_laps": int(match[3])})
    return validate_control_schedule(entries)


def validate_control_schedule_snapshot(snapshot, schedule):
    """Reject old inputs that would silently drop the experiment's control source."""
    if type(snapshot.get("schema_version")) is int and snapshot["schema_version"] == 11:
        if (schedule is None or "control_schedule" not in snapshot
                or snapshot.get("control_schedule_policy") != CONTROL_SCHEDULE_POLICY):
            raise ValueError(
                "Schema 11 requires control_schedule and the supported control_schedule_policy",
            )
    elif any(key in snapshot for key in ("control_schedule", "control_schedule_policy")):
        raise ValueError("Schemas 1-10 cannot contain control_schedule or control_schedule_policy")


def validate_control_schedule_history(history, schedule):
    """Require complete trial evidence matching every requested announcement."""
    if not isinstance(history, list) or len(history) != len(schedule):
        raise ValueError("control schedule history must cover every requested announcement")
    for row, request in zip(history, schedule, strict=True):
        if not isinstance(row, dict) or set(row) != _ENTRY_KEYS | {"status", "reason"}:
            raise ValueError("control schedule history has invalid fields")
        recorded = {key: row[key] for key in _ENTRY_KEYS}
        if validate_control_schedule([recorded]) != [request]:
            raise ValueError("control schedule history differs from its request")
        status, reason = row["status"], row["reason"]
        if (not isinstance(status, str) or status not in CONTROL_OUTCOME_REASONS
                or not isinstance(reason, str) or reason not in CONTROL_OUTCOME_REASONS[status]):
            raise ValueError("control schedule history has an invalid outcome")
    return [row.copy() for row in history]
