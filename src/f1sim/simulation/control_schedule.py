"""Assumed control announcements, observed only at their leading crossing."""

import re

CONTROL_SCHEDULE_POLICY = "observed_control_schedule_v1"
RED_FLAG_CONTROL_SCHEDULE_POLICY = "observed_control_schedule_v2"
WET_RESUMPTION_CONTROL_SCHEDULE_POLICY = "observed_control_schedule_v3"
RED_FLAG_ACTIONS = {"resume", "resume_wet", "abandon"}
CONTROL_CODES = {"sc": "safety_car", "vsc": "vsc"}
_ENTRY_KEYS = {"lap", "control", "duration_laps"}
_RED_KEYS = {"lap", "control", "action"}
CONTROL_OUTCOME_REASONS = {
    "applied": {"scheduled_announcement"},
    "suppressed": {"red_flag", "no_survivors", "existing_neutralization", "race_finished"},
    "not_reached": {"race_ended_before_request"},
}


def validate_control_schedule(value, *, total_laps=None):
    """Preserve None versus an explicit scenario with no SC/VSC deployments."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 20:
        raise ValueError("control_schedule must be a list with at most 20 entries")
    result, previous_end, previous_lap = [], 0, 0
    for entry in value:
        red = isinstance(entry, dict) and entry.get("control") == "red_flag"
        if not isinstance(entry, dict) or set(entry) != (_RED_KEYS if red else _ENTRY_KEYS):
            raise ValueError(
                "control_schedule entries require exactly lap, control and duration_laps; "
                "red_flag entries require exactly lap, control and action",
            )
        lap, control = entry["lap"], entry["control"]
        if type(lap) is not int or not 1 <= lap <= 1000:
            raise ValueError("control_schedule laps must be integers from 1 through 1000")
        if red:
            action = entry["action"]
            if not isinstance(action, str) or action not in RED_FLAG_ACTIONS:
                raise ValueError(
                    "control_schedule red_flag action must be resume, resume_wet or abandon")
            duration = 0 if action == "abandon" else 1
        else:
            duration = entry["duration_laps"]
            if type(duration) is not int or not 1 <= duration <= 6:
                raise ValueError("control_schedule duration_laps must be integers from 1 through 6")
            if not isinstance(control, str) or control not in CONTROL_CODES.values():
                raise ValueError("control_schedule control must be safety_car or vsc")
        if lap <= previous_lap or not red and lap <= previous_end:
            raise ValueError(
                "control_schedule announcements must follow the previous clearance lap",
            )
        if total_laps is not None and lap > total_laps:
            raise ValueError("control_schedule laps must not exceed total_laps")
        result.append({"lap": lap, "control": control, "action": action} if red else
                      {"lap": lap, "control": control, "duration_laps": duration})
        previous_end = lap + duration
        previous_lap = lap
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
        red = re.fullmatch(r"([0-9]+):red:(resume|resume_wet|abandon)", token.strip())
        if red:
            entries.append({"lap": int(red[1]), "control": "red_flag", "action": red[2]})
            continue
        match = re.fullmatch(r"([0-9]+):(sc|vsc):([0-9]+)", token.strip())
        if match is None:
            raise ValueError("control schedule expects lap:sc/vsc:duration or "
                             "lap:red:resume/resume_wet/abandon,... or none")
        entries.append({"lap": int(match[1]), "control": CONTROL_CODES[match[2]],
                        "duration_laps": int(match[3])})
    return validate_control_schedule(entries)


def control_schedule_policy(schedule):
    if has_wet_resumption_requests(schedule):
        return WET_RESUMPTION_CONTROL_SCHEDULE_POLICY
    return (RED_FLAG_CONTROL_SCHEDULE_POLICY if has_red_flag_requests(schedule)
            else CONTROL_SCHEDULE_POLICY)


def has_red_flag_requests(schedule):
    return any(entry["control"] == "red_flag" for entry in schedule or [])


def has_abandonment_requests(schedule):
    return any(entry["control"] == "red_flag" and entry["action"] == "abandon"
               for entry in schedule or [])


def has_wet_resumption_requests(schedule):
    return any(entry["control"] == "red_flag" and entry["action"] == "resume_wet"
               for entry in schedule or [])


def control_schedule_schema_version(schedule):
    if has_wet_resumption_requests(schedule):
        return 13
    return 12 if has_red_flag_requests(schedule) else 11


def red_flag_action_description(action):
    return "resume (full-wet tyres compulsory)" if action == "resume_wet" else action


def validate_control_schedule_snapshot(snapshot, schedule):
    """Reject old inputs that would silently drop the experiment's control source."""
    version = snapshot.get("schema_version")
    if type(version) is int and version in (11, 12, 13):
        expected = {11: CONTROL_SCHEDULE_POLICY, 12: RED_FLAG_CONTROL_SCHEDULE_POLICY,
                    13: WET_RESUMPTION_CONTROL_SCHEDULE_POLICY}[version]
        if (schedule is None or "control_schedule" not in snapshot
                or snapshot.get("control_schedule_policy") != expected):
            raise ValueError(
                f"Schema {version} requires control_schedule and the supported "
                "control_schedule_policy",
            )
        if version == 11 and has_red_flag_requests(schedule):
            raise ValueError("Schema 11 cannot contain red_flag control requests; use schema 12")
        if version < 13 and has_wet_resumption_requests(schedule):
            raise ValueError("Compulsory wet resumption requests require schema 13")
    elif any(key in snapshot for key in ("control_schedule", "control_schedule_policy")):
        raise ValueError("Schemas 1-10 cannot contain control_schedule or control_schedule_policy")


def validate_control_schedule_history(history, schedule):
    """Require complete trial evidence matching every requested announcement."""
    if not isinstance(history, list) or len(history) != len(schedule):
        raise ValueError("control schedule history must cover every requested announcement")
    for row, request in zip(history, schedule, strict=True):
        keys = _RED_KEYS if request["control"] == "red_flag" else _ENTRY_KEYS
        if not isinstance(row, dict) or set(row) != keys | {"status", "reason"}:
            raise ValueError("control schedule history has invalid fields")
        recorded = {key: row[key] for key in keys}
        if validate_control_schedule([recorded]) != [request]:
            raise ValueError("control schedule history differs from its request")
        status, reason = row["status"], row["reason"]
        if (not isinstance(status, str) or status not in CONTROL_OUTCOME_REASONS
                or not isinstance(reason, str) or reason not in CONTROL_OUTCOME_REASONS[status]):
            raise ValueError("control schedule history has an invalid outcome")
    return [row.copy() for row in history]
