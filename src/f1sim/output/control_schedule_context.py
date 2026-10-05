"""Report control assumptions and complete execution evidence from saved results."""

from html import escape

from f1sim.simulation.control_schedule import (
    has_red_flag_requests,
    red_flag_action_description,
    validate_control_schedule,
    validate_control_schedule_snapshot,
)


def control_schedule_context(snapshot) -> str:
    if not isinstance(snapshot, dict) or not any(
        key in snapshot for key in ("control_schedule", "control_schedule_policy")
    ):
        return ""
    try:
        track = snapshot.get("track")
        distance = track.get("total_laps") if isinstance(track, dict) else None
        schedule = validate_control_schedule(snapshot.get("control_schedule"), total_laps=distance)
        validate_control_schedule_snapshot(snapshot, schedule)
        if schedule is None:
            raise ValueError("missing source")
    except (TypeError, ValueError):
        return "SC/VSC scenario: invalid saved schedule or policy."
    red = has_red_flag_requests(schedule)
    label = "race-control" if red else "SC/VSC"
    requests = "; ".join(
        (f"after leading lap {row['lap']}: red flag, {red_flag_action_description(row['action'])}"
         if row["control"] == "red_flag"
         else f"after leading lap {row['lap']}: "
         f"{'SC' if row['control'] == 'safety_car' else 'VSC'} for {row['duration_laps']} laps")
        for row in schedule
    ) or "no SC/VSC announcements"
    return (
        f"Assumed {label} schedule: {requests}. Random SC/VSC deployments are disabled; "
        "red flags retain priority. Strategies observe announcements only at their crossings. "
        "An applied request records deployment, not completion of its full duration."
    )


def control_schedule_evidence_text(result) -> str:
    getter = getattr(result, "get_control_schedule_statistics", None)
    stats = getter() if callable(getter) else {}
    if stats.get("source") != "controlled":
        return ""
    label = "Race-control" if has_red_flag_requests(stats["requested_schedule"]) else "SC/VSC"
    return (
        f"{label} execution evidence: {stats['valid_history_races']} complete, "
        f"{stats['missing_history_races']} missing, {stats['invalid_history_races']} invalid "
        f"of {stats['recorded_races']} recorded trials; "
        f"{stats['unrecorded_races']} unrecorded trials; "
        f"{stats['unexpected_histories']} unexpected histories. Complete histories record "
        f"{stats['applied']} applied, {stats['suppressed']} suppressed and "
        f"{stats['not_reached']} unreached requests."
    )


def control_schedule_statistics_html(result, scenario="run") -> str:
    context = control_schedule_context(getattr(result, "input_snapshot", None))
    if not context:
        return ""
    getter = getattr(result, "get_control_schedule_statistics", None)
    stats = getter() if callable(getter) else {}
    text = control_schedule_evidence_text(result)
    body = f"<p>{escape(context)}</p>" + (f"<p>{escape(text)}</p>" if text else "")
    rows = []
    label = "Race-control" if has_red_flag_requests(stats.get("requested_schedule")) else "SC/VSC"
    for entry in stats.get("entries", []):
        control = {"safety_car": "SC", "vsc": "VSC", "red_flag": "Red flag"}[entry["control"]]
        request = entry.get("action", entry.get("duration_laps"))
        if entry["control"] == "red_flag":
            request = red_flag_action_description(request)
        rows.append(
            f"<tr><th scope='row'>After lap {entry['lap']}</th><td>{control}</td>"
            f"<td>{escape(str(request))}</td><td>{entry['applied']}</td>"
            f"<td>{entry['suppressed']}</td><td>{entry['not_reached']}</td></tr>",
        )
    if rows:
        body += (
            '<div class="table-wrap" tabindex="0" role="region" '
            f'aria-label="{escape(str(scenario), quote=True)} {label} execution">'
            f'<table><caption>{label} requests in complete trial histories</caption><thead><tr>'
            '<th scope="col">Announcement</th><th scope="col">Control</th>'
            '<th scope="col">Requested laps / action</th><th scope="col">Applied</th>'
            '<th scope="col">Suppressed</th><th scope="col">Not reached</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
        )
    return '<section class="control-schedule-evidence">' + body + '</section>'
