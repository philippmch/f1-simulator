"""Report control assumptions and complete execution evidence from saved results."""

from html import escape

from f1sim.simulation.control_schedule import (
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
    requests = "; ".join(
        f"after leading lap {row['lap']}: {'SC' if row['control'] == 'safety_car' else 'VSC'} "
        f"for {row['duration_laps']} laps" for row in schedule
    ) or "no SC/VSC announcements"
    return (
        f"Assumed SC/VSC schedule: {requests}. Random SC/VSC deployments are disabled; "
        "red flags retain priority. Strategies observe announcements only at their crossings. "
        "An applied request records deployment, not completion of its full duration."
    )


def control_schedule_evidence_text(result) -> str:
    getter = getattr(result, "get_control_schedule_statistics", None)
    stats = getter() if callable(getter) else {}
    if stats.get("source") != "controlled":
        return ""
    return (
        f"SC/VSC execution evidence: {stats['valid_history_races']} complete, "
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
    for entry in stats.get("entries", []):
        label = "SC" if entry["control"] == "safety_car" else "VSC"
        rows.append(
            f"<tr><th scope='row'>After lap {entry['lap']}</th><td>{label}</td>"
            f"<td>{entry['duration_laps']}</td><td>{entry['applied']}</td>"
            f"<td>{entry['suppressed']}</td><td>{entry['not_reached']}</td></tr>",
        )
    if rows:
        body += (
            '<div class="table-wrap" tabindex="0" role="region" '
            f'aria-label="{escape(str(scenario), quote=True)} SC/VSC execution">'
            '<table><caption>SC/VSC requests in complete trial histories</caption><thead><tr>'
            '<th scope="col">Announcement</th><th scope="col">Control</th>'
            '<th scope="col">Requested laps</th><th scope="col">Applied</th>'
            '<th scope="col">Suppressed</th><th scope="col">Not reached</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
        )
    return '<section class="control-schedule-evidence">' + body + '</section>'
