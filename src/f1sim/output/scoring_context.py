"""Describe recorded race points without inferring missing scoring inputs."""

from html import escape

from f1sim.simulation.race_points import POINTS_REASON_LABELS


def scoring_statistics_text(results) -> str:
    getter = getattr(results, "get_race_scoring_statistics", None)
    if not callable(getter):
        return "Race scoring evidence: Not recorded."
    return _statistics_text(getter())


def _statistics_text(stats) -> str:
    known = stats["races_with_scoring_evidence"]
    recorded = stats["recorded_races"]
    if not known:
        return f"Race scoring evidence: Not recorded for {recorded} recorded races."
    return (
        f"Race scoring evidence: {known} of {recorded} recorded races; "
        f"{stats['races_without_scoring_evidence']} unknown. "
        f"Full points: {stats['full_points_races']}; "
        f"reduced points: {stats['reduced_points_races']}; "
        f"no points: {stats['zero_points_races']}."
    )


def scoring_statistics_html(results) -> str:
    getter = getattr(results, "get_race_scoring_statistics", None)
    stats = getter() if callable(getter) else None
    text = _statistics_text(stats) if stats is not None else "Race scoring evidence: Not recorded."
    body = f"<p>{escape(text)}</p>"
    if callable(getter):
        reasons = stats["zero_points_reasons"]
        items = [f"<li>{escape(POINTS_REASON_LABELS[reason])}: {count}</li>"
                 for reason, count in reasons.items() if count]
        if items:
            body += "<ul>" + "".join(items) + "</ul>"
    return '<section class="race-scoring-evidence">' + body + '</section>'
