"""Explain verified abandoned classifications without inferring legacy outcomes."""

from html import escape


def abandonment_statistics_text(results):
    getter = getattr(results, "get_abandonment_statistics", None)
    stats = getter() if callable(getter) else {}
    count = stats.get("recorded_abandoned_races", 0)
    invalid = stats.get("invalid_abandonment_context_races", 0)
    if not count and not invalid:
        return ""
    return (f"Recorded abandoned races: {count}; "
            f"{stats.get('recorded_no_result_races', 0)} without a completed countback lap; "
            f"{invalid} invalid countback records; "
            f"{stats.get('recorded_penalized_drivers', 0)} dry-compound penalties. "
            "Classifications use the historical finish, "
            "including any dry-compound penalty. Later running and suspension are excluded "
            "from countback crossing clocks.")


def abandonment_statistics_html(results):
    text = abandonment_statistics_text(results)
    if not text:
        return ""
    stats = results.get_abandonment_statistics()
    rows = ''.join(f'<tr><th scope="row">{escape(str(lap))}</th>'
                   f'<td>{escape(str(count))}</td></tr>'
                   for lap, count in sorted(stats["countback_laps"].items(),
                                            key=lambda row: int(row[0])))
    return ('<section class="abandonment-evidence"><h2>Abandoned races</h2>'
            f'<p>{escape(text)}</p><table><caption>Recorded countback distance</caption>'
            '<thead><tr><th scope="col">Leader laps</th><th scope="col">Races</th>'
            '</tr></thead><tbody>' + rows + '</tbody></table></section>')
