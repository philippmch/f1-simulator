"""Offline, descriptive comparison reports for saved simulation scenarios."""

from html import escape
from math import isfinite
from numbers import Real

from f1sim.analysis.montecarlo import SimulationResults


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _starting_tires(result: SimulationResults) -> str:
    snapshot = result.input_snapshot
    if not isinstance(snapshot, dict):
        return "Not recorded"
    overrides = snapshot.get("starting_tires", {})
    if not isinstance(overrides, dict):
        return "Not recorded"
    return ", ".join(f"{driver}={tire}" for driver, tire in overrides.items()) or "Automatic"


def _weather(result: SimulationResults) -> str:
    snapshot = result.input_snapshot
    weather = snapshot.get("weather") if isinstance(snapshot, dict) else None
    if not isinstance(weather, dict):
        return "Not recorded"

    def percent(key):
        value = weather.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            return "not recorded"
        return f"{value * 100:.0f}%"

    return f'Rain {percent("rain_intensity")}; surface wetness {percent("track_wetness")}'


def render_comparison_report(
    scenario_results: dict[str, SimulationResults], *, focus_driver: str | None = None,
) -> str:
    """Render supplied scenario order without ranking or causal interpretation."""
    context = []
    drivers = {}
    summaries = {}
    for name, result in scenario_results.items():
        context.append("<tr>" + f'<th scope="row">{_text(name)}</th>' + "".join(
            f"<td>{_text(value)}</td>" for value in (
                result.track_name, result.race_engine, result.num_simulations,
                result.seed if result.seed is not None else "Not recorded",
                _weather(result),
                _starting_tires(result),
            )
        ) + "</tr>")
        for driver_id, stats in result.driver_stats.items():
            drivers.setdefault(driver_id, stats)
        summaries[name] = (result.get_probability_intervals(), result.get_pit_stop_statistics())

    sections = []
    for driver_id, identity in drivers.items():
        rows = []
        for name, result in scenario_results.items():
            stats = result.driver_stats.get(driver_id)
            trials = len(stats.positions) if stats else 0
            cells = [f'<th scope="row">{_text(name)}</th>']
            if not trials:
                cells.append('<td colspan="6">Not recorded (no observed trials)</td>')
            else:
                intervals, stops = summaries[name]
                cells.append(f"<td>{trials}</td>")
                for metric, count in (
                    ("win", stats.wins), ("podium", stats.podiums), ("dnf", stats.dnfs),
                ):
                    bounds = intervals[driver_id][metric]
                    cells.append(
                        f"<td>{100 * count / trials:.1f}% "
                        f'<span class="interval">[{bounds["lower"]:.1f}–'
                        f'{bounds["upper"]:.1f}%]</span></td>'
                    )
                cells.append(f"<td>{stats.total_points / trials:.2f}</td>")
                pit = stops.get(driver_id)
                cells.append(
                    f'<td>{pit["average_stops"]:.2f} '
                    f'<span class="interval">({pit["races"]} recorded races)</span></td>'
                    if pit and pit["races"] else "<td>Not recorded</td>"
                )
            rows.append("<tr>" + "".join(cells) + "</tr>")
        opened = " open" if driver_id == focus_driver else ""
        label = f"{identity.driver_name} ({driver_id}) · {identity.team}"
        sections.append(
            f'<details{opened}><summary>{_text(label)}</summary>'
            '<div class="table-wrap" tabindex="0" role="region" '
            f'aria-label="{_text(label)} scenario outcomes">'
            f'<table><caption>Scenario outcomes for {_text(driver_id)}</caption><thead><tr>'
            '<th scope="col">Scenario</th><th scope="col">Observed trials</th>'
            '<th scope="col">Win % [95% interval]</th>'
            '<th scope="col">Podium % [95% interval]</th>'
            '<th scope="col">DNF % [95% interval]</th>'
            '<th scope="col">Points / observed race</th>'
            '<th scope="col">Mean paid stops</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div></details>"
        )

    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulation comparison</title><style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #0f1220; color: #e8ebff;
  font-family: "Segoe UI", system-ui, sans-serif; line-height: 1.55; }
main { max-width: 1440px; margin: auto; padding: clamp(16px, 3vw, 40px); }
h1 { margin: 0 0 12px; font-size: clamp(1.6rem, 4vw, 2.4rem); }
h2 { font-size: 1.2rem; margin-top: 28px; }
p { max-width: 75ch; color: #b6c0ff; }
.table-wrap { overflow-x: auto; max-width: 100%; border-radius: 6px; }
.context table { min-width: 1000px; }
.scroll-hint { display: none; }
@media (max-width: 700px) { .scroll-hint { display: block; } }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
caption { text-align: left; padding: 12px; color: #b6c0ff; }
th, td { padding: 12px; text-align: left; border-bottom: 1px solid #2a3156;
  vertical-align: top; overflow-wrap: anywhere; min-width: 110px; }
thead th { color: #b6c0ff; font-weight: 600; }
tbody th { font-weight: 600; }
details, .context { background: #181c30; border: 1px solid #2a3156; }
details { margin: 12px 0; border-radius: 8px; }
summary { cursor: pointer; padding: 16px; overflow-wrap: anywhere; }
summary:hover { color: #9cbbff; }
:focus-visible { outline: 3px solid #9cbbff; outline-offset: 2px; }
.interval { display: block; color: #b6c0ff; white-space: nowrap; font-size: .875rem; }
</style></head><body><main><h1>Simulation comparison</h1>
<p>Scenarios appear in supplied order. Check their context and saved inputs when
comparing outcomes.</p>
<p class="scroll-hint">Scroll tables sideways to see every column.</p>
<h2>Scenario context</h2><div class="table-wrap context" tabindex="0" role="region"
aria-label="Scenario context"><table><caption>Recorded run context</caption><thead><tr>
<th scope="col">Scenario</th><th scope="col">Track</th><th scope="col">Engine</th>
<th scope="col">Requested trials</th><th scope="col">Base seed</th>
<th scope="col">Initial weather</th>
<th scope="col">Starting tyre overrides</th></tr></thead><tbody>""" + (
        "".join(context) or '<tr><td colspan="7">No scenarios recorded</td></tr>'
    ) + """</tbody></table></div><h2>Driver outcomes</h2>
<p>Rates and mean points use observed trials, including retirements. Paid stops
exclude free tyre changes and show their own recorded-race counts.</p>
<p>Individual 95% Wilson intervals describe sampling uncertainty,
not real-world accuracy or intervals of differences between scenarios.
Equal seeds do not freeze later race events.</p>""" + (
        "".join(sections) or "<p>No driver outcomes recorded.</p>"
    ) + "</main></body></html>"
