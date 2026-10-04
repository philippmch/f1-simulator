"""Offline, descriptive comparison reports for saved simulation scenarios."""

import json
from enum import Enum
from html import escape
from math import isfinite
from numbers import Integral, Real

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.output.export import Exporter
from f1sim.output.paired_context import (
    FINISHED_TIME_NOTE,
    finished_race_time_text,
    paired_coverage_text,
    paired_exclusion_detail,
)
from f1sim.output.qualifying_context import qualifying_weather_context
from f1sim.output.timing import format_seconds, suspension_statistics
from f1sim.output.warmup_context import warmup_context
from f1sim.output.weather_schedule_context import weather_schedule_context

_PIT_DECISION_LABELS = {
    "forced_repair": "Forced repair",
    "critical_weather": "Critical weather",
    "weather_reaction": "Weather reaction",
    "compound_requirement": "Compound requirement",
    "dry_forecast": "Dry forecast",
    "rain_forecast": "Rain forecast",
    "inventory_forecast": "Inventory forecast",
    "tyre_usage_limit": "Tyre usage limit",
    "neutralization_window": "Neutralization window",
    "planned_window": "Planned window",
    "user_plan": "Custom pit plan",
}

_PIT_PLAN_STATUS_LABELS = {
    "executed": "Executed",
    "overridden": "Overridden",
    "skipped": "Skipped",
    "not_reached": "Not reached",
}

_PIT_PLAN_REASON_LABELS = {
    **_PIT_DECISION_LABELS,
    "requested_compound_unavailable": "Requested compound unavailable",
    "critical_requested_compound": "Critical requested compound",
    "retired": "Retired",
    "race_finished": "Race finished",
}

_PIT_PLAN_HISTORY_LAZY_RENDER_ROW_THRESHOLD = 1000
_PIT_PLAN_HISTORY_JSON_CHUNK_ROWS = 256


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _overtake_summary_row(scope: str, summary: object) -> str:
    if not isinstance(summary, dict):
        summary = {}
    status = summary.get("status")
    status_text = {
        "recorded": "Recorded",
        "partial": "Partial coverage",
        "not_recorded": "Not recorded",
    }.get(status, "Not recorded")
    attempts, successes, contacts = (
        summary.get("attempts"), summary.get("successes"), summary.get("contacts"),
    )
    valid_counts = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (attempts, successes, contacts)
    ) and successes + contacts <= attempts
    if valid_counts and status in ("recorded", "partial"):
        count_text = f"{attempts} / {successes} / {contacts}"
    else:
        count_text = "Not recorded"

    def rate_text(value: object) -> str:
        if valid_counts and attempts == 0 and status in ("recorded", "partial"):
            return "Not defined (0 attempts)"
        if (not valid_counts or status not in ("recorded", "partial")
                or isinstance(value, bool) or not isinstance(value, Real)
                or not isfinite(value) or not 0 <= value <= 1):
            return "Not recorded"
        return f"{value * 100:.1f}%"

    recorded = summary.get("recorded_driver_races")
    missing = summary.get("missing_driver_races")
    if (isinstance(recorded, int) and not isinstance(recorded, bool) and recorded >= 0
            and isinstance(missing, int) and not isinstance(missing, bool) and missing >= 0):
        coverage = f"{recorded} recorded / {missing} missing available driver-race rows"
    else:
        coverage = "Not recorded"
    return (
        f"<tr><th scope=\"row\">{_text(scope)}</th>"
        f"<td>{status_text}</td><td>{_text(coverage)}</td>"
        f"<td>{_text(count_text)}</td>"
        f"<td>{_text(rate_text(summary.get('success_rate')))}</td>"
        f"<td>{_text(rate_text(summary.get('contact_rate')))}</td></tr>"
    )


def _overtaking_statistics_html(
    result: SimulationResults,
    *,
    scenario: str = "run",
    focus_driver: str | None = None,
    include_note: bool = True,
) -> str:
    """Render pooled overtake-call counts while distinguishing legacy/missing rows."""
    method = getattr(result, "get_overtake_statistics", None)
    statistics = method() if callable(method) else None
    overall = statistics.get("overall") if isinstance(statistics, dict) else None
    drivers = statistics.get("drivers") if isinstance(statistics, dict) else None
    rows = [_overtake_summary_row("Full field", overall)]
    if isinstance(drivers, dict):
        selected = sorted(
            (driver_id, summary) for driver_id, summary in drivers.items()
            if isinstance(driver_id, str)
            and (focus_driver is None or driver_id == focus_driver)
        )
        for driver_id, summary in selected:
            identity = getattr(result, "driver_stats", {}).get(driver_id)
            name = getattr(identity, "driver_name", None)
            label = f"{name} ({driver_id})" if isinstance(name, str) and name else driver_id
            rows.append(_overtake_summary_row(label, summary))
        if focus_driver is not None and not any(row[0] == focus_driver for row in selected):
            rows.append(_overtake_summary_row(focus_driver, None))
    elif focus_driver is not None:
        rows.append(_overtake_summary_row(focus_driver, None))

    note = (
        "Counts include calls to the passing model only; rejected proximity gates and "
        "compliant blue-flag yields are not attempts. Contact counts cover passing calls, "
        "not every collision. Rates are pooled per attempt and describe this model; attempts "
        "are correlated, so no binomial interval is shown."
        if include_note else ""
    )
    note_html = f'<p>{_text(note)}</p>' if note else ""
    return (
        note_html
        + '<div class="table-wrap overtaking-statistics" tabindex="0" role="region" '
        f'aria-label="{_text(scenario)} overtaking attempts and outcomes">'
        f'<table><caption>Overtaking attempts and outcomes for {_text(scenario)}</caption>'
        '<thead><tr><th scope="col">Scope</th><th scope="col">Coverage status</th>'
        '<th scope="col">Available driver-race row coverage</th>'
        '<th scope="col">Attempts / successes / contacts</th>'
        '<th scope="col">Success per attempt</th><th scope="col">Contact per attempt</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
    )


def _paired_coverage_html(paired: dict) -> str:
    lines = []
    for label, comparison in paired["variants"].items():
        if comparison["status"] != "paired":
            continue
        lines.append(
            f'<strong>{_text(label)}</strong>: {_text(paired_coverage_text(comparison))}'
        )
    return '<p>Paired coverage: ' + '<br>'.join(lines) + '</p>' if lines else ''


def _starting_tires(result: SimulationResults) -> str:
    snapshot = result.input_snapshot
    if not isinstance(snapshot, dict):
        return "Not recorded"
    overrides = snapshot.get("starting_tires", {})
    if not isinstance(overrides, dict):
        return "Not recorded"
    ages = snapshot.get("starting_tire_ages", {})
    if not isinstance(ages, dict):
        ages = {}
    return ", ".join(f"{driver}={tire}" + (f"@{ages[driver]}" if ages.get(driver) else "")
                     for driver, tire in overrides.items()) or "Automatic"


def _pit_plans(result: SimulationResults) -> str:
    """Describe configured custom plans while preserving automatic vs empty-plan meaning."""
    snapshot = result.input_snapshot
    if not isinstance(snapshot, dict) or "pit_plans" not in snapshot:
        return "Automatic (no custom plans)"
    plans = snapshot.get("pit_plans")
    if not isinstance(plans, dict):
        return "Not recorded"
    if not plans:
        return "Automatic (no custom plans)"
    rendered = []
    for driver, instructions in plans.items():
        if instructions == []:
            rendered.append(f"{driver}=no elective stops")
            continue
        if not isinstance(instructions, list):
            rendered.append(f"{driver}=not recorded")
            continue
        stops = ", ".join(
            f"{item.get('lap', '—')} own lap: {item.get('compound', '—')}"
            for item in instructions if isinstance(item, dict)
        )
        rendered.append(f"{driver}={stops or 'not recorded'}")
    return "; ".join(rendered) or "Automatic (no custom plans)"


def _pit_plan_history_display_row(
    simulation: int, driver_id: object, history: object, configured_plan: object,
) -> list[list[str]]:
    """Convert one saved history into the exact display cells used by both renderers."""
    driver = str(driver_id)
    if history is None:
        return [[str(simulation), driver, "Not recorded"]]
    if not isinstance(history, list):
        return [[str(simulation), driver, "Malformed history"]]
    if not history:
        if isinstance(configured_plan, list) and not configured_plan:
            message = "Explicit no elective stops"
        elif isinstance(configured_plan, list):
            message = "Incomplete history; requested instructions are missing"
        else:
            message = "Invalid saved plan"
        return [[str(simulation), driver, message]]

    rows = []
    for record in history:
        if not isinstance(record, dict):
            rows.append([str(simulation), driver, "Malformed history record"])
            continue
        raw_status = record.get("status")
        status = (
            "—" if raw_status is None else _PIT_PLAN_STATUS_LABELS.get(
                raw_status, raw_status,
            ) if isinstance(raw_status, str) else str(raw_status)
        )
        requested_lap = record.get("lap", "Not recorded")
        requested_text = (
            f"{requested_lap} (own lap)" if isinstance(requested_lap, int)
            and not isinstance(requested_lap, bool) else str(requested_lap)
        )
        reason = record.get("reason")
        reason_text = (
            "—" if reason is None else _PIT_PLAN_REASON_LABELS.get(
                reason, reason,
            ) if isinstance(reason, str) else str(reason)
        )
        actual_compound = record.get("actual_compound")
        actual_set_id = record.get("actual_set_id")
        rows.append([
            str(simulation), driver, requested_text,
            str(record.get("compound", "Not recorded")), str(status), str(reason_text),
            str(actual_compound if actual_compound is not None else "—"),
            str(actual_set_id if actual_set_id is not None else "—"),
        ])
    return rows


def _pit_plan_history_row_html(row: list[str]) -> str:
    if len(row) == 3:
        return (
            f"<tr><td>{_text(row[0])}</td><td>{_text(row[1])}</td>"
            f'<td colspan="6">{_text(row[2])}</td></tr>'
        )
    return "<tr>" + "".join(f"<td>{_text(value)}</td>" for value in row) + "</tr>"


def _pit_plan_history_json(value: object) -> str:
    """Encode JSON safely inside an inert script element, including hostile saved text."""
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"))
            .replace("&", r"\u0026").replace("<", r"\u003c").replace(">", r"\u003e"))


def _pit_plan_history_html(result: SimulationResults, scenario: str) -> str:
    """Render recorded histories compactly, switching large tables by selected trial."""
    snapshot = result.input_snapshot
    plans = snapshot.get("pit_plans") if isinstance(snapshot, dict) else None
    if not isinstance(plans, dict) or not plans:
        return ""
    listed = set(plans)
    rows_by_trial = []
    compact_trials = []
    compact_trial_counts = []
    total_rows = 0
    lazy = False
    selected_trial = None
    selected_trial_html = []
    races = getattr(result, "race_results", [])
    if not isinstance(races, (list, tuple)):
        return f'<p>No individual pit-plan histories are available for {_text(scenario)}.</p>'
    for simulation, race in enumerate(races, start=1):
        if not isinstance(race, (list, tuple)):
            continue
        trial_rows = []
        encoded_batches = []
        encoded_batch = []
        trial_row_count = 0
        for row in race:
            driver_id = row.get("driver_id") if isinstance(row, dict) else getattr(
                row, "driver_id", None,
            )
            try:
                is_listed = driver_id in listed
            except TypeError:
                is_listed = False
            if not is_listed:
                continue
            history = row.get("pit_plan_history") if isinstance(row, dict) else getattr(
                row, "pit_plan_history", None,
            )
            display_rows = _pit_plan_history_display_row(
                simulation, driver_id, history, plans[driver_id],
            )
            for display_row in display_rows:
                total_rows += 1
                trial_row_count += 1
                encoded_batch.append(display_row)
                if len(encoded_batch) >= _PIT_PLAN_HISTORY_JSON_CHUNK_ROWS:
                    encoded_batches.append(_pit_plan_history_json(encoded_batch)[1:-1])
                    encoded_batch.clear()

                if not lazy:
                    trial_rows.append(display_row)
                    if selected_trial is None:
                        selected_trial = simulation
                    if total_rows > _PIT_PLAN_HISTORY_LAZY_RENDER_ROW_THRESHOLD:
                        lazy = True
                        all_small_trials = rows_by_trial + [(simulation, trial_rows)]
                        selected_rows = next(
                            trial for number, trial in all_small_trials
                            if trial and (number == selected_trial)
                        )
                        selected_trial_html = [
                            _pit_plan_history_row_html(selected_row)
                            for selected_row in selected_rows
                        ]
                        rows_by_trial.clear()
                        trial_rows.clear()
                elif simulation == selected_trial:
                    selected_trial_html.append(_pit_plan_history_row_html(display_row))
        if encoded_batch:
            encoded_batches.append(_pit_plan_history_json(encoded_batch)[1:-1])
        if trial_row_count:
            compact_trials.append((simulation, trial_row_count, ",".join(encoded_batches)))
            compact_trial_counts.append((simulation, trial_row_count))
            if not lazy:
                rows_by_trial.append((simulation, trial_rows))
    if not total_rows:
        return (
            f'<p>No recorded custom pit-plan history for {_text(scenario)}. '
            'Missing histories remain unknown.</p>'
        )
    table = (
        f'<table><caption>Requested and executed custom pit-plan history for {_text(scenario)} '
        '(requested laps are each driver\'s own lap)</caption><thead><tr>'
        '<th scope="col">Trial</th><th scope="col">Driver</th>'
        '<th scope="col">Requested lap</th><th scope="col">Requested compound</th>'
        '<th scope="col">Status</th><th scope="col">Reason</th>'
        '<th scope="col">Actual compound</th><th scope="col">Actual set</th>'
        '</tr></thead><tbody>'
    )
    if not lazy:
        return (
            '<div class="table-wrap pit-plan-history" tabindex="0" role="region" '
            f'aria-label="{_text(scenario)} custom pit-plan history">'
            + table
            + "".join(
                _pit_plan_history_row_html(row)
                for _, trial_rows in rows_by_trial for row in trial_rows
            )
            + '</tbody></table></div>'
        )

    selected_index = next(
        index for index, (number, _) in enumerate(compact_trial_counts)
        if number == selected_trial
    )
    options = "".join(
        f'<option value="{index}"{" selected" if index == selected_index else ""}>'
        f"Trial {number} ({count} rows)</option>"
        for index, (number, count) in enumerate(compact_trial_counts)
    )
    selected_row_count = compact_trial_counts[selected_index][1]
    selected_row_unit = "row" if selected_row_count == 1 else "rows"
    coverage = (
        f"Showing recorded trial {selected_trial} ({len(compact_trial_counts)} trials available "
        f"in this table; {selected_row_count} display {selected_row_unit}). All {total_rows} "
        f"display rows from {len(compact_trial_counts)} trials are retained and available "
        "using this selector."
    )
    data = "[" + ",".join(
        f"[{number},{count},[{rows}]]" for number, count, rows in compact_trials
    ) + "]"
    return (
        '<div class="table-wrap pit-plan-history" tabindex="0" role="region" '
        f'aria-label="{_text(scenario)} custom pit-plan history" data-pit-plan-lazy>'
        '<label>Trial to display: <select data-pit-plan-history-selector>'
        + options
        + '</select></label>'
        f'<p data-pit-plan-history-coverage role="status" aria-live="polite">'
        f'{_text(coverage)}</p>'
        + table
        + "".join(selected_trial_html)
        + '</tbody></table>'
        '<noscript><p>Showing the first available trial. Selecting another trial requires '
        'JavaScript; this offline report retains all history records in its data.</p></noscript>'
        f'<script type="application/json" data-pit-plan-history-data>{data}</script>'
        '<script>(function(){'
        'const section=document.currentScript.closest("[data-pit-plan-lazy]");'
        'const selector=section.querySelector("[data-pit-plan-history-selector]");'
        'const coverage=section.querySelector("[data-pit-plan-history-coverage]");'
        'const body=section.querySelector("tbody");'
        'const trials=JSON.parse(section.querySelector('
        '"[data-pit-plan-history-data]").textContent);'
        'function render(){'
        'const index=Number(selector.value);const [number,count,rows]=trials[index];'
        'const fragment=document.createDocumentFragment();'
        'for(const row of rows){const tr=document.createElement("tr");'
        'const cells=row.length===3?[row[0],row[1]]:row;'
        'for(const value of cells){const td=document.createElement("td");'
        'td.textContent=value;tr.append(td);}'
        'if(row.length===3){const td=document.createElement("td");td.colSpan=6;'
        'td.textContent=row[2];tr.append(td);}fragment.append(tr);}'
        'body.replaceChildren(fragment);'
        'coverage.textContent=`Showing recorded trial ${number} (${trials.length} trials '
        'available in this table; ${count} display ${count===1?"row":"rows"}). All '
        '${trials.reduce((n,t)=>n+t[1],0)} display rows from ${trials.length} trials are retained '
        'and available using this selector.`;'
        '}'
        'selector.addEventListener("change",render);'
        '})();</script></div>'
    )


def _pit_plan_statistics_html(result: SimulationResults, scenario: str) -> str:
    """Render aggregate custom-plan coverage and status counts without raw labels."""
    statistics = result.get_pit_plan_statistics()
    status = statistics["status"]
    if status == "not_recorded":
        return '<p>Custom pit-plan outcomes were not recorded for this run.</p>'
    if status == "invalid":
        return '<p>Saved custom pit-plan inputs are invalid; outcomes cannot be summarized.</p>'

    recorded = statistics["recorded_trials"]
    unit = "trial" if recorded == 1 else "trials"
    if not statistics["drivers"]:
        return (
            f'<p>No driver-specific custom pit plans were configured for {_text(scenario)}. '
            f'{recorded} recorded {unit}.</p>'
        )

    rows = []
    for driver in statistics["drivers"]:
        coverage = (
            f'{driver["valid_histories"]} valid, {driver["missing_histories"]} missing, '
            f'{driver["invalid_histories"]} invalid of {recorded} recorded {unit}'
        )
        if driver["no_elective_stops"]:
            rows.append(
                f'<tr><th scope="row">{_text(driver["driver_id"])}</th>'
                f'<td>{_text(coverage)}</td><td colspan="5">No elective stops configured</td></tr>'
            )
            continue
        for instruction in driver["instructions"]:
            counts = "".join(
                f'<td>{instruction[key]}</td>'
                for key in ("executed", "overridden", "skipped", "not_reached")
            )
            rows.append(
                f'<tr><th scope="row">{_text(driver["driver_id"])}</th>'
                f'<td>{_text(coverage)}</td>'
                f'<td>{instruction["lap"]} (own lap): '
                f'{_text(instruction["compound"])}</td>{counts}</tr>'
            )
    return (
        f'<p>Instruction counts use complete valid histories only. Coverage is '
        f'per driver out of {recorded} recorded {unit}.</p>'
        '<div class="table-wrap pit-plan-statistics" tabindex="0" role="region" '
        f'aria-label="{_text(scenario)} aggregate custom pit-plan outcomes">'
        '<table><caption>Custom pit-plan outcomes for ' + _text(scenario) + '</caption>'
        '<thead><tr><th scope="col">Driver</th><th scope="col">History coverage</th>'
        '<th scope="col">Requested instruction</th><th scope="col">Executed</th>'
        '<th scope="col">Overridden</th><th scope="col">Skipped</th>'
        '<th scope="col">Not reached</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table></div>'
    )


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

    condition = weather.get("condition", "Not recorded")
    if isinstance(condition, Enum):
        condition = condition.value
    policy = snapshot.get("rng_policy", "shared_v1")
    if isinstance(policy, str) and policy == "isolated_weather_v1":
        randomness = "independent of race decisions"
    elif isinstance(policy, str) and policy == "isolated_weather_mechanical_v1":
        randomness = (
            "independent of race decisions; stable per-driver mechanical draws; changed heat, "
            "risk or exposure can change failures; other events share the race stream"
        )
    elif isinstance(policy, str) and policy == "shared_v1":
        randomness = "shared with race events (legacy)"
    else:
        randomness = "not recorded"
    schedule = weather_schedule_context(snapshot)
    atmosphere = ("random atmosphere changes disabled" if schedule else
                  f'weather change {percent("change_probability")}/lap; '
                  f'weather draws {randomness}')
    return (f'{condition}; rain {percent("rain_intensity")}; '
            f'surface wetness {percent("track_wetness")}; '
            + atmosphere
            + (f"; {context}" if (context := warmup_context(snapshot)) else "")
            + (f"; {context}" if (context := qualifying_weather_context(snapshot)) else "")
            + (f"; {schedule}" if schedule else ""))


def _paired_driver_table(driver_id: str, paired: dict | None) -> str:
    if paired is None:
        return ""
    reference = _text(paired["reference_scenario"])
    rows = []
    distance_rows = []
    cost_rows = []
    time_rows = []
    for label, comparison in paired["variants"].items():
        prefix = f'<tr><th scope="row">{_text(label)}</th>'
        time_stats = comparison.get("driver_statistics", {}).get(driver_id, {}).get(
            "finished_race_time",
        )
        time_text = (f"Unavailable: {comparison['reason']}"
                     if comparison["status"] == "unavailable"
                     else finished_race_time_text(time_stats))
        time_rows.append(prefix + f'<td>{_text(time_text)}</td></tr>')
        if comparison["status"] == "unavailable":
            rows.append(prefix + f'<td colspan="4">Unavailable: '
                        f'{_text(comparison["reason"])}</td></tr>')
            distance_rows.append(prefix + f'<td colspan="4">Unavailable: '
                                  f'{_text(comparison["reason"])}</td></tr>')
            cost_rows.append(prefix + f'<td colspan="6">Unavailable: '
                               f'{_text(comparison["reason"])}</td></tr>')
            continue
        stats = comparison["driver_statistics"].get(driver_id)
        if not stats or not stats["paired_races"]:
            excluded = stats["excluded_pairs"] if stats else comparison["available_seed_pairs"]
            detail = _text(paired_exclusion_detail(comparison, excluded))
            rows.append(prefix + f'<td colspan="4">No usable paired results '
                        f'({excluded} excluded pairs) '
                        f'<span class="interval">({detail})'
                        '</span></td></tr>')
            distance = stats.get("completed_distance") if stats else None
            distance_excluded = (
                distance["excluded_pairs"] if distance else comparison["available_seed_pairs"]
            )
            distance_rows.append(
                prefix + f'<td colspan="4">No recorded distance pairs '
                f'({distance_excluded} excluded from distance subset) '
                '<span class="interval">Missing distance is not zero.</span></td></tr>'
            )
            costs = stats.get("paid_stop_costs") if stats else None
            cost_excluded = (
                costs["excluded_pairs"] if costs else comparison["available_seed_pairs"]
            )
            cost_rows.append(
                prefix + f'<td colspan="6">No complete paid-stop cost pairs '
                f'({cost_excluded} excluded from paid-stop cost subset) '
                '<span class="interval">Missing or invalid paid-stop details are not '
                'zero.</span></td></tr>'
            )
            continue
        error = stats["points_difference_standard_error"]
        error_text = f"SE {error:.3f} points" if error is not None else "SE needs at least 2 pairs"
        dnf_error = stats["dnf_rate_difference_standard_error_percentage_points"]
        dnf_error_text = (
            f"DNF rate SE {dnf_error:.3f} pp" if dnf_error is not None
            else "DNF rate SE needs at least 2 pairs"
        )
        joint_counts = (
            f'<span class="interval joint-count">Both finished: '
            f'{stats["both_finished_races"]}</span>'
            f'<span class="interval joint-count">Both DNFs: '
            f'{stats["both_dnf_races"]}</span>'
            f'<span class="interval joint-count">Reference-only DNF: '
            f'{stats["reference_only_dnf_races"]}</span>'
            f'<span class="interval joint-count">Variant-only DNF: '
            f'{stats["variant_only_dnf_races"]}</span>'
        )
        exclusion_detail = _text(
            paired_exclusion_detail(comparison, stats["excluded_pairs"])
        )
        exclusion_html = (
            f'<span class="interval">({exclusion_detail})</span>'
            if stats["excluded_pairs"] else ""
        )
        rows.append(
            prefix + f'<td>{stats["paired_races"]} '
            f'<span class="interval">({stats["excluded_pairs"]} excluded pairs)</span>'
            f'{exclusion_html}</td>'
            f'<td>{stats["mean_points_difference"]:+.3f} '
            f'<span class="interval">{error_text}</span></td>'
            f'<td>{stats["more_points_races"]} / {stats["equal_points_races"]} / '
            f'{stats["fewer_points_races"]}</td>'
            f'<td>{stats["dnf_rate_difference_percentage_points"]:+.1f} pp '
            f'<span class="interval">{dnf_error_text}</span>'
            f'{joint_counts}</td></tr>'
        )
        distance = stats["completed_distance"]
        if not distance["paired_races"]:
            distance_rows.append(
                prefix + f'<td colspan="4">No recorded distance pairs '
                f'({distance["excluded_pairs"]} excluded from distance subset) '
                '<span class="interval">Missing distance is not zero.</span></td></tr>'
            )
        else:
            distance_error = distance["laps_difference_standard_error"]
            distance_error_text = (
                f'SE {distance_error:.3f} laps'
                if distance_error is not None else "SE needs at least 2 distance pairs"
            )
            distance_rows.append(
                prefix + f'<td>{distance["paired_races"]} recorded distance pairs '
                f'<span class="interval">({distance["excluded_pairs"]} excluded from '
                f'distance subset)</span></td>'
                f'<td>{distance["reference_mean_laps"]:.3f} → '
                f'{distance["variant_mean_laps"]:.3f} laps</td>'
                f'<td>{distance["mean_laps_difference"]:+.3f} laps '
                f'<span class="interval">{distance_error_text}</span></td>'
                f'<td>{distance["more_laps_races"]} / {distance["equal_laps_races"]} / '
                f'{distance["fewer_laps_races"]}</td></tr>'
            )
        costs = stats["paid_stop_costs"]
        if not costs["paired_races"]:
            cost_rows.append(
                prefix + f'<td colspan="6">No complete paid-stop cost pairs '
                f'({costs["excluded_pairs"]} excluded from paid-stop cost subset) '
                '<span class="interval">Missing or invalid paid-stop details are not '
                'zero.</span></td></tr>'
            )
        else:
            def cost_cell(label, unit, digits):
                error = costs[f"{label}_difference_standard_error"]
                error_text = (
                    f'SE {error:.{digits}f} {unit}'
                    if error is not None else "SE needs at least 2 cost pairs"
                )
                return (
                    f'{costs[f"reference_mean_{label}"]:.{digits}f} → '
                    f'{costs[f"variant_mean_{label}"]:.{digits}f} {unit}'
                    f'<span class="interval">change '
                    f'{costs[f"mean_{label}_difference"]:+.{digits}f} {unit}; '
                    f'{error_text}</span>'
                )

            cost_rows.append(
                prefix + f'<td>{costs["paired_races"]} complete cost pairs '
                f'<span class="interval">({costs["excluded_pairs"]} excluded from '
                'paid-stop cost subset)</span></td>'
                f'<td>{cost_cell("paid_stops", "paid stops", 2)}</td>'
                f'<td>{cost_cell("total_loss_seconds", "s", 3)}</td>'
                f'<td>{cost_cell("lane_loss_seconds", "s", 3)}</td>'
                f'<td>{cost_cell("service_time_seconds", "s", 3)}</td>'
                f'<td>{cost_cell("queue_time_seconds", "s", 3)}</td></tr>'
            )
    main_table = (
        '<div class="table-wrap" tabindex="0" role="region" '
        f'aria-label="{_text(driver_id)} paired changes">'
        f'<table><caption>Changes for {_text(driver_id)} compared with {reference}</caption>'
        '<thead><tr><th scope="col">Choice</th><th scope="col">Paired races</th>'
        '<th scope="col">Mean points change</th><th scope="col">More / equal / fewer points</th>'
        '<th scope="col">Retirement rate change</th></tr></thead><tbody>'
        + ("".join(rows) or '<tr><td colspan="5">No alternative choices supplied</td></tr>')
        + '</tbody></table></div>'
    )
    distance_table = (
        '<div class="table-wrap paired-distance" tabindex="0" role="region" '
        f'aria-label="{_text(driver_id)} completed distance changes">'
        f'<table><caption>Completed distance for {_text(driver_id)} compared with '
        f'{reference}</caption>'
        '<thead><tr><th scope="col">Choice</th>'
        '<th scope="col">Recorded distance pairs</th>'
        '<th scope="col">Mean laps (reference → variant)</th>'
        '<th scope="col">Mean lap change</th>'
        '<th scope="col">More / equal / fewer laps</th></tr></thead><tbody>'
        + ("".join(distance_rows) or
           '<tr><td colspan="5">No alternative choices supplied</td></tr>')
        + '</tbody></table></div>'
    )
    cost_table = (
        '<div class="table-wrap paired-costs" tabindex="0" role="region" '
        f'aria-label="{_text(driver_id)} paid-stop cost changes">'
        f'<table><caption>Paid-stop costs for {_text(driver_id)} compared with '
        f'{reference}</caption>'
        '<thead><tr><th scope="col">Choice</th>'
        '<th scope="col">Complete cost pairs</th>'
        '<th scope="col">Mean paid stops (reference → variant)</th>'
        '<th scope="col">Mean total loss</th>'
        '<th scope="col">Mean lane loss</th>'
        '<th scope="col">Mean service time</th>'
        '<th scope="col">Mean queue time</th></tr></thead><tbody>'
        + ("".join(cost_rows) or
           '<tr><td colspan="7">No alternative choices supplied</td></tr>')
        + '</tbody></table></div>'
    )
    time_table = (
        '<div class="table-wrap paired-time" tabindex="0" role="region" '
        f'aria-label="{_text(driver_id)} elapsed time changes">'
        f'<table><caption>Elapsed time for {_text(driver_id)} compared with {reference}</caption>'
        '<thead><tr><th scope="col">Choice</th><th scope="col">Same-distance finishes</th>'
        '</tr></thead><tbody>' + ("".join(time_rows) or
        '<tr><td colspan="2">No alternative choices supplied</td></tr>')
        + '</tbody></table></div>'
        + f'<p class="paired-time-note">{_text(FINISHED_TIME_NOTE)}</p>'
    )
    return main_table + time_table + (
        '<p class="paired-distance-note">Completed distance includes '
        'recorded laps for finishes and retirements. Positive changes mean more laps; '
        'missing distance is not zero. These descriptive differences do not establish '
        'causality or rank an optimal strategy.</p>'
    ) + distance_table + (
        '<p class="paired-cost-note">Paid-stop cost pairs require complete detail '
        'lists in both runs. Known zero-stop lists and retired entrants are included; '
        'missing or invalid details are excluded from this subset and are not treated '
        'as zero. Losses are modeled seconds per recorded race, excluding free tyre '
        'changes and later on-track traffic. These descriptive differences do not '
        'isolate causal strategy savings because exposure and race events can change.</p>'
    ) + cost_table


def _paired_constructor_table(paired: dict | None, focus_driver: str | None = None) -> str:
    """Render team points with complete-team coverage and the full focused team roster."""
    if paired is None:
        return ""

    def metric(value: object, *, signed: bool = False) -> str:
        if isinstance(value, bool) or not isinstance(value, Real):
            return "Not recorded"
        try:
            normalized = float(value)
        except (OverflowError, TypeError, ValueError):
            return "Not recorded"
        if not isfinite(normalized):
            return "Not recorded"
        return f"{normalized:+.3f}" if signed else f"{normalized:.3f}"

    def count(value: object) -> str:
        return (str(value) if isinstance(value, int) and not isinstance(value, bool)
                and value >= 0 else "Not recorded")

    rows = []
    for label, comparison in paired.get("variants", {}).items():
        prefix = f'<tr><th scope="row">{_text(label)}</th>'
        if not isinstance(comparison, dict):
            rows.append(prefix + '<td colspan="7">Not recorded</td></tr>')
            continue
        if comparison.get("status") != "paired":
            rows.append(
                prefix + '<td colspan="7">Unavailable: '
                f'{_text(comparison.get("reason") or "Comparison unavailable.")}</td></tr>'
            )
            continue
        summaries = comparison.get("constructor_statistics")
        if not isinstance(summaries, dict):
            rows.append(prefix + '<td colspan="7">Not recorded in this saved comparison.</td></tr>')
            continue
        selected = []
        for team_id, summary in summaries.items():
            if not isinstance(team_id, str) or not isinstance(summary, dict):
                continue
            driver_ids = summary.get("driver_ids")
            driver_ids = sorted(set(
                driver for driver in driver_ids
                if isinstance(driver, str)
            )) if isinstance(driver_ids, (list, tuple)) else []
            if focus_driver is not None and focus_driver not in driver_ids:
                continue
            selected.append((team_id, summary, driver_ids))
        if not selected:
            detail = (
                "Not recorded for the focused driver's team."
                if focus_driver is not None else "No constructor summaries recorded."
            )
            rows.append(prefix + f'<td colspan="7">{_text(detail)}</td></tr>')
            continue
        for team_id, summary, driver_ids in selected:
            pairs = summary.get("paired_races")
            excluded = summary.get("excluded_pairs")
            coverage = (
                f'{count(pairs)} paired / {count(excluded)} excluded'
                if count(pairs) != "Not recorded" and count(excluded) != "Not recorded"
                else "Not recorded"
            )
            mean_text = (
                f'{metric(summary.get("reference_mean_points"))} → '
                f'{metric(summary.get("variant_mean_points"))}'
            )
            error = summary.get("points_difference_standard_error")
            if isinstance(error, bool) or not isinstance(error, Real):
                normalized_error = None
            else:
                try:
                    normalized_error = float(error)
                except (OverflowError, TypeError, ValueError):
                    normalized_error = None
            if normalized_error is None or not isfinite(normalized_error):
                valid_pairs = (
                    pairs if isinstance(pairs, int) and not isinstance(pairs, bool)
                    and pairs >= 0 else None
                )
                if valid_pairs == 0:
                    error_text = "No usable pairs"
                elif valid_pairs == 1:
                    error_text = "Needs at least 2 pairs"
                else:
                    error_text = "Not recorded"
            else:
                error_text = f"{normalized_error:.3f} points"
            team_name = summary.get("team_name")
            team_name = team_name if isinstance(team_name, str) and team_name else team_id
            members = ", ".join(_text(driver) for driver in driver_ids) or "Not recorded"
            more_equal_fewer = " / ".join(count(summary.get(field)) for field in (
                "more_points_races", "equal_points_races", "fewer_points_races",
            ))
            rows.append(
                prefix + f'<td>{_text(team_name)} <span class="interval">({_text(team_id)})'
                f'</span></td><td>{members}</td><td>{coverage}</td><td>{mean_text}</td>'
                f'<td>{metric(summary.get("mean_points_difference"), signed=True)}</td>'
                f'<td>{error_text}</td><td>{more_equal_fewer}</td></tr>'
            )

    if not rows:
        rows.append('<tr><td colspan="8">No alternative choices supplied</td></tr>')
    note = (
        "Complete pairs require every modeled runnable team member in both alternatives. "
        "Points are summed within each seed before the SE is calculated; positive changes "
        "mean more points. The SE describes sampling variation, not a causal effect."
    )
    return (
        '<p class="paired-constructor-note">' + _text(note) + '</p>'
        '<div class="table-wrap paired-constructor" tabindex="0" role="region" '
        'aria-label="Constructor paired points">'
        f'<table><caption>Constructor paired points compared with '
        f'{_text(paired.get("reference_scenario", "reference"))}</caption>'
        '<thead><tr><th scope="col">Alternative</th><th scope="col">Constructor</th>'
        '<th scope="col">Modeled members</th><th scope="col">Complete pairs / excluded</th>'
        '<th scope="col">Mean points (reference → variant)</th>'
        '<th scope="col">Change</th><th scope="col">SE</th>'
        '<th scope="col">More / equal / fewer</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table></div>'
    )


def _pit_decision_driver_table(
    driver_id: str, label: str, scenario_results: dict[str, SimulationResults], summaries: dict,
) -> str:
    rows = []
    for name in scenario_results:
        decisions = summaries[name][4].get(driver_id)
        prefix = f'<tr><th scope="row">{_text(name)}</th>'
        if not decisions:
            rows.append(prefix + '<td colspan="4">Not recorded (no race rows recorded)</td></tr>')
            continue

        races = decisions["races"]
        complete = decisions["races_with_recorded_details"]
        missing_details = decisions["missing_details_races"]
        recorded_stops = decisions["recorded_stops"]
        recorded_reasons = decisions["stops_with_recorded_reasons"]
        missing_reasons = decisions["missing_reason_stops"]
        race_unit = "race" if races == 1 else "races"
        reason_unit = "reason" if recorded_reasons == 1 else "reasons"
        stop_unit = "stop" if recorded_stops == 1 else "stops"
        missing_reason_unit = "reason" if missing_reasons == 1 else "reasons"
        coverage = (
            f'{recorded_reasons} recorded {reason_unit} / {recorded_stops} paid {stop_unit} '
            'in complete records; '
            f'{missing_reasons} missing {missing_reason_unit}; {complete} / {races} {race_unit} '
            'with complete details'
        )
        if missing_details:
            coverage += f'; {missing_details} with missing or inconsistent details'
        reasons = decisions["reasons"]
        if not reasons and not missing_reasons and not missing_details:
            if complete and not missing_details and not recorded_stops and not missing_reasons:
                message = f'No paid stops recorded ({_text(coverage)})'
            else:
                message = f'Not recorded ({_text(coverage)})'
            rows.append(
                prefix + f'<td colspan="4">{message}</td></tr>'
            )
            continue
        for reason, reason_stats in reasons.items():
            reason_label = _PIT_DECISION_LABELS.get(reason, reason)
            rows.append(
                prefix + f'<td>{_text(reason_label)}</td>'
                f'<td>{reason_stats["stops"]}</td>'
                f'<td>{100 * reason_stats["share"]:.1f}% '
                f'<span class="interval">of {recorded_reasons} recorded {reason_unit}'
                '</span></td>'
                f'<td>{_text(coverage)}</td></tr>'
            )
            prefix = f'<tr><th scope="row">{_text(name)}</th>'
        if missing_reasons or missing_details:
            missing_count = missing_reasons if missing_reasons else "Not recorded"
            rows.append(
                prefix + '<td>Not recorded</td>'
                f'<td>{missing_count}</td><td>Not recorded</td>'
                f'<td>{_text(coverage)}</td></tr>'
            )

    return (
        '<div class="table-wrap" tabindex="0" role="region" '
        f'aria-label="{_text(label)} paid-stop decisions">'
        f'<table><caption>Paid-stop decisions for {_text(driver_id)}</caption>'
        '<thead><tr><th scope="col">Scenario</th><th scope="col">Decision reason</th>'
        '<th scope="col">Paid stops</th><th scope="col">Share</th>'
        '<th scope="col">Coverage</th></tr></thead><tbody>'
        + ("".join(rows) or '<tr><td colspan="5">No scenarios recorded</td></tr>')
        + '</tbody></table></div>'
    )


def render_comparison_report(
    scenario_results: dict[str, SimulationResults], *, focus_driver: str | None = None,
    reference_scenario: str | None = None,
    selection: dict | None = None,
) -> str:
    """Render supplied scenario order without ranking or causal interpretation."""
    context = []
    distance_rows = []
    suspension_rows = []
    overtake_sections = []
    drivers = {}
    summaries = {}
    paired = (
        paired_comparison_statistics(scenario_results, reference_scenario)
        if reference_scenario is not None else None
    )
    for name, result in scenario_results.items():
        context.append("<tr>" + f'<th scope="row">{_text(name)}</th>' + "".join(
            f"<td>{_text(value)}</td>" for value in (
                result.track_name, result.race_engine, result.num_simulations,
                result.seed if result.seed is not None else "Not recorded",
                _weather(result),
                _starting_tires(result),
                _pit_plans(result),
                json.dumps((result.input_snapshot or {}).get("tire_inventory", {})),
            )
        ) + "</tr>")
        for driver_id, stats in result.driver_stats.items():
            drivers.setdefault(driver_id, stats)
        overtake_sections.append(
            f'<section><h3 class="overtake-scenario-heading">{_text(name)}</h3>'
            + _overtaking_statistics_html(
                result, scenario=name, focus_driver=focus_driver,
            )
            + '</section>'
        )
        summaries[name] = (
            result.get_probability_intervals(), result.get_pit_stop_statistics(),
            result.get_strategy_statistics(),
            result.get_pit_loss_statistics(), result.get_pit_decision_statistics(),
        )
        distance = result.get_race_distance_statistics()
        recorded = distance["recorded_races"]
        known = distance["races_with_known_winner_distance"]
        comparable = distance["finishers_with_comparable_distance"]
        mean = distance["mean_winner_laps"]
        distance_cells = [str(recorded)]
        distance_cells.append(
            f'{mean:.2f} laps <span class="interval">({known} known winners)</span>'
            if mean is not None else "Not recorded"
        )
        for count, denominator, unit in (
            (distance["time_limited_races"], recorded, "recorded races"),
            (distance["lapped_finishers"], comparable, "comparable finishers"),
            (distance["races_without_winner"], recorded, "recorded races"),
        ):
            distance_cells.append(
                f'{100 * count / denominator:.1f}% '
                f'<span class="interval">({count} / {denominator} {unit})</span>'
                if denominator else "Not recorded"
            )
        distance_rows.append(
            f'<tr><th scope="row">{_text(name)}</th>'
            + "".join(f"<td>{cell}</td>" for cell in distance_cells) + "</tr>"
        )
        suspension = suspension_statistics(result)
        recorded_suspensions = suspension["recorded_races"]
        positive_suspensions = suspension["races_with_recorded_suspension"]
        suspension_unit = "race" if recorded_suspensions == 1 else "races"
        suspension_rows.append(
            f'<tr><th scope="row">{_text(name)}</th>'
            f'<td>{_text(format_seconds(suspension["mean_completed_suspension_seconds"]))}'
            f' <span class="interval">({recorded_suspensions} recorded '
            f'{suspension_unit}; {positive_suspensions} with suspension)</span></td>'
            f'<td>{positive_suspensions} of {recorded_suspensions} recorded '
            f'{suspension_unit}</td></tr>'
        )

    sections = []
    for driver_id, identity in drivers.items():
        rows = []
        strategy_rows = []
        for name, result in scenario_results.items():
            intervals, stops, strategies, losses, _ = summaries[name]
            strategy = strategies.get(driver_id)
            if strategy:
                for sequence in strategy["strategies"]:
                    strategy_rows.append(
                        f'<tr><th scope="row">{_text(name)}</th>'
                        f'<td>{_text(" → ".join(sequence["compounds"]))}</td>'
                        f'<td>{sequence["races"]} / '
                        f'{strategy["races_with_recorded_strategy"]} '
                        f'({100 * sequence["share"]:.1f}%)</td>'
                        f'<td>{sequence["finished_races"]}</td>'
                        f'<td>{sequence["dnf_races"]}</td></tr>'
                    )
            if not strategy or strategy["missing_strategy_races"]:
                missing = (
                    f'missing sequence in {strategy["missing_strategy_races"]} '
                    f'{"race" if strategy["missing_strategy_races"] == 1 else "races"}'
                    if strategy else "No race rows recorded"
                )
                strategy_rows.append(
                    f'<tr><th scope="row">{_text(name)}</th>'
                    f'<td colspan="4">Not recorded ({missing})</td></tr>'
                )
            stats = result.driver_stats.get(driver_id)
            trials = len(stats.positions) if stats else 0
            cells = [f'<th scope="row">{_text(name)}</th>']
            if not trials:
                cells.append('<td colspan="7">Not recorded (no observed trials)</td>')
            else:
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
                loss = losses.get(driver_id)
                cells.append(
                    f'<td>{loss["mean_total_loss_per_race"]:.3f} s '
                    f'<span class="interval">({loss["races_with_recorded_details"]} '
                    'races with complete details)</span></td>'
                    if loss and loss["races_with_recorded_details"] else "<td>Not recorded</td>"
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
            '<th scope="col">Mean paid stops</th><th scope="col">Mean paid-stop loss</th>'
            '</tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>"
            '<div class="table-wrap" tabindex="0" role="region" '
            f'aria-label="{_text(label)} recorded tyre sequences">'
            f'<table><caption>Recorded tyre sequences for {_text(driver_id)}</caption>'
            '<thead><tr><th scope="col">Scenario</th><th scope="col">Tyre sequence</th>'
            '<th scope="col">Races / recorded sequences</th>'
            '<th scope="col">Finished</th><th scope="col">DNF</th></tr></thead><tbody>'
            + "".join(strategy_rows) + "</tbody></table></div>"
            + _pit_decision_driver_table(driver_id, label, scenario_results, summaries)
            + _paired_driver_table(driver_id, paired) + "</details>"
        )

    inventory_sections = ''.join(
        f'<details><summary>{_text(name)}</summary>'
        + Exporter._tire_set_ledger_html(result) + '</details>'
        for name, result in scenario_results.items()
        if any(getattr(row, 'tire_set_history', None) is not None
               for race in result.race_results for row in race)
    )
    plan_sections = []
    for name, result in scenario_results.items():
        aggregate = _pit_plan_statistics_html(result, name)
        if result.get_pit_plan_statistics()["status"] == "not_recorded":
            continue
        history = _pit_plan_history_html(result, name)
        plan_sections.append(
            f'<section class="pit-plan-scenario"><h3>{_text(name)}</h3>{aggregate}'
            f'<details><summary>Trial-by-trial pit-plan history</summary>'
            f'{history or "<p>No individual pit-plan history was recorded.</p>"}'
            '</details></section>'
        )
    plan_sections = ''.join(plan_sections)
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
.paired-distance-note { font-size: .875rem; }
.paired-cost-note { font-size: .875rem; }
.paired-constructor-note { font-size: .875rem; }
.paired-constructor { margin: 12px 0 20px; }
.overtake-scenario-heading { overflow-wrap: anywhere; }
.joint-count { white-space: normal; }
</style></head><body><main><h1>Simulation comparison</h1>
<p>Scenarios appear in supplied order. Check their context and saved inputs when
comparing outcomes.</p>
<p class="scroll-hint">Scroll tables sideways to see every column.</p>
<h2>Scenario context</h2><div class="table-wrap context" tabindex="0" role="region"
aria-label="Scenario context"><table><caption>Recorded run context</caption><thead><tr>
<th scope="col">Scenario</th><th scope="col">Track</th><th scope="col">Engine</th>
<th scope="col">Requested trials</th><th scope="col">Base seed</th>
<th scope="col">Initial weather</th>
<th scope="col">Starting tyre overrides</th>
<th scope="col">Custom pit plans</th>
<th scope="col">Input race set pools (unlisted drivers unlimited)</th></tr></thead><tbody>""" + (
        "".join(context) or '<tr><td colspan="9">No scenarios recorded</td></tr>'
    ) + """</tbody></table></div><h2>Completed race suspension</h2>
<p>Mean suspension uses races with a valid shared duration, including known
zero-second pauses. Positive suspension counts exclude those zero-second
observations. The duration covers Race-wide collection + restart pause, already
included in finish clocks. It is not an individual driver's stopped or driving
time.</p>
<div class="table-wrap context" tabindex="0" role="region" aria-label="Completed race suspension">
<table><caption>Recorded race-wide suspension durations</caption><thead><tr>
<th scope="col">Scenario</th><th scope="col">Mean completed suspension</th>
<th scope="col">Positive suspensions</th></tr></thead><tbody>""" + (
        "".join(suspension_rows) or '<tr><td colspan="3">No scenarios recorded</td></tr>'
    ) + """</tbody></table></div><h2>Overtaking attempts and outcomes</h2>
<p>Counts cover available driver-race rows. The standard engine counts passing-model
calls after its adjacent-car proximity gate; the chronological engine counts attempts
when the physical predecessor is caught. Rejected gates and compliant blue-flag yields
are not attempts. Contact counts cover passing calls, not every collision. Rates are
descriptive, not real-world calibration; attempts can be correlated.</p>""" + (
        "".join(overtake_sections) or '<p>Not recorded.</p>'
    ) + """<h2>Race distance</h2>
<p>Shortened races and lapped finishes can change points and pit-stop counts.
Winning distance uses finished P1 results with a recorded distance. Lapping
compares only finishers whose distance and winner's distance are both known.</p>
<div class="table-wrap context" tabindex="0" role="region" aria-label="Race distance">
<table><caption>Recorded race distances and finish outcomes</caption><thead><tr>
<th scope="col">Scenario</th><th scope="col">Recorded races</th>
<th scope="col">Mean winning distance</th><th scope="col">Time-limited races</th>
<th scope="col">Lapped finishers</th><th scope="col">Races without a winner</th>
</tr></thead><tbody>""" + (
        "".join(distance_rows) or '<tr><td colspan="6">No scenarios recorded</td></tr>'
    ) + """</tbody></table></div><h2>Driver outcomes</h2>
<p>Rates and mean points use observed trials, including retirements. Paid stops
exclude free tyre changes and show their own recorded-race counts. Mean paid-stop
loss uses only races with complete stop details, including recorded zero-stop
races. It includes lane, service and queue loss, excluding later on-track traffic.</p>
<p>Paid-stop decision labels use only recognized recorded reasons. Their shares use
recognized reasons as the denominator; coverage shows complete detail races and
missing or unknown reason records. These labels describe policy context, not
which strategy is better.</p>
<p>Tyre sequences show what was actually fitted, including free changes and
truncated retirement runs. Shares use races with recorded sequences; missing
records appear separately. Sequence frequencies do not measure which strategy
is best. A requested opening tyre may be replaced before lap one.</p>
<p>Individual 95% Wilson intervals describe sampling uncertainty,
not real-world accuracy or intervals of differences between scenarios.
Equal seeds do not freeze later race events.</p>
<p>With independent weather draws, matching weather inputs and seeds give the
same sequence over shared weather-update intervals. Those intervals can occur
at different elapsed times, and a shorter race records a shorter sequence.
Legacy shared draws can change the weather when race decisions change.</p>
<p>When mechanical isolation is selected, each driver's mechanical draws stay
stable by own lap. Heat, risk inputs and actual exposure can still change
failures; other race processes continue sharing the race stream.</p>""" + (
        f'<p>Paired changes compare each choice with {_text(reference_scenario)} using '
        'overlapping recorded trial seeds, matching saved models and qualifying. '
        'Positive points changes mean more points; positive retirement changes mean more DNFs. '
        'SEs estimate sampling error, not confidence intervals: the points SE applies to the '
        'mean points change and the DNF rate SE applies to the DNF rate change. Zero observed '
        'variation does not prove the choices equivalent. Joint DNF counts describe paired '
        'status outcomes but do not identify retirement causes or causal effects. Missing, '
        'invalid or unmatched results are excluded with their counts. Completed distance '
        'includes recorded laps for finishes and retirements and uses a separate denominator; '
        'positive changes mean more laps, while missing distance is not zero. Distance changes '
        'are descriptive and do not establish causality or rank an optimal strategy. Adaptive '
        'race events can differ.</p>'
        if paired is not None else ""
    ) + (
        _paired_coverage_html(paired) if paired is not None else ""
    ) + (
        _paired_constructor_table(paired, focus_driver) if paired is not None else ""
    ) + (
        "".join(sections) or "<p>No driver outcomes recorded.</p>"
    ) + (
        '<h2>Race tyre set ledgers</h2>' + inventory_sections if inventory_sections else ''
    ) + (
        '<h2>Custom pit-plan execution</h2>' + plan_sections if plan_sections else ''
    ) + _selection_objective_html(selection or {}) + "</main></body></html>"


def _selection_report_filename(value: object) -> str | None:
    """Accept generated local basenames only; report data never supplies URLs."""
    if not isinstance(value, str) or not value or len(value) > 255:
        return None
    if value in {".", ".."} or not value[0].isascii() or not value[0].isalnum():
        return None
    if any(not (character.isascii() and (character.isalnum() or character in "._-"))
           for character in value):
        return None
    return value


def _selection_report_link(filename: object, label: object) -> str:
    safe_name = _selection_report_filename(filename)
    if safe_name is None:
        return _text(label)
    return f'<a href="{_text(safe_name)}">{_text(label)}</a>'


def _selection_report_number(value: object) -> str:
    try:
        if (not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value)):
            return "Not recorded"
        number = float(value)
        fixed = f"{number:.3f}"
        return f"{number:.3e}" if number != 0 and float(fixed) == 0 else fixed
    except (TypeError, ValueError, OverflowError):
        return "Not recorded"


def _selection_report_shortfall(row: dict, selected: object, objective="points") -> str:
    gap = row.get("mean_points_behind_selected")
    if objective != "points":
        return _selection_report_number(gap)
    tied = row.get("tied_for_best")
    formatted = _selection_report_number(gap)
    if formatted == "Not recorded" or type(tied) is not bool or gap < 0:
        return "Not recorded"
    if tied:
        if gap != 0:
            return "Not recorded"
        return "0.000 (selected)" if row.get("label") == selected else "0.000 (exact tie)"
    return "Below numeric reporting precision" if gap == 0 else formatted


def _selection_objective_html(selection: dict) -> str:
    """Report probability scores alongside the existing points evidence."""
    objective = selection.get("objective", "points")
    if objective not in ("win", "podium"):
        return ""

    def percent(value, unit="%"):
        if _selection_report_number(value) == "Not recorded":
            return "Not recorded"
        return f"{_selection_report_number(100 * value)}{unit}"

    training_rows = []
    scenario_tables = selection.get("training_scenario_score_tables", {})
    scenario_tables = scenario_tables if isinstance(scenario_tables, dict) else {}
    tables = [("Aggregate", {"scores": selection.get("training_score_table", [])}),
              *scenario_tables.items()]
    for name, table in tables:
        if not isinstance(table, dict) or not isinstance(table.get("scores"), list):
            continue
        for row in table["scores"]:
            if isinstance(row, dict):
                training_rows.append(
                    f'<tr><th scope="row">{_text(name)}</th><td>{_text(row.get("label"))}</td>'
                    f'<td>{percent(row.get("mean_score"))}</td>'
                    f'<td>{_text(row.get("trials", "Not recorded"))}</td></tr>'
                )
    scenario_metrics = selection.get("validation_scenario_metrics", {})
    scenario_metrics = scenario_metrics if isinstance(scenario_metrics, dict) else {}
    metrics_by_name = [("Aggregate", selection.get("validation_target_metrics", {})),
                       *scenario_metrics.items()]
    identity = selection.get("validation_status") == "no_change"
    validation_rows = []
    for name, metrics in metrics_by_name:
        if not isinstance(metrics, dict):
            continue
        error = metrics.get("score_difference_standard_error")
        error_text = (
            "No independent alternative estimate" if identity else
            "Not estimated (1 paired race); this is not zero uncertainty"
            if error is None and metrics.get("paired_races") == 1 else
            "Not estimated" if error is None else percent(error, " percentage points sample SE")
        )
        validation_rows.append(
            f'<tr><th scope="row">{_text(name)}</th>'
            f'<td>{percent(metrics.get("reference_mean_score"))}</td>'
            f'<td>{percent(metrics.get("selected_mean_score"))}</td>'
            f'<td>{percent(metrics.get("mean_score_difference"), " percentage points")}</td>'
            f'<td>{_text(error_text)}</td>'
            f'<td>{_text(metrics.get("paired_races", "Not recorded"))}</td></tr>'
        )
    return (
        '<section aria-label="Selection objective"><h2>Selection objective</h2>'
        f'<p>{_text(selection.get("objective_description", objective))}. '
        f'Training winner: {_text(selection.get("selected_label", "Not recorded"))}. '
        'The objective was fixed before training; the choice remains frozen in validation. '
        'A constructor succeeds once per race when at least one member is classified in '
        'the required position. These are simulator probabilities.</p>'
        '<h3>Training objective probabilities</h3><div class="table-wrap" tabindex="0">'
        '<table><thead><tr><th>Scenario</th><th>Candidate</th><th>Probability</th>'
        '<th>Trials</th></tr></thead><tbody>' + ''.join(training_rows) + '</tbody></table></div>'
        '<h3>Held-out objective probabilities</h3><p>Changes and sample standard errors use '
        'percentage points. Weighted outcomes are combined within each seed before computing '
        'the SE, retaining covariance across scenarios. An identity has no independent '
        'alternative estimate.</p><div class="table-wrap" tabindex="0"><table><thead><tr>'
        '<th>Scenario</th><th>Reference</th><th>Selected</th><th>Change</th><th>Uncertainty</th>'
        '<th>Paired races</th></tr></thead><tbody>' + ''.join(validation_rows)
        + '</tbody></table></div></section>'
    )


def _selection_report_reason(value: object) -> str:
    return {
        "unique_lowest_training_maximum_regret": (
            "It had the unique lowest maximum training shortfall across scenarios."
        ),
        "unique_highest_weighted_training_mean": (
            "It had the unique highest weighted training mean."
        ),
        "reference_preferred_on_exact_tie": "An exact training tie preferred the reference.",
        "first_plan_order_on_exact_tie": (
            "An exact training tie used the first candidate in plan order."
        ),
    }.get(value, "Selection reason not recorded.") if isinstance(value, str) else (
        "Selection reason not recorded."
    )


def _selection_regret_html(selection: dict) -> str:
    if selection.get("selection_method") != "minimax_regret":
        return ""
    probability = selection.get("objective", "points") in ("win", "podium")
    scale = 100 if probability else 1
    unit = "percentage points" if probability else "points"

    def number(value):
        return _selection_report_number(scale * value) if isinstance(value, Real) \
            and not isinstance(value, bool) else "Not recorded"

    rows = []
    details = []
    table = selection.get("training_regret_table", [])
    for row in table if isinstance(table, list) else []:
        if not isinstance(row, dict):
            continue
        label = row.get("label", "Not recorded")
        worst = row.get("worst_scenarios", [])
        worst = (", ".join(str(name) for name in worst)
                 if isinstance(worst, list) else "Not recorded")
        rows.append(
            f'<tr><th scope="row">{_text(label)}</th>'
            f'<td>{number(row.get("maximum_regret"))}</td><td>{_text(worst)}</td>'
            f'<td>{_text(row.get("trials_per_scenario", "Not recorded"))}</td></tr>',
        )
        scenarios = row.get("scenarios", {})
        for name, evidence in scenarios.items() if isinstance(scenarios, dict) else []:
            if not isinstance(evidence, dict):
                continue
            details.append(
                f'<tr><th scope="row">{_text(label)}</th><td>{_text(name)}</td>'
                f'<td>{number(evidence.get("mean_score"))}</td>'
                f'<td>{number(evidence.get("best_candidate_mean_score"))}</td>'
                f'<td>{number(evidence.get("regret"))}</td></tr>',
            )
    return (
        '<section aria-label="Minimax regret training choice">'
        '<h3>Minimax regret training choice</h3>'
        '<p>Choose the smallest maximum scenario shortfall from that scenario\'s best candidate '
        'mean. Scenario weights do not affect this choice. All values use training means; '
        'the maximum is not a bound on individual races or unseen scenarios.</p>'
        f'<p>Shortfalls and scenario means use {unit}. Exact ties prefer the reference, '
        'then plan order. The candidate and scenario sets affect the choice.</p>'
        '<div class="table-wrap" tabindex="0"><table><thead><tr><th>Candidate</th>'
        '<th>Maximum training shortfall</th><th>Worst scenarios</th><th>Trials per scenario</th>'
        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
        '<div class="table-wrap" tabindex="0"><table><thead><tr><th>Candidate</th><th>Scenario</th>'
        '<th>Candidate mean</th><th>Best candidate mean</th><th>Training shortfall</th>'
        '</tr></thead><tbody>' + ''.join(details) + '</tbody></table></div></section>'
    )


def _selection_report_standard_error(value: object, paired_races: object) -> str:
    if value is not None:
        return f"{_selection_report_number(value)} sample SE"
    if isinstance(paired_races, Integral) and not isinstance(paired_races, bool) \
            and paired_races == 1:
        return "Not estimated (1 paired race); this is not zero uncertainty"
    return "Not estimated"


def _selection_report_points_outcome_cells(
    profile: object, paired_races: object = None,
) -> list[str]:
    """Format the profile defensively, since old manifests lack this additive field."""
    if not isinstance(profile, dict):
        return ["Not recorded"] * 6
    count_fields = (
        "paired_races", "more_points_races", "equal_points_races", "fewer_points_races",
    )
    raw_counts = [profile.get(field) for field in count_fields]
    valid_counts = [
        isinstance(value, Integral) and not isinstance(value, bool) and value >= 0
        for value in raw_counts
    ]
    if all(valid_counts) and sum(raw_counts[1:]) != raw_counts[0]:
        return ["Not recorded"] * 6
    if (
        all(valid_counts)
        and isinstance(paired_races, Integral) and not isinstance(paired_races, bool)
        and paired_races >= 0 and raw_counts[0] != paired_races
    ):
        return ["Not recorded"] * 6

    count_text = [str(int(value)) if valid else "Not recorded"
                  for value, valid in zip(raw_counts, valid_counts)]
    conditional_text = []
    for field, count_index, category in (
        ("mean_points_gain_when_ahead", 1, "more-points"),
        ("mean_points_loss_when_behind", 3, "fewer-points"),
    ):
        value = profile.get(field)
        if not valid_counts[count_index]:
            conditional_text.append("Not recorded")
        elif raw_counts[count_index] == 0:
            conditional_text.append(
                f"None (no {category} seeds)" if value is None else "Not recorded"
            )
        elif isinstance(value, Real) and not isinstance(value, bool):
            formatted = _selection_report_number(value)
            conditional_text.append(
                formatted if formatted != "Not recorded" and value > 0
                else "Not recorded"
            )
        else:
            conditional_text.append("Not recorded")
    return [*count_text, *conditional_text]


def render_rival_strategy_selection_report(manifest: dict) -> str:
    """Render the declared training criterion and frozen validation evidence."""
    selection = manifest.get("selection", {})
    selection = selection if isinstance(selection, dict) else {}
    selected = selection.get("selected_label", "Not recorded")
    reference = selection.get("reference_label", "Not recorded")
    target_mode = selection.get("target_mode", "Not recorded")
    target_id = selection.get("target_id", "Not recorded")
    report_context = manifest.get("report_context", {})
    report_context = report_context if isinstance(report_context, dict) else {}
    track_name = report_context.get("track_name", "Not recorded")
    race_engine = report_context.get("race_engine", "Not recorded")
    frozen_qualifying = selection.get("frozen_qualifying_weather")
    weather_active = isinstance(frozen_qualifying, dict)
    weather_heading = "Weather and rival assumptions" if weather_active else "Rival assumptions"
    minimax = selection.get("selection_method") == "minimax_regret"
    choice_name = "Minimax regret" if minimax else "Weighted"
    title = f"{choice_name} weather and rival strategy selection" if weather_active else (
        f"{choice_name} rival strategy selection"
    )
    weather_context = (
        '<p>One frozen target plan applies across all race weather cases below. '
        'Partial weather inputs inherit source values; the table records their full effective '
        'values and schedules. Source race charts retain their original conditions.</p>'
        '<details><summary>Shared qualifying weather (frozen)</summary><code>'
        + _text(json.dumps(frozen_qualifying, ensure_ascii=False, sort_keys=True))
        + '</code></details>'
    ) if weather_active else ""
    qualifying_context = report_context.get("qualifying_weather_context")
    qualifying_html = (f"<p>{_text(qualifying_context)}</p>"
                       if isinstance(qualifying_context, str) and qualifying_context else "")
    schedule_context = report_context.get("weather_schedule_context")
    schedule_html = (f"<p>{_text(schedule_context)}</p>"
                     if not weather_active and isinstance(schedule_context, str)
                     and schedule_context else "")
    members = selection.get("target_member_ids", [])
    members_text = ", ".join(str(member) for member in members) if members else "Not recorded"
    target_plans = manifest.get("target_plans", {})
    target_plans = target_plans if isinstance(target_plans, dict) else {}
    plan_rows = []
    for label, role in ((reference, "Fixed reference"), (selected, "Selected and frozen")):
        plan = target_plans.get(label, "Not included in manifest")
        plan_text = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        plan_rows.append(
            f'<tr><th scope="row">{_text(role)}</th><td>{_text(label)}</td>'
            f'<td><code>{_text(plan_text)}</code></td></tr>'
        )
    if selected == reference:
        plan_rows = plan_rows[:1]

    training = selection.get("training_score_table", [])
    objective = selection.get("objective", "points")
    training_note = (
        "Maximum scenario shortfall chooses the plan. Weighted means below are context; "
        "selected-minus-candidate point differences can be negative for higher-mean candidates."
        if minimax else
        "The objective probabilities above determine the training winner. The following "
        "tables retain points for context; points do not determine this selection."
        if objective in ("win", "podium") else
        "Weighted mean points are per-seed target points averaged after weighting rival "
        "scenarios within each seed. Shortfalls use exact scores before numeric reporting; "
        "equal displayed means do not establish an exact tie."
    )
    training_rows = "".join(
        f'<tr><th scope="row">{_text(row.get("label", "Not recorded"))}</th>'
        f'<td>{_selection_report_number(row.get("mean_points"))}</td>'
        '<td>' + (_selection_report_number(row.get("mean_points_behind_selected"))
                  if minimax else _selection_report_shortfall(row, selected, objective)) + '</td>'
        f'<td>{_text(row.get("trials", "Not recorded"))}</td></tr>'
        for row in training if isinstance(row, dict)
    ) or '<tr><td colspan="4">No training scores recorded.</td></tr>'
    scenario_training = selection.get("training_scenario_score_tables", {})
    scenario_training = scenario_training if isinstance(scenario_training, dict) else {}
    rival_training_sections = []
    for scenario_name, score_data in scenario_training.items():
        if not isinstance(score_data, dict):
            continue
        scores = score_data.get("scores", [])
        score_rows = "".join(
            f'<tr><th scope="row">{_text(row.get("label", "Not recorded"))}</th>'
            f'<td>{_selection_report_number(row.get("mean_points"))}</td>'
            f'<td>{_text(row.get("trials", "Not recorded"))}</td></tr>'
            for row in scores if isinstance(row, dict)
        ) or '<tr><td colspan="3">No per-rival training scores recorded.</td></tr>'
        rival_training_sections.append(
            f'<h3>{_text(scenario_name)}</h3><div class="table-wrap" tabindex="0" '
            f'role="region" aria-label="Training scores for {_text(scenario_name)}">'
            '<table><thead><tr><th scope="col">Candidate plan</th>'
            '<th scope="col">Mean target points</th><th scope="col">Training trials</th>'
            f'</tr></thead><tbody>{score_rows}</tbody></table></div>'
        )
    rival_training_html = (
        "".join(rival_training_sections) or "<p>No rival training tables recorded.</p>"
    )

    rival_rows = []
    scenario_exports = manifest.get("rival_scenarios", {})
    scenario_exports = scenario_exports if isinstance(scenario_exports, dict) else {}
    for scenario in selection.get("rival_scenarios", []):
        if not isinstance(scenario, dict):
            continue
        name = scenario.get("name", "Not recorded")
        overrides = scenario.get("rival_pit_plans", {})
        overrides = overrides if isinstance(overrides, dict) else {}
        override_text = "; ".join(
            f"{driver_id}: {json.dumps(plan, ensure_ascii=False, sort_keys=True)}"
            for driver_id, plan in overrides.items()
        ) or "No rival overrides; source plans apply"
        files = scenario_exports.get(name, {})
        files = files if isinstance(files, dict) else {}
        detail_links = " · ".join((
            _selection_report_link(files.get("training_comparison_html"), "Training details"),
            _selection_report_link(files.get("validation_comparison_html"), "Held-out details"),
            _selection_report_link(files.get("training_comparison_json"), "Training replay JSON"),
            _selection_report_link(files.get("validation_comparison_json"), "Held-out replay JSON"),
        ))
        weather_cell = ""
        if weather_active:
            weather_text = json.dumps(scenario.get("weather", {}), ensure_ascii=False,
                                      sort_keys=True)
            schedule_text = json.dumps(scenario.get("weather_schedule", []), ensure_ascii=False)
            weather_cell = (
                f'<td>Initial race weather: <code>{_text(weather_text)}</code><br>'
                f'Known rainfall steps: <code>{_text(schedule_text)}</code></td>'
            )
        rival_rows.append(
            f'<tr><th scope="row">{_text(name)}</th>'
            f'<td>{_selection_report_number(scenario.get("weight"))}</td>'
            f'<td>{_selection_report_number(scenario.get("normalized_weight"))}</td>'
            f'<td><code>{_text(override_text)}</code></td>{weather_cell}'
            f'<td>{detail_links}</td></tr>'
        )
    rival_rows_html = (
        "".join(rival_rows)
        or f'<tr><td colspan="{6 if weather_active else 5}">No assumptions recorded.</td></tr>'
    )

    target_metrics = selection.get("validation_target_metrics", {})
    target_metrics = target_metrics if isinstance(target_metrics, dict) else {}
    identity = selection.get("validation_status") == "no_change" or selected == reference
    if identity:
        weighted_note = (
            "Identity comparison: selected plan equals the reference, so the difference is "
            "zero by definition. No independent alternative estimate or standard error exists."
        )
        weighted_delta = "0.000 (identity by definition)"
        weighted_se = "No independent alternative estimate"
    else:
        weighted_note = (
            "Weighted selected-minus-reference changes are combined within each seed before "
            "the sample SE is computed, retaining covariance across rival scenarios."
        )
        weighted_delta = _selection_report_number(target_metrics.get("mean_points_difference"))
        weighted_se = _selection_report_standard_error(
            target_metrics.get("points_difference_standard_error"),
            target_metrics.get("paired_races"),
        )
    weighted_rows = (
        f'<tr><th scope="row">Weighted across rival scenarios</th>'
        f'<td>{_selection_report_number(target_metrics.get("reference_mean_points"))}</td>'
        f'<td>{_selection_report_number(target_metrics.get("selected_mean_points"))}</td>'
        f'<td>{weighted_delta}</td><td>{_text(weighted_se)}</td>'
        f'<td>{_text(target_metrics.get("paired_races", "Not recorded"))}</td></tr>'
    )
    scenario_metrics = selection.get("validation_scenario_metrics", {})
    scenario_metrics = scenario_metrics if isinstance(scenario_metrics, dict) else {}
    heldout_rows = []
    for name, metrics in scenario_metrics.items():
        if not isinstance(metrics, dict):
            continue
        delta = "0.000 (identity by definition)" if identity else _selection_report_number(
            metrics.get("mean_points_difference"),
        )
        se = (
            "No independent alternative estimate" if identity else
            _selection_report_standard_error(
                metrics.get("points_difference_standard_error"), metrics.get("paired_races"),
            )
        )
        heldout_rows.append(
            f'<tr><th scope="row">{_text(name)}</th>'
            f'<td>{_selection_report_number(metrics.get("reference_mean_points"))}</td>'
            f'<td>{_selection_report_number(metrics.get("selected_mean_points"))}</td>'
            f'<td>{delta}</td><td>{_text(se)}</td>'
            f'<td>{_text(metrics.get("paired_races", "Not recorded"))}</td></tr>'
        )
    scenario_rows_html = (
        "".join(heldout_rows)
        or '<tr><td colspan="6">No per-rival held-out metrics recorded.</td></tr>'
    )

    outcome_rows = []
    profile_metrics = [("Weighted across rival scenarios", target_metrics)]
    profile_metrics.extend(
        (name, metrics) for name, metrics in scenario_metrics.items()
        if isinstance(metrics, dict)
    )
    for label, metrics in profile_metrics:
        if identity:
            outcome_rows.append(
                f'<tr><th scope="row">{_text(label)}</th>'
                '<td colspan="6">Not independently estimated: selected plan equals the '
                'reference, so no separate alternative outcome profile exists.</td></tr>'
            )
            continue
        cells = _selection_report_points_outcome_cells(
            metrics.get("points_outcome_profile"), metrics.get("paired_races"),
        )
        outcome_rows.append(
            f'<tr><th scope="row">{_text(label)}</th>'
            + "".join(f"<td>{_text(value)}</td>" for value in cells)
            + "</tr>"
        )
    outcome_rows_html = "".join(outcome_rows) or (
        '<tr><td colspan="7">No held-out points outcome profiles recorded.</td></tr>'
    )

    methodology_limits = selection.get("methodology_limits", [])
    methodology_html = "".join(
        f"<li>{_text(limit)}</li>" for limit in methodology_limits
    ) if isinstance(methodology_limits, list) else ""
    methodology_section = (
        f"<h2>Methodology limits</h2><ul>{methodology_html}</ul>"
        if methodology_html else ""
    )

    ranges = selection.get("seed_ranges", {})
    ranges = ranges if isinstance(ranges, dict) else {}
    seed_rows = []
    for phase in ("training", "validation"):
        cohort = ranges.get(phase, {})
        cohort = cohort if isinstance(cohort, dict) else {}
        seed_rows.append(
            f'<tr><th scope="row">{_text(phase.title())}</th>'
            f'<td>{_text(cohort.get("first_seed", "Not recorded"))}–'
            f'{_text(cohort.get("last_seed", "Not recorded"))} inclusive</td>'
            f'<td>{_text(cohort.get("trials", "Not recorded"))}</td></tr>'
        )
    manifest_link = _selection_report_link(
        manifest.get("manifest_filename"), "Selection manifest (JSON)",
    )
    choice_description = (
        "minimum maximum scenario shortfall on training results" if minimax
        else "weighted training results"
    )
    point_difference_heading = (
        "Selected minus candidate mean points" if minimax else "Mean points behind selected"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rival strategy selection report</title><style>
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #0f1220; color: #e8ebff;
  font-family: "Segoe UI", system-ui, sans-serif; line-height: 1.55; }}
main {{ max-width: 1440px; margin: auto; padding: clamp(16px, 3vw, 40px); }}
h1 {{ margin: 0 0 12px; font-size: clamp(1.6rem, 4vw, 2.4rem); }}
h2 {{ font-size: 1.2rem; margin-top: 30px; }}
h3 {{ font-size: 1rem; margin: 20px 0 8px; }}
p {{ max-width: 78ch; color: #b6c0ff; }}
h1, h2, h3, p, li {{ overflow-wrap: anywhere; }}
a {{ color: #9cbbff; }}
.table-wrap {{ overflow-x: auto; max-width: 100%; border-radius: 6px; margin: 12px 0; }}
.scroll-hint {{ display: none; }}
@media (max-width: 700px) {{ .scroll-hint {{ display: block; }} }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #2a3156;
  vertical-align: top; overflow-wrap: anywhere; min-width: 110px; }}
thead th {{ color: #b6c0ff; font-weight: 600; }}
tbody th {{ font-weight: 600; }}
.context {{ background: #181c30; border: 1px solid #2a3156; border-radius: 8px; }}
code {{ white-space: pre-wrap; overflow-wrap: anywhere; color: #d7dcff; }}
:focus-visible {{ outline: 3px solid #9cbbff; outline-offset: 2px; }}
</style></head><body><main>
<h1>{title}</h1>
<p>The target plan was chosen using {choice_description} and then frozen for a
separate held-out seed cohort. Supplied rival-scenario weights are analysis assumptions,
not probabilities learned from race data. Held-out results do not feed back into selection.</p>
<p>Track: {_text(track_name)}. Race engine: {_text(race_engine)}.</p>
{qualifying_html}
{schedule_html}
<p class="scroll-hint">Scroll tables sideways to see every column.</p>
<h2>Frozen target plans</h2>
<p>Target: {_text(target_mode)} {_text(target_id)}. Member driver IDs: {_text(members_text)}.
Reference: {_text(reference)}. Selected and frozen: {_text(selected)}.</p>
<div class="table-wrap context" tabindex="0" role="region" aria-label="Frozen target plans">
<table><thead><tr><th scope="col">Role</th><th scope="col">Plan label</th>
<th scope="col">Target pit plan</th></tr></thead><tbody>{''.join(plan_rows)}</tbody></table></div>
<h2>Training scores</h2>
{_selection_regret_html(selection)}
{_selection_objective_html(selection)}
<p>All values in this section use the training cohort only. {_text(training_note)}
A positive shortfall below float precision is labeled below numeric reporting precision.
These training shortfalls are not fresh validation estimates.</p>
<p>{_text(_selection_report_reason(selection.get("tiebreak_applied")))}</p>
<div class="table-wrap context" tabindex="0" role="region" aria-label="Weighted training scores">
<table><thead><tr><th scope="col">Candidate plan</th>
<th scope="col">Weighted mean target points</th>
<th scope="col">{point_difference_heading}</th>
<th scope="col">Training trials</th></tr></thead><tbody>{training_rows}</tbody></table></div>
{rival_training_html}
<h2>{weather_heading}</h2>
<p>Supplied weights are normalized to sum to one for the weighted result. A listed null
override restores that rival driver to automatic policy; an empty list means no elective
pit stops; an instruction list supplies the custom plan. Unlisted drivers keep their saved
source pit-plan configuration. These weights describe assumptions for this comparison.</p>
{weather_context}
<div class="table-wrap context" tabindex="0" role="region" aria-label="Rival assumptions">
<table><thead><tr><th scope="col">Rival scenario</th><th scope="col">Supplied weight</th>
<th scope="col">Normalized weight</th><th scope="col">Rival driver overrides</th>
{'<th scope="col">Race weather and schedule</th>' if weather_active else ''}
<th scope="col">Detailed local exports</th></tr></thead>
<tbody>{rival_rows_html}</tbody></table></div>
<h2>Held-out validation</h2>
<p>{_text(weighted_note)} Standard errors are sample standard errors of paired differences,
not confidence intervals. A missing one-trial standard error is not zero uncertainty and
must not be read as zero.</p>
<div class="table-wrap context" tabindex="0" role="region" aria-label="Weighted held-out metrics">
<table><thead><tr><th scope="col">Comparison</th><th scope="col">Reference mean points</th>
<th scope="col">Selected mean points</th><th scope="col">Selected minus reference</th>
<th scope="col">Uncertainty</th><th scope="col">Paired races</th></tr></thead>
<tbody>{weighted_rows}</tbody></table></div>
<h3>Per-rival held-out changes</h3>
<div class="table-wrap context" tabindex="0" role="region" aria-label="Per-rival held-out metrics">
<table><thead><tr><th scope="col">Rival scenario</th><th scope="col">Reference mean points</th>
<th scope="col">Selected mean points</th><th scope="col">Selected minus reference</th>
<th scope="col">Uncertainty</th><th scope="col">Paired races</th></tr></thead>
<tbody>{scenario_rows_html}</tbody></table></div>
<h3>Paired points outcome profile</h3>
<p>These are descriptive seed-paired simulator outcomes, not calibrated win probabilities,
real-world causal effects, or confidence bounds. The weighted row first combines rival
scenario points within each seed, so its counts are seed outcomes rather than separate
scenario or race probabilities. Conditional means apply only to seeds in the named category.
Older or malformed manifests show “Not recorded.”</p>
<div class="table-wrap context" tabindex="0" role="region"
aria-label="Held-out points outcome profile">
<table><thead><tr><th scope="col">Comparison</th><th scope="col">Paired observations (seeds)</th>
<th scope="col">More points (seeds)</th><th scope="col">Equal points (seeds)</th>
<th scope="col">Fewer points (seeds)</th><th scope="col">Mean gain when ahead (points)</th>
<th scope="col">Mean loss when behind (points)</th></tr></thead>
<tbody>{outcome_rows_html}</tbody></table></div>
<p>Weighted differences combine scenario results within each seed before calculating their
sample SE, retaining within-seed cross-scenario covariance. Per-rival differences are paired
within each scenario by seed. Results describe this saved simulator experiment; they do not
establish causal or real-world advantage.</p>
{methodology_section}
<h2>Seed cohorts and procedure</h2>
<p>These are the actual disjoint seed cohorts recorded by the selection run; each range is
inclusive. Only the fixed reference and training winner were evaluated in validation. The
selected plan remained frozen regardless of held-out results.</p>
<div class="table-wrap context" tabindex="0" role="region" aria-label="Seed cohorts">
<table><thead><tr><th scope="col">Phase</th><th scope="col">Seeds</th>
<th scope="col">Trials</th></tr></thead><tbody>{''.join(seed_rows)}</tbody></table></div>
<p>{manifest_link}</p>
</main></body></html>"""
