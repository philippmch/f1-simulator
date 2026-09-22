"""Offline, descriptive comparison reports for saved simulation scenarios."""

import json
from enum import Enum
from html import escape
from math import isfinite
from numbers import Real

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.output.export import Exporter
from f1sim.output.paired_context import paired_coverage_text, paired_exclusion_detail
from f1sim.output.timing import format_seconds, suspension_statistics

_PIT_DECISION_LABELS = {
    "forced_repair": "Forced repair",
    "critical_weather": "Critical weather",
    "weather_reaction": "Weather reaction",
    "compound_requirement": "Compound requirement",
    "dry_forecast": "Dry forecast",
    "rain_forecast": "Rain forecast",
    "inventory_forecast": "Inventory forecast",
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


def _text(value: object) -> str:
    return escape(str(value), quote=True)


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


def _pit_plan_history_html(result: SimulationResults, scenario: str) -> str:
    """Render recorded requested-plan histories, keeping absent data explicit."""
    snapshot = result.input_snapshot
    plans = snapshot.get("pit_plans") if isinstance(snapshot, dict) else None
    if not isinstance(plans, dict) or not plans:
        return ""
    listed = set(plans)
    rows = []
    for simulation, race in enumerate(result.race_results, start=1):
        for row in race:
            if row.driver_id not in listed:
                continue
            history = getattr(row, "pit_plan_history", None)
            if history is None:
                rows.append(
                    f"<tr><td>{simulation}</td><td>{_text(row.driver_id)}</td>"
                    '<td colspan="6">Not recorded</td></tr>'
                )
                continue
            if not history:
                rows.append(
                    f"<tr><td>{simulation}</td><td>{_text(row.driver_id)}</td>"
                    '<td colspan="6">Explicit no elective stops</td></tr>'
                )
                continue
            for record in history:
                if not isinstance(record, dict):
                    rows.append(
                        f"<tr><td>{simulation}</td><td>{_text(row.driver_id)}</td>"
                        '<td colspan="6">Malformed history record</td></tr>'
                    )
                    continue
                raw_status = record.get("status")
                status = (
                    "—" if raw_status is None else
                    _PIT_PLAN_STATUS_LABELS.get(raw_status, raw_status)
                )
                requested_lap = record.get("lap", "Not recorded")
                requested_text = (
                    f"{requested_lap} (own lap)" if isinstance(requested_lap, int)
                    and not isinstance(requested_lap, bool) else str(requested_lap)
                )
                reason = record.get("reason")
                reason_text = "—" if reason is None else _PIT_PLAN_REASON_LABELS.get(
                    reason, reason,
                )
                actual_compound = record.get("actual_compound")
                actual_set_id = record.get("actual_set_id")
                rows.append(
                    f"<tr><td>{simulation}</td><td>{_text(row.driver_id)}</td>"
                    f"<td>{_text(requested_text)}</td>"
                    f"<td>{_text(record.get('compound', 'Not recorded'))}</td>"
                    f"<td>{_text(status)}</td>"
                    f"<td>{_text(reason_text)}</td>"
                    f"<td>{_text(actual_compound if actual_compound is not None else '—')}</td>"
                    f"<td>{_text(actual_set_id if actual_set_id is not None else '—')}</td></tr>"
                )
    if not rows:
        return (
            f'<p>No recorded custom pit-plan history for {_text(scenario)}. '
            'Missing histories remain unknown.</p>'
        )
    return (
        '<div class="table-wrap pit-plan-history" tabindex="0" role="region" '
        f'aria-label="{_text(scenario)} custom pit-plan history">'
        f'<table><caption>Requested and executed custom pit-plan history for {_text(scenario)} '
        '(requested laps are each driver\'s own lap)</caption><thead><tr>'
        '<th scope="col">Trial</th><th scope="col">Driver</th>'
        '<th scope="col">Requested lap</th><th scope="col">Requested compound</th>'
        '<th scope="col">Status</th><th scope="col">Reason</th>'
        '<th scope="col">Actual compound</th><th scope="col">Actual set</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
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
    return (f'{condition}; rain {percent("rain_intensity")}; '
            f'surface wetness {percent("track_wetness")}; '
            f'weather change {percent("change_probability")}/lap; '
            f'weather draws {randomness}')


def _paired_driver_table(driver_id: str, paired: dict | None) -> str:
    if paired is None:
        return ""
    reference = _text(paired["reference_scenario"])
    rows = []
    distance_rows = []
    cost_rows = []
    for label, comparison in paired["variants"].items():
        prefix = f'<tr><th scope="row">{_text(label)}</th>'
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
    return main_table + (
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
) -> str:
    """Render supplied scenario order without ranking or causal interpretation."""
    context = []
    distance_rows = []
    suspension_rows = []
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
    plan_sections = ''.join(
        f'<details><summary>{_text(name)}</summary>'
        + _pit_plan_history_html(result, name) + '</details>'
        for name, result in scenario_results.items()
        if isinstance(result.input_snapshot, dict)
        and isinstance(result.input_snapshot.get("pit_plans"), dict)
        and result.input_snapshot.get("pit_plans")
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
.paired-distance-note { font-size: .875rem; }
.paired-cost-note { font-size: .875rem; }
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
    ) + """</tbody></table></div><h2>Race distance</h2>
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
        "".join(sections) or "<p>No driver outcomes recorded.</p>"
    ) + (
        '<h2>Race tyre set ledgers</h2>' + inventory_sections if inventory_sections else ''
    ) + (
        '<h2>Custom pit-plan execution</h2>' + plan_sections if plan_sections else ''
    ) + "</main></body></html>"
