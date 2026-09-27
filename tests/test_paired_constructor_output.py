"""Constructor paired-points summaries remain clear across output surfaces."""

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import ConsoleOutput
from f1sim.output import comparison as comparison_output
from f1sim.output import console as console_output


def _paired_variants():
    drivers = [
        Driver(id="A", name="Driver A", team_id="T"),
        Driver(id="B", name="Driver B", team_id="T"),
        Driver(id="C", name="Driver C", team_id="U"),
    ]
    cars = {
        "T": Car(team_id="T", team_name="Team T"),
        "U": Car(team_id="U", team_name="Team U"),
    }
    track = Track(id="test", name="Test", country="Test", total_laps=5,
                  base_lap_time=90)
    variants = {
        compound: MonteCarloRunner(
            drivers, cars, track, Weather(change_probability=0), seed=41,
            starting_tires={"A": compound},
        ).run(3, parallel=False)
        for compound in ("soft", "hard")
    }
    reference_awards = {"A": 10, "B": 10, "C": 5}
    variant_awards = (
        {"A": 15, "B": 4, "C": 5},
        {"A": 19, "B": 0, "C": 5},
        {"A": 10, "B": 9, "C": 5},
    )
    for race in variants["hard"].race_results:
        for row in race:
            row.points_awarded = reference_awards[row.driver_id]
    for index, race in enumerate(variants["soft"].race_results):
        for row in race:
            row.points_awarded = variant_awards[index][row.driver_id]
    return variants


def test_constructor_summary_sums_seed_points_and_focus_keeps_teammate(capsys):
    variants = _paired_variants()
    paired = paired_comparison_statistics(variants, "hard")["variants"]["soft"]
    constructor = paired["constructor_statistics"]["T"]
    assert constructor == {
        "team_name": "Team T",
        "driver_ids": ["A", "B"],
        "paired_races": 3,
        "excluded_pairs": 0,
        "reference_mean_points": 20.0,
        "variant_mean_points": 19.0,
        "mean_points_difference": -1.0,
        "points_difference_standard_error": 0.0,
        "more_points_races": 0,
        "equal_points_races": 0,
        "fewer_points_races": 3,
    }
    assert paired["driver_statistics"]["A"]["points_difference_standard_error"] > 0
    assert paired["driver_statistics"]["B"]["points_difference_standard_error"] > 0

    report = comparison_output.render_comparison_report(
        variants, focus_driver="A", reference_scenario="hard",
    )
    constructor_region = report.split('aria-label="Constructor paired points"', 1)[1].split(
        "</div>", 1,
    )[0]
    for text in (
        "Team T", "(T)", "A, B", "3 paired / 0 excluded", "20.000 → 19.000",
        "-1.000", "0.000 points", "0 / 0 / 3",
    ):
        assert text in constructor_region
    assert "Team U" not in constructor_region

    ConsoleOutput.print_paired_comparison(variants, "hard", driver_id="A")
    focused_console = capsys.readouterr().out
    assert "Team T (T): members A, B" in focused_console
    assert "20.000 -> 19.000" in focused_console
    assert "change -1.000" in focused_console
    assert "more/equal/fewer=0/0/3" in focused_console
    assert "Team U" not in focused_console

    ConsoleOutput.print_paired_comparison(variants, "hard")
    global_console = capsys.readouterr().out
    assert "Team U (U): members C" in global_console


def test_unfiltered_constructor_rows_each_identify_the_escaped_alternative():
    variants = _paired_variants()
    paired = paired_comparison_statistics(variants, "hard")
    alternative = paired["variants"].pop("soft")
    paired["variants"]["early <stop>"] = alternative
    report = comparison_output._paired_constructor_table(paired)
    rows = report.split("<tbody>", 1)[1].split("</tbody>", 1)[0].split("</tr>")
    team_rows = [row for row in rows if "Team T" in row or "Team U" in row]
    assert len(team_rows) == 2
    assert all('<th scope="row">early &lt;stop&gt;</th>' in row for row in team_rows)
    assert '<th scope="row"></th>' not in report


def test_constructor_output_escapes_labels_and_distinguishes_missing_from_zero(
    monkeypatch, capsys,
):
    variants = _paired_variants()
    paired = paired_comparison_statistics(variants, "hard")
    summary = paired["variants"]["soft"]
    summary["constructor_statistics"] = {
        '<svg onload="run()">': {
            "team_name": '<img src=x onerror="run()">',
            "driver_ids": ['<script>run()</script>'],
            "paired_races": 0,
            "excluded_pairs": 3,
            "reference_mean_points": 0.0,
            "variant_mean_points": None,
            "mean_points_difference": 0.0,
            "points_difference_standard_error": None,
            "more_points_races": 0,
            "equal_points_races": 0,
            "fewer_points_races": 0,
        },
    }
    monkeypatch.setattr(comparison_output, "paired_comparison_statistics", lambda *_: paired)
    monkeypatch.setattr(console_output, "paired_comparison_statistics", lambda *_: paired)
    report = comparison_output.render_comparison_report(
        variants, reference_scenario="hard",
    )
    constructor_region = report.split('aria-label="Constructor paired points"', 1)[1].split(
        "</div>", 1,
    )[0]
    assert "&lt;svg" in constructor_region and "<svg" not in constructor_region
    assert "&lt;img" in constructor_region and "<img" not in constructor_region
    assert "&lt;script&gt;" in constructor_region and "<script>" not in constructor_region
    assert "0.000 → Not recorded" in constructor_region
    assert "+0.000" in constructor_region
    assert "0 paired / 3 excluded" in constructor_region
    assert "0 / 0 / 0" in constructor_region

    ConsoleOutput.print_paired_comparison(variants, "hard")
    printed = capsys.readouterr().out
    assert "Constructor paired points" in printed
    assert "0 paired / 3 excluded" in printed
    assert "points 0.000 -> Not recorded" in printed

    del summary["constructor_statistics"]
    monkeypatch.setattr(console_output, "paired_comparison_statistics", lambda *_: paired)
    report = comparison_output.render_comparison_report(
        variants, reference_scenario="hard",
    )
    assert "Not recorded in this saved comparison." in report
    ConsoleOutput.print_paired_comparison(variants, "hard")
    legacy_console = capsys.readouterr().out
    assert "Constructor paired points: Not recorded in this comparison." in legacy_console
    assert legacy_console.rstrip().endswith(
        "Constructor paired points: Not recorded in this comparison."
    )
