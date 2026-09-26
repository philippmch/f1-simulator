"""Paired analysis reaches explicit comparison exports and command-line reports."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.analysis.strategy_comparison import compare_saved_starting_tires
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import ConsoleOutput, Exporter
from f1sim.output.comparison import render_comparison_report
from f1sim.output.paired_context import finished_race_time_text


def test_legacy_elapsed_time_is_not_recorded():
    assert finished_race_time_text(None) == "Elapsed time: Not recorded."


def test_missing_elapsed_time_keeps_other_outputs(tmp_path, capsys):
    variants = compare_saved_starting_tires(
        saved(tmp_path), "0", ["soft", "hard"], num_simulations=1,
    )
    for race in variants["soft"].race_results:
        for row in race:
            row.total_time = float("nan")
    stats = paired_comparison_statistics(variants, "hard")["variants"]["soft"]
    assert stats["driver_statistics"]["0"]["paired_races"] == 1
    report = render_comparison_report(variants, reference_scenario="hard")
    assert "0 same-distance finish pairs (1 excluded from time subset)" in report
    assert "no comparable elapsed times" in report
    ConsoleOutput.print_paired_comparison(variants, "hard")
    printed = capsys.readouterr().out
    assert "no comparable elapsed times" in printed
    assert "Points change" in printed


def saved(tmp_path, engine="standard"):
    drivers = [Driver(id=str(i), name=str(i), team_id=str(i)) for i in range(2)]
    result = MonteCarloRunner(
        drivers, {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers},
        Track(id="t", name="Saved", country="T", total_laps=5, base_lap_time=90),
        Weather(change_probability=0), seed=81, race_engine=engine,
    ).run(2, parallel=False)
    return Exporter(tmp_path).export_statistics_json(result)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_console_json_html_and_replay_share_paired_summary(tmp_path, capsys, engine):
    path = saved(tmp_path, engine)
    source = path.read_bytes()
    variants = compare_saved_starting_tires(path, "0", ["soft", "hard"], num_simulations=3)
    expected = paired_comparison_statistics(variants, "hard")
    stats = expected["variants"]["soft"]["driver_statistics"]["0"]
    assert stats["paired_races"] == 3
    exporter = Exporter(tmp_path)
    combined = exporter.export_scenario_comparison_json(variants, reference_scenario="hard")
    payload = json.loads(combined.read_text(encoding="utf-8"))
    assert payload["paired_comparisons"] == expected
    report = exporter.export_scenario_comparison_html(
        variants, focus_driver="0", reference_scenario="hard",
    ).read_text(encoding="utf-8")
    assert "Changes for 0 compared with hard" in report
    assert f'{stats["mean_points_difference"]:+.3f}' in report
    assert f'SE {stats["points_difference_standard_error"]:.3f} points' in report
    assert (
        f'DNF rate SE {stats["dnf_rate_difference_standard_error_percentage_points"]:.3f} pp'
        in report
    )
    assert f'Both finished: {stats["both_finished_races"]}' in report
    assert f'Both DNFs: {stats["both_dnf_races"]}' in report
    assert f'Reference-only DNF: {stats["reference_only_dnf_races"]}' in report
    assert f'Variant-only DNF: {stats["variant_only_dnf_races"]}' in report
    distance = stats["completed_distance"]
    assert distance["paired_races"] == 3
    assert 'aria-label="0 completed distance changes"' in report
    assert "Completed distance for 0 compared with hard" in report
    assert f'{distance["mean_laps_difference"]:+.3f} laps' in report
    assert f'SE {distance["laps_difference_standard_error"]:.3f} laps' in report
    assert (
        f'{distance["more_laps_races"]} / {distance["equal_laps_races"]} / '
        f'{distance["fewer_laps_races"]}' in report
    )
    assert "missing distance is not zero" in report
    elapsed = stats["finished_race_time"]
    assert elapsed["paired_races"] > 0
    assert 'aria-label="0 elapsed time changes"' in report
    assert f'{elapsed["mean_seconds_difference"]:+.3f} s' in report
    assert f'{elapsed["paired_races"]} same-distance finish pairs' in report
    assert "not an overall strategy ranking" in report
    costs = stats["paid_stop_costs"]
    assert costs["paired_races"] == 3
    assert 'aria-label="0 paid-stop cost changes"' in report
    assert "Paid-stop costs for 0 compared with hard" in report
    assert f'{costs["mean_total_loss_seconds_difference"]:+.3f} s' in report
    assert f'SE {costs["total_loss_seconds_difference_standard_error"]:.3f} s' in report
    assert f'{costs["mean_lane_loss_seconds_difference"]:+.3f} s' in report
    assert f'{costs["mean_service_time_seconds_difference"]:+.3f} s' in report
    assert f'{costs["mean_queue_time_seconds_difference"]:+.3f} s' in report
    assert "missing or invalid details are excluded" in report
    ConsoleOutput.print_paired_comparison(variants, "hard", driver_id="0")
    printed = capsys.readouterr().out
    assert "Paired changes compared with hard" in printed
    assert f'{elapsed["mean_seconds_difference"]:+.3f} s' in printed
    assert f'{elapsed["excluded_pairs"]} excluded from time subset' in printed

    assert f'{stats["mean_points_difference"]:+.3f}' in printed
    assert (
        f'DNF rate SE: {stats["dnf_rate_difference_standard_error_percentage_points"]:.3f} pp'
        in printed
    )
    assert f'both finished={stats["both_finished_races"]}' in printed
    assert f'both DNF={stats["both_dnf_races"]}' in printed
    assert f'reference-only DNF={stats["reference_only_dnf_races"]}' in printed
    assert f'variant-only DNF={stats["variant_only_dnf_races"]}' in printed
    assert "Completed distance includes recorded laps for finishes and retirements" in printed
    assert f'{distance["mean_laps_difference"]:+.3f} laps' in printed
    assert f'({distance["excluded_pairs"]} excluded from distance subset)' in printed
    assert (
        f'more/equal/fewer laps={distance["more_laps_races"]}/'
        f'{distance["equal_laps_races"]}/{distance["fewer_laps_races"]}' in printed
    )
    assert "Paid-stop costs: 3 complete pairs" in printed
    assert f'{costs["mean_total_loss_seconds_difference"]:+.3f} s' in printed
    assert f'{costs["mean_lane_loss_seconds_difference"]:+.3f} s' in printed
    assert f'{costs["mean_service_time_seconds_difference"]:+.3f} s' in printed
    assert f'{costs["mean_queue_time_seconds_difference"]:+.3f} s' in printed
    assert "not a confidence interval" in printed
    replay = replay_saved_simulation(combined, simulation=2, scenario="soft")
    assert replay.race_results == [variants["soft"].race_results[1]]
    assert path.read_bytes() == source


def test_general_exports_omit_paired_analysis_without_reference(tmp_path):
    path = saved(tmp_path)
    variants = compare_saved_starting_tires(path, "0", ["soft", "hard"], num_simulations=1)
    exported = Exporter(tmp_path).export_scenario_comparison_json(variants)
    assert "paired_comparisons" not in json.loads(exported.read_text(encoding="utf-8"))
    assert "paired changes" not in render_comparison_report(variants)


def test_missing_pair_single_pair_and_incompatible_context_are_explained(tmp_path, capsys):
    variants = compare_saved_starting_tires(
        saved(tmp_path), "0", ["soft", "hard"], num_simulations=1,
    )
    report = render_comparison_report(variants, reference_scenario="hard")
    assert "SE needs at least 2 pairs" in report
    variants["soft"].race_results[0] = []
    report = render_comparison_report(variants, reference_scenario="hard")
    assert "No usable paired results (1 excluded pairs)" in report
    ConsoleOutput.print_paired_comparison(variants, "hard")
    assert "No usable paired results" in capsys.readouterr().out
    variants["soft"].input_snapshot["weather"]["humidity"] = .9
    report = render_comparison_report(variants, reference_scenario="hard")
    assert "Unavailable:" in report
    ConsoleOutput.print_paired_comparison(variants, "hard")
    assert "Unavailable:" in capsys.readouterr().out


def test_paired_output_explains_seed_qualification_and_driver_coverage(tmp_path, capsys):
    variants = compare_saved_starting_tires(
        saved(tmp_path), "0", ["soft", "hard"], num_simulations=1,
    )
    reference = variants["hard"]
    variant = variants["soft"]
    for sample in (reference, variant):
        sample.num_simulations = 3
        sample.race_results = [deepcopy(sample.race_results[0]) for _ in range(3)]
        sample.qualifying_results = [deepcopy(sample.qualifying_results[0]) for _ in range(3)]
    reference.seed = 81
    variant.seed = 82
    variant.qualifying_results[0][0].best_time += 0.123
    variant.race_results[1] = [row for row in variant.race_results[1] if row.driver_id != "0"]

    expected = paired_comparison_statistics(variants, "hard")["variants"]["soft"]
    assert (expected["seed_from"], expected["seed_to"], expected["available_seed_pairs"],
            expected["qualifying_mismatches"]) == (82, 83, 2, 1)
    assert expected["driver_statistics"]["0"]["paired_races"] == 0
    assert expected["driver_statistics"]["0"]["excluded_pairs"] == 2
    assert expected["driver_statistics"]["1"]["paired_races"] == 1
    assert expected["driver_statistics"]["1"]["excluded_pairs"] == 1

    exported = Exporter(tmp_path).export_scenario_comparison_json(
        variants, reference_scenario="hard",
    )
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["paired_comparisons"]["variants"]["soft"] == expected

    report = render_comparison_report(variants, reference_scenario="hard")
    coverage = "Overlapping recorded seeds 82–83 (2 trials); 1 qualifying mismatch excluded."
    explanation = (
        "A qualifying mismatch means the qualifying record was missing, invalid, or different."
    )
    detail = "of these, 1 qualifying mismatch and 1 missing/invalid driver observation"
    assert coverage in report
    assert explanation in report
    assert "No usable paired results (2 excluded pairs)" in report
    assert detail in report

    ConsoleOutput.print_paired_comparison(variants, "hard")
    printed = capsys.readouterr().out
    assert coverage in printed
    assert explanation in printed
    assert "No usable paired results (2 excluded pairs)" in printed
    assert detail in printed


def test_reference_labels_are_escaped_and_invalid_reference_writes_nothing(tmp_path):
    variants = compare_saved_starting_tires(
        saved(tmp_path), "0", ["soft", "hard"], num_simulations=1,
    )
    hostile = '<script>window.bad=true</script>'
    variants[hostile] = variants.pop("hard")
    report = render_comparison_report(variants, reference_scenario=hostile)
    assert "compared with &lt;script&gt;" in report
    assert "<script>" not in report
    exporter = Exporter(tmp_path)
    for method, filename in ((exporter.export_scenario_comparison_json, "invalid.json"),
                             (exporter.export_scenario_comparison_html, "invalid.html")):
        with pytest.raises(ValueError, match="reference"):
            method(variants, filename=filename, reference_scenario="missing")
        assert not (tmp_path / filename).exists()


@pytest.mark.parametrize("script,arguments,reference", [
    ("compare_starting_tyres.py", ["--driver", "0", "--compounds", "soft,hard"], "hard"),
    ("compare_race_engines.py", [], "chronological"),
])
def test_cli_reference_reaches_console_and_exports(tmp_path, script, arguments, reference):
    path = saved(tmp_path)
    destination = tmp_path / "exports"
    command = [sys.executable, str(Path(__file__).parents[1] / "examples" / script),
               str(path), *arguments, "--simulations", "2", "--reference", reference,
               "--export", "--output-dir", str(destination)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert f"Paired changes compared with {reference}" in completed.stdout
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in destination.glob("*.json")]
    comparisons = [p["paired_comparisons"] for p in payloads if "paired_comparisons" in p]
    assert len(comparisons) == 1
    assert comparisons[0]["reference_scenario"] == reference
    assert all(v["status"] == "paired" for v in comparisons[0]["variants"].values())
    invalid = subprocess.run(command[:2] + ["missing.json", *arguments, "--reference", "missing"],
                             capture_output=True, text=True, check=False)
    assert invalid.returncode == 2
    assert "reference must be one of" in invalid.stderr
