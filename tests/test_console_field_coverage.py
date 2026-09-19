"""Console tables retain the complete observed driver field."""

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output import ConsoleOutput


def _stats(driver_id: str, positions: list[int], *, wins: int = 0) -> DriverStatistics:
    return DriverStatistics(
        driver_id=driver_id,
        driver_name=f"Driver {driver_id}",
        team="Test",
        wins=wins,
        positions=positions,
    )


def _results(stats: dict[str, DriverStatistics]) -> SimulationResults:
    return SimulationResults(
        num_simulations=1,
        track_name="Test",
        driver_stats=stats,
        race_results=[],
        qualifying_results=[],
    )


def _table_rows(output: str) -> dict[str, list[str]]:
    rows = {}
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0].startswith("D") and fields[0][1:].isdigit():
            rows[fields[0]] = fields[1:]
    return rows


def test_scenario_comparison_keeps_union_order_and_late_variant_winner(capsys):
    driver_ids = [f"D{number:02d}" for number in range(1, 23)]
    baseline = {
        driver_id: _stats(driver_id, [number], wins=int(number == 1))
        for number, driver_id in enumerate(driver_ids, 1)
    }
    variant = {
        driver_id: _stats(driver_id, [number + 1])
        for number, driver_id in enumerate(driver_ids[:-1], 1)
    }
    variant["D22"] = _stats("D22", [1], wins=1)

    ConsoleOutput.print_scenario_comparison(
        {"baseline": _results(baseline), "variant": _results(variant)},
        top_n=10,
    )

    rows = _table_rows(capsys.readouterr().out)
    assert list(rows) == driver_ids
    assert rows["D22"] == ["0.0%", "100.0%"]


def test_scenario_comparison_includes_late_only_driver_after_empty_scenario(capsys):
    ConsoleOutput.print_scenario_comparison(
        {"empty": _results({}), "late": _results({"LATE": _stats("LATE", [1], wins=1)})},
        top_n=1,
    )

    output = capsys.readouterr().out
    late_row = next(line.split() for line in output.splitlines() if line.startswith("LATE"))
    assert late_row == ["LATE", "--", "100.0%"]


def test_scenario_comparison_distinguishes_unobserved_and_zero_win_cells(capsys):
    baseline = {
        "zero": _stats("zero", [2]),
        "unobserved": _stats("unobserved", []),
    }
    variant = {
        "zero": _stats("zero", [1], wins=1),
        "missing": _stats("missing", [1], wins=1),
    }

    ConsoleOutput.print_scenario_comparison(
        {"baseline": _results(baseline), "variant": _results(variant)},
        top_n=1,
    )

    rows = {
        fields[0]: fields[1:]
        for line in capsys.readouterr().out.splitlines()
        if (fields := line.split())
        and fields[0] in {"zero", "unobserved", "missing"}
    }
    assert rows == {
        "zero": ["0.0%", "100.0%"],
        "unobserved": ["--", "--"],
        "missing": ["--", "100.0%"],
    }


def test_driver_deep_dive_shows_all_observed_positions(capsys):
    results = _results({"BACK": _stats("BACK", [21, 22, 21])})

    ConsoleOutput.print_driver_deep_dive(results, "BACK")

    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("  P21:") for line in lines)
    assert any(line.startswith("  P22:") for line in lines)
