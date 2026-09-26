"""Compare adaptive fitting-cost choices with independently executed schedules."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("case_name, expected_stops", [
    ("low_lane_high_wear", 1), ("long_reference_high_wear", 2),
])
def test_assumed_cost_changes_strategy_to_executed_bounded_optimum(
    monkeypatch, engine, case_name, expected_stops,
):
    diagnostic = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                    / "examples" / "check_dry_pit_schedules.py"))
    run = diagnostic["run_race"]
    namespace = run.__globals__
    original = namespace["RaceSimulator"]
    profile = {compound.value: .5 for compound in diagnostic["SLICKS"]}

    def configured(*args, **kwargs):
        return original(*args, **kwargs, tire_warmup=profile)

    monkeypatch.setitem(namespace, "RaceSimulator", configured)
    case = next(case for case in diagnostic["CASES"] if case["name"] == case_name)
    result = diagnostic["compare_schedules"](cases=(case,), engines=(engine,))[0]
    assert result["schedules_checked"] == 1092
    assert result["gap_seconds"] == pytest.approx(0, abs=1e-8)
    selected = result["selected"]
    assert selected["paid_stops"] == expected_stops

    # Independently isolate the fee on the selected fixed schedule: opening
    # tyres are ready, each later fit costs once, and ordinary pit loss is intact.
    schedule = tuple(zip(selected["pit_laps"], selected["compounds"][1:], strict=True))
    charged = run(case, engine, schedule)
    monkeypatch.setitem(namespace, "RaceSimulator", original)
    uncharged = run(case, engine, schedule)
    assert charged["total_seconds"] - uncharged["total_seconds"] == pytest.approx(
        .5 * expected_stops, abs=1e-8,
    )
