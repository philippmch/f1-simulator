"""An explicit sensitivity profile survives workers, saved inputs and comparisons."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.analysis.strategy_comparison import compare_saved_pit_plans
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter


def runner(engine="standard", profile=None, finite=False):
    return MonteCarloRunner(
        [Driver(id="A", name="A", team_id="A")],
        {"A": Car(team_id="A", team_name="A")},
        Track(id="t", name="Sensitivity", country="T", total_laps=6, base_lap_time=90),
        Weather(change_probability=0), seed=19, race_engine=engine,
        starting_tires={"A": "soft"}, tire_warmup=profile,
        tire_inventory={"A": [{"id": "s", "compound": "soft"},
                              {"id": "m", "compound": "medium"}]} if finite else None,
        pit_plans={"A": [{"lap": 3, "compound": "medium"}]},
    )


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
def test_profile_survives_parallel_replay_and_plan_variants(tmp_path, engine, finite):
    profile = {"soft": .5, "medium": 2.0}
    configured = runner(engine, profile, finite)
    serial = configured.run(2, parallel=False)
    parallel = configured.run(2, parallel=True, max_workers=2)
    assert serial.race_results == parallel.race_results
    assert serial.qualifying_results == parallel.qualifying_results
    assert serial.input_snapshot == parallel.input_snapshot
    assert serial.input_snapshot["schema_version"] == 6
    assert serial.input_snapshot["tire_warmup"] == profile
    assert serial.input_snapshot["tire_warmup_policy"] == "post_fit_first_lap_v1"
    path = Exporter(tmp_path).export_statistics_json(serial)
    for trial in (1, 2):
        replay = replay_saved_simulation(path, simulation=trial)
        assert replay.race_results == [serial.race_results[trial - 1]]
        assert replay.input_snapshot["tire_warmup"] == profile
    variants = compare_saved_pit_plans(
        path, "A", {"saved": [{"lap": 3, "compound": "medium"}], "automatic": None},
        num_simulations=2,
    )
    assert variants["saved"].race_results == serial.race_results
    assert all(result.input_snapshot["tire_warmup"] == profile for result in variants.values())
    assert paired_comparison_statistics(variants, "saved")["variants"]["automatic"][
        "status"
    ] == "paired"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_zero_profile_retains_legacy_snapshot_and_seeded_outputs(engine):
    original = runner(engine).run(2, parallel=False)
    zero = runner(engine, {"soft": 0, "medium": 0}).run(2, parallel=False)
    assert zero.input_snapshot == original.input_snapshot
    assert zero.race_results == original.race_results
    assert zero.qualifying_results == original.qualifying_results
    assert "tire_warmup" not in zero.input_snapshot


def test_different_assumptions_are_not_paired():
    first = runner(profile={"medium": 1}).run(1, parallel=False)
    second = deepcopy(first)
    second.input_snapshot["tire_warmup"]["medium"] = 2
    assert paired_comparison_statistics({"a": first, "b": second}, "a")["variants"]["b"][
        "status"
    ] == "unavailable"


@pytest.mark.parametrize("change", [
    {"tire_warmup_policy": "unknown"}, {"tire_warmup": {}},
    {"tire_warmup": {"soft": True}}, {"tire_warmup": {"soft": float("nan")}},
    {"tire_warmup": {"soft": 61}}, {"schema_version": 5},
    {"pit_plans": {"A": [{"lap": True, "compound": "medium"}]}},
])
def test_saved_profile_is_strictly_validated(tmp_path, change):
    result = runner(profile={"medium": 1}).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(result)
    payload = json.loads(path.read_text())
    payload["simulation_inputs"].update(change)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        replay_saved_simulation(path)


def test_invalid_dashboard_profile_rejected_before_loading(monkeypatch):
    from f1sim.web.server import DashboardRunRequest, run_dashboard_simulation

    def no_load(*args, **kwargs):
        pytest.fail("invalid profile must fail before live loading")

    monkeypatch.setattr("f1sim.web.server.CurrentSeasonDataLoader", no_load)
    with pytest.raises(ValueError, match="warmup|warm-up"):
        run_dashboard_simulation(DashboardRunRequest(tire_warmup={"soft": True}))


@pytest.mark.parametrize("profile", ["soft=nan", "soft=61", "soft=1,soft=2", "slick=1"])
def test_cli_rejects_invalid_profile_before_loading(profile):
    script = Path(__file__).resolve().parents[1] / "examples" / "simulate_race.py"
    result = subprocess.run(
        [sys.executable, str(script), "--tire-warmup", profile],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "--tire-warmup" in result.stderr
    assert "Loading" not in result.stdout


def test_dashboard_runner_and_metadata_preserve_assumed_profile():
    from f1sim.web.server import (
        DashboardRunRequest,
        _dashboard_request_metadata,
        _dashboard_runner,
    )

    base = runner()
    request = DashboardRunRequest(tire_warmup={"medium": 1, "soft": 0})
    arguments = dict(
        request=request, tire_inventory=None, starting_tires={}, starting_tire_ages={},
        pit_plans={},
    )
    configured = _dashboard_runner(
        **arguments, drivers=base.drivers, cars=base.cars, track=base.track,
        weather=base.weather, seed=19, copy_inputs=True,
    )
    metadata = _dashboard_request_metadata(
        **arguments, canonical_race="1", effective_max_workers=1, compare_automatic=True,
    )
    assert configured.tire_warmup == metadata["tire_warmup"] == {"medium": 1.0}
    for disabled in (None, {}, {"medium": 0}):
        off = _dashboard_request_metadata(
            **(arguments | {"request": DashboardRunRequest(tire_warmup=disabled)}),
            canonical_race="1", effective_max_workers=1, compare_automatic=True,
        )
        assert "tire_warmup" not in off and "tire_warmup_policy" not in off
    request.tire_warmup["medium"] = 5
    assert configured.tire_warmup == metadata["tire_warmup"] == {"medium": 1.0}


def test_reports_identify_saved_assumptions(tmp_path, capsys):
    from f1sim.output import ConsoleOutput
    from f1sim.output.comparison import render_comparison_report

    result = runner(profile={"medium": 1}).run(1, parallel=False)
    ConsoleOutput.print_monte_carlo_summary(result)
    assert "medium=1 s" in capsys.readouterr().out
    report = render_comparison_report({"assumed": result})
    assert "medium=1 s" in report
    assert "not calibrated physics" in report
    exported = Exporter(tmp_path).export_report_html(result)
    assert "medium=1 s" in exported.read_text()
