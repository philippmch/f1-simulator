"""Published race order must survive engines, workers, replay and live cutoffs."""

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest
from test_dashboard_plan_comparison import MultiDriverLoader
from test_practice_qualifying import event, inputs, loader
from test_qualifying_session_weather import runner

from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.analysis.strategy_comparison import (
    _runner_variant,
    compare_saved_race_engines,
    compare_saved_starting_tires,
)
from f1sim.data.current import CurrentSeasonDataError
from f1sim.data.grid import fetch_current_starting_grid, parse_starting_grid
from f1sim.output import Exporter
from f1sim.output.grid_context import race_grid_context
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import RaceSimulator
from f1sim.web import server


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_supplied_race_order_reaches_the_real_engine_without_relabeling_qualifying(
    monkeypatch, engine,
):
    target = ChronologicalRace if engine == "chronological" else RaceSimulator
    name = "run" if engine == "chronological" else "simulate_race"
    original = getattr(target, name)
    grids = []

    def capture(self, *args, **kwargs):
        grids.append(kwargs["starting_grid"].copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(target, name, capture)
    plain = runner(engine=engine).run(2, parallel=False)
    fixed = runner(engine=engine, starting_grid=["B", "A"]).run(2, parallel=False)
    assert grids[-2:] == [["B", "A"], ["B", "A"]]
    assert fixed.qualifying_results == plain.qualifying_results
    assert fixed.input_snapshot["starting_grid"] == ["B", "A"]
    assert "Qualifying lap results remain simulated" in race_grid_context(fixed.input_snapshot)


@pytest.mark.parametrize("value", [[], "AB", ("A", "B"), ["A"], ["A", "A"],
                                   ["A", "C"], ["A", True], ["A", " "]])
def test_invalid_roster_rejected_before_any_trial(value):
    with pytest.raises(ValueError, match="starting_grid"):
        runner(starting_grid=value)


def test_grid_is_copied_and_post_construction_corruption_is_rejected(monkeypatch):
    grid = ["B", "A"]
    configured = runner(starting_grid=grid)
    grid.reverse()
    assert configured.starting_grid == ["B", "A"]
    configured.starting_grid.pop()
    monkeypatch.setattr("f1sim.analysis.montecarlo._run_single_simulation",
                        lambda *args: pytest.fail("invalid grid reached a trial"))
    with pytest.raises(ValueError, match="every modeled driver"):
        configured.run(1, parallel=False)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_grid_worker_and_second_trial_replay_are_exact(tmp_path, engine):
    configured = runner(engine=engine, starting_grid=["B", "A"])
    serial = configured.run(2, parallel=False)
    parallel = configured.run(2, parallel=True, max_workers=1)
    assert parallel.race_results == serial.race_results
    assert parallel.qualifying_results == serial.qualifying_results
    assert parallel.event_stats == serial.event_stats
    path = Exporter(tmp_path).export_statistics_json(serial)
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.race_results == [serial.race_results[1]]
    assert replay.qualifying_results == [serial.qualifying_results[1]]
    assert replay.input_snapshot["schema_version"] == 14
    assert replay.input_snapshot["starting_grid"] == ["B", "A"]


@pytest.mark.parametrize("schedule", [None, [],
    [{"lap": 2, "control": "safety_car", "duration_laps": 1}],
    [{"lap": 2, "control": "red_flag", "action": "resume_wet"}],
])
def test_new_schema_combines_grid_with_existing_weather_and_strategy_inputs(tmp_path, schedule):
    configured = runner(
        starting_grid=["B", "A"], control_schedule=schedule,
        weather_schedule=[{"lap": 3, "rain_intensity": .3}],
        qualifying_weather={"Q1": {"rain_intensity": .1}},
        starting_tires={"A": "soft"}, starting_tire_ages={"A": 1},
        tire_inventory={"A": [{"id": "s", "compound": "soft", "age": 1,
                                "remaining_laps": 3}, {"id": "m", "compound": "medium"}]},
        pit_plans={"A": [{"lap": 3, "earliest_lap": 2, "trigger": "neutralized",
                          "compound": "medium"}]}, tire_warmup={"medium": .5},
    )
    original = configured.run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    replay = replay_saved_simulation(path)
    assert replay.race_results == original.race_results
    assert replay.input_snapshot == original.input_snapshot
    assert paired_comparison_statistics({"a": original, "b": replay}, "a")[
        "variants"]["b"]["status"] == "paired"


@pytest.mark.parametrize("change", ["missing", "null", "old_schema", "bad_roster",
                                   "control_policy", "orphan_control_policy"])
def test_corrupted_or_downgraded_grid_snapshots_cannot_silently_replay(tmp_path, change):
    original = runner(starting_grid=["B", "A"], control_schedule=[]).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    data = json.loads(path.read_text(encoding="utf-8"))
    snapshot = data["simulation_inputs"]
    if change == "missing":
        del snapshot["starting_grid"]
    elif change == "null":
        snapshot["starting_grid"] = None
    elif change == "old_schema":
        snapshot["schema_version"] = 11
    elif change == "bad_roster":
        snapshot["starting_grid"] = ["A"]
    elif change == "control_policy":
        snapshot["control_schedule_policy"] = "unknown"
    else:
        del snapshot["control_schedule"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        replay_saved_simulation(path)


def test_different_grids_are_incompatible_and_strategy_variants_keep_the_grid(tmp_path):
    configured = runner(starting_grid=["B", "A"])
    reference = configured.run(1, parallel=False)
    different = runner(starting_grid=["A", "B"]).run(1, parallel=False)
    status = paired_comparison_statistics({"a": reference, "b": different}, "a")
    assert status["variants"]["b"]["status"] != "paired"
    assert _runner_variant(configured).starting_grid == ["B", "A"]
    path = Exporter(tmp_path).export_statistics_json(reference)
    variants = compare_saved_race_engines(path, num_simulations=1)
    variants.update(compare_saved_starting_tires(path, "A", num_simulations=1))
    assert all(r.input_snapshot["starting_grid"] == ["B", "A"] for r in variants.values())


def grid_html():
    return "<h1>Test 2026 - STARTING GRID</h1><table>" + "".join(
        f"<tr><td>{i}</td><td>1</td><td>{code}</td><td>{team}</td><td>1:20.0</td></tr>"
        for i, (code, team) in enumerate((("B2", "b"), ("A1", "a"),
                                         ("B1", "b"), ("A2", "a")), 1)
    ) + "</table>"


def grid_index():
    return ('<h1>2026 RACE RESULTS</h1>'
            '<a href="/en/results/2026/races/1234/test/race-result">Test</a>')


@pytest.mark.parametrize("change", ["missing", "duplicate", "team", "season",
                                   "pit_lane_slot", "pit_lane_note"])
def test_live_grid_parser_rejects_ambiguous_or_unsupported_rows(monkeypatch, change):
    adapter = loader(monkeypatch, lambda _: "")
    html = grid_html()
    if change == "missing":
        html = html.replace("<td>4</td>", "<td>5</td>")
    elif change == "duplicate":
        html = html.replace("<td>A2</td>", "<td>A1</td>")
    elif change == "team":
        html = html.replace("<td>b</td>", "<td>a</td>", 1)
    elif change == "season":
        html = html.replace("2026", "2025")
    elif change == "pit_lane_slot":
        html = html.replace("<td>4</td>", "<td>PL</td>")
    else:
        html += "<p>A2 required to start from the pit lane after modifications.</p>"
    with pytest.raises(CurrentSeasonDataError):
        parse_starting_grid(html, year=2026, loader=adapter, drivers=inputs()[0])


def test_live_grid_uses_gp_qualifying_cutoff_and_fresh_observed_event_link(monkeypatch):
    calls = []

    def getter(url, **kwargs):
        calls.append(url)
        assert kwargs["headers"]["Cache-Control"] == "no-cache, no-store, max-age=0"
        return grid_index() if url.endswith("/races") else grid_html()

    adapter = loader(monkeypatch, getter)
    target = event()
    target.update(race="Test Grand Prix", sprint=True)
    target["sessions"]["SprintQualifying"] = {"date": "2026-03-27", "time": "12:00:00Z"}
    drivers = inputs()[0]
    before = datetime(2026, 3, 28, 14, 59, tzinfo=timezone.utc)
    assert fetch_current_starting_grid(adapter, 2026, target, drivers, now=before) is None
    assert calls == []
    monkeypatch.setattr(adapter, "_event_for_race", lambda *a: target)
    grid = adapter.get_starting_grid(2026, 2, drivers)
    assert grid == ["B2", "A1", "B1", "A2"]
    assert len(calls) == 2 and calls[-1].endswith("/1234/test/starting-grid")
    assert adapter.get_provenance()["race_grid"]["mode"] == "published"
    grid.reverse()
    assert adapter.get_provenance()["race_grid"]["starting_grid"][0] == "B2"


def test_unavailable_published_grid_falls_back_with_a_reason(monkeypatch):
    adapter = loader(monkeypatch, lambda *a, **k: "invalid page")
    monkeypatch.setattr(adapter, "_event_for_race", lambda *a: event())
    assert adapter.get_starting_grid(2026, 2, inputs()[0]) is None
    assert adapter.get_provenance()["race_grid"]["reason"] == "published_grid_unavailable"


@pytest.mark.parametrize("mode,weather,expected", [
    ("auto", None, ["B", "A"]), ("simulated", None, None),
    ("auto", {"Q1": {"rain_intensity": .2}}, None),
])
def test_dashboard_shares_grid_across_scenarios_and_reference_variants(
    monkeypatch, mode, weather, expected,
):
    adapter = MultiDriverLoader()
    calls = []
    adapter.get_starting_grid = lambda *a: calls.append(a) or ["B", "A"]
    monkeypatch.setattr(server, "_get_loader", lambda: adapter)
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    request = server.DashboardRunRequest(
        year=2026, simulations=10, scenarios="dry", parallel=False,
        pit_plans={"A": []}, compare_automatic=True,
        race_grid_mode=mode, qualifying_weather=weather,
    )
    before = deepcopy(request)
    payload = server.run_dashboard_simulation(request)
    assert request == before
    for variant in (payload, payload["automatic_reference"]):
        for scenario in variant["scenarios"].values():
            assert scenario["simulation_inputs"].get("starting_grid") == expected
    assert bool(calls) == (expected is not None)
