"""Published pit starters are distinct from ordinary grid slots and paid stops."""

import json
from copy import deepcopy

import pytest
from test_chronological_race import fixture
from test_dashboard_plan_comparison import MultiDriverLoader
from test_practice_qualifying import inputs, loader
from test_qualifying_session_weather import runner
from test_starting_grid import grid_html

from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.analysis.strategy_comparison import _runner_variant, compare_saved_starting_tires
from f1sim.data.current import CurrentSeasonDataError
from f1sim.data.grid import parse_starting_grid
from f1sim.models import TireCompound
from f1sim.output import Exporter
from f1sim.simulation.execution import PIT_LANE_START_DELAY_SECONDS, PIT_LANE_START_POLICY
from f1sim.web import server


def test_chronological_pit_starter_enters_late_without_paid_service_or_extra_lap(monkeypatch):
    engine, args, calls, _, _, _ = fixture(monkeypatch, {"Fast": 90, "Slow": 90})
    results = engine.run(*args, starting_tires={key: TireCompound.INTERMEDIATE for key in args[-1]},
                         pit_lane_starters=["Slow"])
    assert engine.pit_exits == [("Slow", 1, PIT_LANE_START_DELAY_SECONDS)]
    assert [(r.driver_id, r.total_time, r.laps_completed) for r in results] == [
        ("Fast", 900, 10), ("Slow", 905, 10),
    ]
    assert len(calls) == 20
    assert all(r.pit_stops == 0 and r.pit_laps == [] for r in results)
    assert engine.pit_service_records == []


def test_multiple_pit_starters_keep_the_published_queue(monkeypatch):
    engine, args, _, _, _, _ = fixture(monkeypatch, {"Fast": 90, "Slow": 90, "Last": 90})
    engine.run(*args, starting_tires={key: TireCompound.INTERMEDIATE for key in args[-1]},
               pit_lane_starters=["Slow", "Last"])
    assert engine.pit_exits == [("Slow", 1, 5.0), ("Last", 1, 5.0)]


def test_standard_pit_start_pays_the_same_delay_without_paid_service(monkeypatch):
    engine, args, _, _, _, _ = fixture(monkeypatch, {"Fast": 90, "Slow": 90})
    monkeypatch.setattr(engine.simulator.lap_simulator, "calculate_lap_time", lambda *a, **k: 90.)
    results = engine.simulator.simulate_race(
        *args, starting_tires={key: TireCompound.INTERMEDIATE for key in args[-1]},
        pit_lane_starters=["Slow"],
    )
    assert [(r.driver_id, r.total_time, r.laps_completed) for r in results] == [
        ("Fast", 900, 10), ("Slow", 905, 10),
    ]
    assert all(r.pit_stops == 0 and r.pit_laps == [] for r in results)


@pytest.mark.parametrize("value", [True, "B", (), ["A"], ["B", "B"], ["C"], ["A", "B"]])
def test_invalid_pit_roster_is_rejected_before_trials(value):
    with pytest.raises(ValueError, match="pit_lane_starters"):
        runner(starting_grid=["A", "B"], pit_lane_starters=value)


def test_pit_starters_require_an_explicit_grid():
    with pytest.raises(ValueError, match="pit_lane_starters"):
        runner(pit_lane_starters=["B"])


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_pit_grid_serial_workers_replay_and_variants_are_exact(tmp_path, engine):
    configured = runner(engine=engine, starting_grid=["A", "B"], pit_lane_starters=["B"])
    serial = configured.run(2, parallel=False)
    parallel = configured.run(2, parallel=True, max_workers=1)
    assert serial.race_results == parallel.race_results
    assert serial.qualifying_results == parallel.qualifying_results
    snapshot = serial.input_snapshot
    assert snapshot["schema_version"] == 15
    assert snapshot["pit_lane_starters"] == ["B"]
    assert snapshot["pit_lane_start_policy"] == PIT_LANE_START_POLICY
    path = Exporter(tmp_path).export_statistics_json(serial)
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.race_results == [serial.race_results[1]]
    assert replay.qualifying_results == [serial.qualifying_results[1]]
    assert replay.input_snapshot == snapshot
    assert _runner_variant(configured).pit_lane_starters == ["B"]
    assert all(value.input_snapshot["pit_lane_starters"] == ["B"]
               for value in compare_saved_starting_tires(path, "A", num_simulations=1).values())
    ordinary = runner(engine=engine, starting_grid=["A", "B"]).run(2, parallel=False)
    comparison = paired_comparison_statistics({"pit": serial, "grid": ordinary}, "pit")
    assert comparison["variants"]["grid"]["status"] != "paired"
    assert serial.qualifying_results == ordinary.qualifying_results


@pytest.mark.parametrize("change", ["missing", "empty", "policy", "downgraded", "no_grid"])
def test_replay_cannot_drop_or_change_the_pit_start_context(tmp_path, change):
    result = runner(starting_grid=["A", "B"], pit_lane_starters=["B"]).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(result)
    document = json.loads(path.read_text(encoding="utf-8"))
    snapshot = document["simulation_inputs"]
    if change == "missing":
        del snapshot["pit_lane_starters"]
    elif change == "empty":
        snapshot["pit_lane_starters"] = []
    elif change == "policy":
        snapshot["pit_lane_start_policy"] = "unknown"
    elif change == "downgraded":
        snapshot["schema_version"] = 14
    else:
        del snapshot["starting_grid"]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        replay_saved_simulation(path)


@pytest.mark.parametrize("notes,pits", [
    ("A2 required to start from the pit lane after modifications.", ["A2"]),
    ("A2 and B2 required to start from pit lane.", ["B2", "A2"]),
    ("A2 granted permission to race after disqualification. Required to start from the pit lane.",
     ["A2"]),
])
def test_explicit_pit_instructions_are_resolved_without_reapplying_grid_penalties(
    monkeypatch, notes, pits,
):
    adapter = loader(monkeypatch, lambda _: "")
    html = grid_html() + f"<p>Note - {notes}</p>"
    result = parse_starting_grid(html, year=2026, loader=adapter, drivers=inputs()[0],
                                include_pit_lane=True)
    assert result["pit_lane_starters"] == pits
    assert result["starting_grid"][-len(pits):] == pits
    assert set(result["starting_grid"]) == {d.id for d in inputs()[0]}


@pytest.mark.parametrize("pit_row_first", [False, True])
def test_numeric_pit_row_can_share_the_last_ordinary_grid_number(monkeypatch, pit_row_first):
    adapter = loader(monkeypatch, lambda _: "")
    tail = [(3, "A2", "a"), (3, "B1", "b")] if pit_row_first else [
        (3, "B1", "b"), (3, "A2", "a"),
    ]
    rows = [(1, "B2", "b"), (2, "A1", "a"), *tail]
    html = "<h1>2026 STARTING GRID</h1><table>" + "".join(
        f"<tr><td>{slot}</td><td>1</td><td>{code}</td><td>{team}</td></tr>"
        for slot, code, team in rows
    ) + "</table><p>Note - A2 required to start from the pit lane.</p>"
    context = parse_starting_grid(html, year=2026, loader=adapter, drivers=inputs()[0],
                                 include_pit_lane=True)
    assert context == {"starting_grid": ["B2", "A1", "B1", "A2"],
                       "pit_lane_starters": ["A2"]}


def test_duplicate_ordinary_slots_remain_invalid_with_a_pit_starter(monkeypatch):
    adapter = loader(monkeypatch, lambda _: "")
    html = grid_html().replace("<td>3</td>", "<td>2</td>")
    html += "<p>Note - A2 required to start from the pit lane.</p>"
    with pytest.raises(CurrentSeasonDataError, match="Conflicting"):
        parse_starting_grid(html, year=2026, loader=adapter, drivers=inputs()[0],
                            include_pit_lane=True)


@pytest.mark.parametrize("note", [
    "Unknown required to start from the pit lane.",
    "A2 received a grid penalty. Unknown required to start from pit lane.",
    "A2 and Unknown required to start from pit lane.",
])
def test_unidentified_pit_instruction_never_becomes_an_ordinary_grid(monkeypatch, note):
    adapter = loader(monkeypatch, lambda _: "")
    html = grid_html() + f"<p>{note}</p>"
    with pytest.raises(CurrentSeasonDataError, match="Unresolved"):
        parse_starting_grid(html, year=2026, loader=adapter, drivers=inputs()[0],
                            include_pit_lane=True)


def test_script_payloads_cannot_add_visible_pit_instructions(monkeypatch):
    adapter = loader(monkeypatch, lambda _: "")
    html = grid_html() + "<script>A2 required to start from pit lane.</script>"
    result = parse_starting_grid(html, year=2026, loader=adapter, drivers=inputs()[0],
                                include_pit_lane=True)
    assert result["pit_lane_starters"] == []


def test_dashboard_auto_grid_carries_the_pit_context_through_every_reference(monkeypatch):
    adapter = MultiDriverLoader()
    adapter.get_starting_grid = lambda *a: ["B", "A"]
    adapter.get_pit_lane_starters = lambda: ["A"]
    monkeypatch.setattr(server, "_get_loader", lambda: adapter)
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    request = server.DashboardRunRequest(
        year=2026, simulations=10, scenarios="dry", parallel=False,
        pit_plans={"A": []}, compare_automatic=True,
    )
    before = deepcopy(request)
    payload = server.run_dashboard_simulation(request)
    assert request == before
    for variant in (payload, payload["automatic_reference"]):
        for scenario in variant["scenarios"].values():
            assert scenario["simulation_inputs"]["starting_grid"] == ["B", "A"]
            assert scenario["simulation_inputs"]["pit_lane_starters"] == ["A"]
            assert scenario["simulation_inputs"]["schema_version"] == 15
