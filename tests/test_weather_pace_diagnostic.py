"""Weather labels cannot change executed pace under identical physical inputs."""

import json
import runpy
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def report():
    script = runpy.run_path(str(Path(__file__).resolve().parents[1]
                               / "examples" / "check_weather_pace.py"))
    return script["check_cases"]()


@pytest.mark.parametrize("case", ["explicit_damp", "automatic_mild_damp"])
def test_numeric_race_and_qualifying_pace_ignore_labels(report, case):
    for compound in {row["compound"] for row in report["label_paces"]}:
        group = [row for row in report["label_paces"]
                 if row["compound"] == compound and row["case"] == case]
        assert len(group) == 4
        assert len({(row["race"], row["qualifying"]) for row in group}) == 1


@pytest.mark.parametrize("case", ["explicit_damp", "automatic_mild_damp"])
def test_actual_races_remain_identical_across_labels(report, case):
    assert len(report["races"]) == 32
    for engine in ("standard", "chronological"):
        for inventory in ("finite", "unlimited"):
            group = [row for row in report["races"]
                     if row["engine"] == engine and row["inventory"] == inventory
                     and row["case"] == case]
            assert len(group) == 4
            physical_results = []
            for row in group:
                physical = deepcopy(row["result"])
                for stop in physical["pit_stop_details"]:
                    assert stop.pop("condition") == row["condition"]
                physical_results.append(physical)
            assert all(result == physical_results[0] for result in physical_results)
            assert all(row["final_rng_state"] == group[0]["final_rng_state"] for row in group)
            for row in group:
                result = row["result"]
                assert result["status"] == "finished"
                assert result["laps_completed"] == report["inputs"]["race_laps"]
                if case == "explicit_damp":
                    assert result["pit_stops"] >= 1
                else:
                    assert result["strategy"][0] == "intermediate"
                assert "intermediate" in result["strategy"]
                assert row["physical_fuel_distances"] == [6]


def test_finite_race_conserves_all_physical_wear(report):
    for row in report["races"]:
        if row["inventory"] != "finite":
            continue
        result = row["result"]
        ages = {item["id"]: item["age"] for item in report["inputs"]["finite_sets"]}
        if row["case"] == "explicit_damp":
            assert result["tire_set_history"][0]["age_at_fit"] == 18
        for stint in result["tire_set_history"]:
            assert stint["age_at_fit"] == ages[stint["set_id"]]
            assert stint["age_at_end"] == stint["age_at_fit"] + stint["laps_used"]
            ages[stint["set_id"]] = stint["age_at_end"]
        assert sum(stint["laps_used"] for stint in result["tire_set_history"]) == 6
        assert {item["id"]: item["age"] for item in result["tire_inventory"]} == ages


def test_tiny_water_perturbations_have_small_executed_pace_changes(report):
    assert len(report["boundary_sweeps"]) == 30
    changes = []
    for sweep in report["boundary_sweeps"]:
        for metric in ("race", "qualifying"):
            values = [sample[metric] for sample in sweep["samples"]]
            changes.append(max(values) - min(values))
    assert max(changes) < .001
    assert max(changes) > 0
    json.dumps(report, allow_nan=False)
