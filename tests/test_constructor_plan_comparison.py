"""Constructor-wide saved pit-plan comparisons preserve paired inputs."""

import json
from copy import deepcopy

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.strategy_comparison import compare_saved_constructor_pit_plans
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter


def _saved(tmp_path, *, engine="standard", one_member=False):
    drivers = [
        Driver(id="A", name="A", team_id="T"),
        *([] if one_member else [Driver(id="B", name="B", team_id="T")]),
        Driver(id="C", name="C", team_id="U"),
    ]
    inventory = {
        driver_id: [
            {"id": f"{driver_id}-medium", "compound": "medium", "age": 0},
            {"id": f"{driver_id}-hard", "compound": "hard", "age": 0},
        ]
        for driver_id in ("A", "B")
        if any(driver.id == driver_id for driver in drivers)
    }
    source_plans = {
        "A": [{"lap": 2, "compound": "medium"}],
        **({"B": [{"lap": 3, "compound": "medium"}]} if not one_member else {}),
        "C": [{"lap": 2, "compound": "hard"}],
    }
    result = MonteCarloRunner(
        drivers,
        {team_id: Car(team_id=team_id, team_name=team_id) for team_id in ("T", "U")},
        Track(id="t", name="Saved", country="T", total_laps=5, base_lap_time=90),
        Weather(change_probability=0), seed=71, race_engine=engine,
        starting_tires={driver_id: "medium" for driver_id in inventory},
        tire_inventory=inventory,
        tire_warmup={"medium": 1.0, "hard": 0.5},
        pit_plans=source_plans,
    ).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(result)
    return path, inventory, source_plans


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_constructor_variants_snapshot_every_member_preserve_rivals_and_pair(
    tmp_path, engine,
):
    path, inventory, source_plans = _saved(tmp_path, engine=engine)
    saved_bytes = path.read_bytes()
    source_drivers = json.loads(saved_bytes)["simulation_inputs"]["drivers"]
    requested = {
        "automatic": None,
        "mixed": {"A": None, "B": []},
        "both-changed": {
            "A": [{"lap": 2, "compound": "hard"}],
            "B": [{"lap": 3, "compound": "hard"}],
        },
    }
    before_requested = deepcopy(requested)

    results = compare_saved_constructor_pit_plans(
        path, "T", requested, num_simulations=2,
    )

    assert list(results) == list(requested)
    assert results["automatic"].input_snapshot["pit_plans"] == {"C": source_plans["C"]}
    assert results["mixed"].input_snapshot["pit_plans"] == {
        "B": [], "C": source_plans["C"],
    }
    assert results["both-changed"].input_snapshot["pit_plans"] == {
        "A": [{"lap": 2, "compound": "hard"}],
        "B": [{"lap": 3, "compound": "hard"}],
        "C": source_plans["C"],
    }
    for result in results.values():
        assert result.input_snapshot["tire_inventory"] == inventory
        assert result.input_snapshot["tire_warmup"] == {"medium": 1.0, "hard": 0.5}
        assert result.input_snapshot["drivers"] == source_drivers
    assert requested == before_requested
    assert path.read_bytes() == saved_bytes

    paired = paired_comparison_statistics(results, "automatic")
    change = paired["variants"]["both-changed"]
    assert change["status"] == "paired"
    assert change["constructor_statistics"]["T"]["paired_races"] == 2
    assert change["constructor_statistics"]["T"]["driver_ids"] == ["A", "B"]


def test_constructor_parallel_and_serial_trials_agree(tmp_path):
    path, _, _ = _saved(tmp_path)
    plans = {"automatic": None, "staggered": {
        "A": [{"lap": 2, "compound": "hard"}],
        "B": [{"lap": 3, "compound": "hard"}],
    }}
    serial = compare_saved_constructor_pit_plans(
        path, "T", plans, num_simulations=2, parallel=False,
    )
    parallel = compare_saved_constructor_pit_plans(
        path, "T", plans, num_simulations=2, parallel=True, max_workers=2,
    )
    for label in plans:
        assert serial[label].race_results == parallel[label].race_results
        assert serial[label].qualifying_results == parallel[label].qualifying_results
        assert serial[label].input_snapshot == parallel[label].input_snapshot


def test_single_member_constructor_is_supported(tmp_path):
    path, _, _ = _saved(tmp_path, one_member=True)
    results = compare_saved_constructor_pit_plans(
        path, "T", {"automatic": None, "member-automatic": {"A": None}},
        num_simulations=1,
    )
    assert results["automatic"].input_snapshot["pit_plans"] == {
        "C": [{"lap": 2, "compound": "hard"}],
    }
    assert results["member-automatic"].input_snapshot["pit_plans"] == (
        results["automatic"].input_snapshot["pit_plans"]
    )


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"": None}, "plan labels must be nonempty strings"),
        ({"x" * 81: None}, "plan labels must be nonempty strings"),
        ({f"variant-{index}": None for index in range(11)}, "at most 10 variants"),
        ({"partial": {"A": []}}, "exactly constructor members"),
        ({"extra": {"A": [], "B": [], "C": []}}, "exactly constructor members"),
        ({"empty": {}}, "name every constructor member"),
        ({"wrong-type": [None, None]}, "member mapping or null"),
        ({"wrong-member-value": {"A": [], "B": "automatic"}}, "must be a list or null"),
        ({"valid": None, "late-inventory-invalid": {
            "A": [{"lap": 2, "compound": "hard"}],
            "B": [{"lap": 3, "compound": "soft"}],
        }}, "absent from B's tire inventory"),
        ({"valid": None, "late-invalid": {"A": [], "B": [{"lap": 6, "compound": "hard"}]}},
         "must not exceed total_laps"),
    ],
)
def test_invalid_alternatives_fail_before_any_trial(tmp_path, monkeypatch, bad, message):
    path, _, _ = _saved(tmp_path)
    calls = []

    def unexpected_run(self, *args, **kwargs):
        calls.append(self)
        raise AssertionError("invalid alternatives must fail before trials")

    monkeypatch.setattr(MonteCarloRunner, "run", unexpected_run)
    with pytest.raises(ValueError, match=message):
        compare_saved_constructor_pit_plans(path, "T", bad, num_simulations=1)
    assert calls == []


def test_unknown_constructor_missing_car_and_no_members_are_distinct(tmp_path):
    path, _, _ = _saved(tmp_path)
    original = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Unknown constructor ID"):
        compare_saved_constructor_pit_plans(path, "UNKNOWN", {"auto": None}, num_simulations=1)

    payload = deepcopy(original)
    payload["simulation_inputs"]["cars"].pop("T")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="No saved car available"):
        compare_saved_constructor_pit_plans(path, "T", {"auto": None}, num_simulations=1)

    payload = deepcopy(original)
    for driver in payload["simulation_inputs"]["drivers"]:
        if driver["team_id"] == "T":
            driver["team_id"] = "U"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="No runnable saved drivers"):
        compare_saved_constructor_pit_plans(path, "T", {"auto": None}, num_simulations=1)
