"""Offline custom-plan variants preserve source inputs and paired eligibility."""

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.strategy_comparison import compare_saved_pit_plans
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter


def _saved(tmp_path):
    result = MonteCarloRunner(
        [Driver(id="A", name="A", team_id="T")],
        {"T": Car(team_id="T", team_name="T")},
        Track(id="t", name="Saved", country="T", total_laps=4, base_lap_time=90),
        Weather(change_probability=0), seed=71, starting_tires={"A": "medium"},
    ).run(1, parallel=False)
    return Exporter(tmp_path).export_statistics_json(result)


def _saved_with_other_driver_plan(tmp_path):
    drivers = [
        Driver(id="A", name="A", team_id="T"),
        Driver(id="B", name="B", team_id="U"),
    ]
    result = MonteCarloRunner(
        drivers,
        {
            "T": Car(team_id="T", team_name="T"),
            "U": Car(team_id="U", team_name="U"),
        },
        Track(id="t", name="Saved", country="T", total_laps=4, base_lap_time=90),
        Weather(change_probability=0), seed=71,
        starting_tires={"A": "medium", "B": "medium"},
        pit_plans={"B": [{"lap": 2, "compound": "hard"}]},
    ).run(1, parallel=False)
    return Exporter(tmp_path).export_statistics_json(result)


def test_compare_plans_keeps_none_and_empty_distinct_and_pairs(tmp_path):
    path = _saved(tmp_path)
    before = path.read_bytes()
    results = compare_saved_pit_plans(
        path,
        "A",
        {
            "automatic": None,
            "planned": [{"lap": 2, "compound": "hard"}],
            "no-elective": [],
        },
        num_simulations=2,
    )
    assert list(results) == ["automatic", "planned", "no-elective"]
    assert results["automatic"].input_snapshot.get("pit_plans") is None
    assert results["planned"].input_snapshot["pit_plans"] == {
        "A": [{"lap": 2, "compound": "hard"}],
    }
    assert results["no-elective"].input_snapshot["pit_plans"] == {"A": []}
    paired = paired_comparison_statistics(results, "automatic")
    assert paired["variants"]["planned"]["status"] == "paired"
    assert paired["variants"]["no-elective"]["status"] == "paired"
    assert path.read_bytes() == before


def test_all_plan_variants_validate_before_trials(tmp_path, monkeypatch):
    path = _saved(tmp_path)
    called = False

    def unexpected_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid alternatives must fail before trials")

    monkeypatch.setattr(MonteCarloRunner, "run", unexpected_run)
    with pytest.raises(ValueError, match="lap|compound"):
        compare_saved_pit_plans(
            path,
            "A",
            {
                "good": [{"lap": 2, "compound": "hard"}],
                "bad": [{"lap": 1, "compound": "soft"}],
            },
            num_simulations=1,
        )
    assert called is False


def test_plan_variants_retain_other_driver_plans(tmp_path):
    path = _saved_with_other_driver_plan(tmp_path)
    results = compare_saved_pit_plans(
        path, "A", {"automatic": None, "empty": []}, num_simulations=1,
    )
    expected = {"B": [{"lap": 2, "compound": "hard"}]}
    assert results["automatic"].input_snapshot["pit_plans"] == expected
    assert results["empty"].input_snapshot["pit_plans"] == {"A": [], **expected}


@pytest.mark.parametrize("bad", [None, {}, {"A": "bad"}])
def test_plan_comparison_rejects_invalid_variant_mapping(tmp_path, bad):
    path = _saved(tmp_path)
    with pytest.raises(ValueError):
        compare_saved_pit_plans(path, "A", bad, num_simulations=1)
