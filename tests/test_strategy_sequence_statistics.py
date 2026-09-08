"""Recorded fitting frequencies retain order, retirements and missing data."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.models.tire import TireCompound
from f1sim.simulation.race import DriverStatus


def row(sequence, driver="a", status=DriverStatus.FINISHED):
    return SimpleNamespace(driver_id=driver, strategy=sequence, status=status)


def summarize(races):
    results = SimulationResults(999, "Test", {"unobserved": None}, races, [])
    before = deepcopy(results)
    summary = results.get_strategy_statistics()
    assert results == before
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    return summary


def test_shares_use_recorded_rows_including_retirements_not_requested_trials():
    summary = summarize([
        [row(["SOFT", "MEDIUM"]), row(["HARD"], "b")],
        [row(["SOFT", "MEDIUM"], status=DriverStatus.DNF)],
        [row(["SOFT"], status=DriverStatus.DNF)],
        [row(None)],
        [],
    ])
    assert summary["a"] == {
        "races": 4, "races_with_recorded_strategy": 3, "missing_strategy_races": 1,
        "strategies": [
            {"compounds": ["SOFT", "MEDIUM"], "races": 2, "share": 2 / 3,
             "finished_races": 1, "dnf_races": 1},
            {"compounds": ["SOFT"], "races": 1, "share": 1 / 3,
             "finished_races": 0, "dnf_races": 1},
        ],
    }
    assert summary["b"]["races"] == 1
    assert summary["b"]["strategies"][0]["share"] == 1.0
    assert "unobserved" not in summary


def test_sequences_preserve_repeated_fittings_order_and_deterministic_ties():
    rows = [row(["SOFT", "MEDIUM"], "z"), row(["MEDIUM", "SOFT"], "z"),
            row(["SOFT", "SOFT"], "z"), row(["SOFT"], "a")]
    summary = summarize([rows])
    assert list(summary) == ["a", "z"]
    assert [variant["compounds"] for variant in summary["z"]["strategies"]] == [
        ["MEDIUM", "SOFT"], ["SOFT", "MEDIUM"], ["SOFT", "SOFT"],
    ]
    assert summarize([list(reversed(rows))]) == summary


@pytest.mark.parametrize("sequence", [None, [], (), "SOFT", {}, {"SOFT"},
                                       [""], ["SOFT", None], [1], [True], [["SOFT"]]])
def test_malformed_sequences_are_missing(sequence):
    assert summarize([[row(sequence)]])["a"] == {
        "races": 1, "races_with_recorded_strategy": 0,
        "missing_strategy_races": 1, "strategies": [],
    }


def test_missing_legacy_attribute_and_no_observed_rows():
    legacy = SimpleNamespace(driver_id="legacy", status=DriverStatus.DNF)
    assert summarize([[legacy]])["legacy"]["missing_strategy_races"] == 1
    assert summarize([[], []]) == {}


def test_enum_tuple_and_custom_labels_are_native_strings_and_group_together():
    summary = summarize([[row((TireCompound.SOFT, "custom experimental")),
                          row([TireCompound.SOFT.value, "custom experimental"],
                              status="unrecognized")]])
    variant, = summary["a"]["strategies"]
    assert variant == {
        "compounds": [TireCompound.SOFT.value, "custom experimental"],
        "races": 2, "share": 1.0, "finished_races": 1, "dnf_races": 0,
    }
    assert all(type(compound) is str for compound in variant["compounds"])


def test_returned_sequences_cannot_mutate_recorded_inputs():
    sequence = ["SOFT", "MEDIUM"]
    results = SimulationResults(1, "Test", {}, [[row(sequence)]], [])
    summary = results.get_strategy_statistics()
    summary["a"]["strategies"][0]["compounds"].append("HARD")
    assert sequence == ["SOFT", "MEDIUM"]
    assert results.get_strategy_statistics()["a"]["strategies"][0]["compounds"] == sequence
