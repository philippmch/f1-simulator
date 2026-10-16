"""Rule validation must not silently change historical replay semantics."""

from collections import UserDict

import pytest

from f1sim.simulation.tire_rules import (
    RACE_TIRE_RULES_POLICY,
    merge_race_tire_rules,
    validate_race_tire_rules,
    validate_race_tire_rules_snapshot,
)


@pytest.mark.parametrize("value", [None, {}, {"minimum_sets": 1}])
def test_absent_or_default_rules_disable_overlay(value):
    assert validate_race_tire_rules(value) == {}


def test_rules_are_copied_in_a_canonical_order_without_default_keys():
    source = UserDict({"minimum_sets": 1, "max_set_laps": 25})
    result = validate_race_tire_rules(source)
    assert result == {"max_set_laps": 25}
    result["max_set_laps"] = 10
    assert source["max_set_laps"] == 25
    assert list(validate_race_tire_rules({"minimum_sets": 3, "max_set_laps": 25})) == [
        "max_set_laps", "minimum_sets",
    ]


@pytest.mark.parametrize("values,expected", [
    ((), {}),
    ((None, {}, {"minimum_sets": 1}), {}),
    (({"max_set_laps": 25}, {"max_set_laps": 40}), {"max_set_laps": 25}),
    (({"max_set_laps": 25}, {"max_set_laps": 18}), {"max_set_laps": 18}),
    (({"minimum_sets": 3}, {"minimum_sets": 1}), {"minimum_sets": 3}),
    (({"minimum_sets": 3}, {"minimum_sets": 4}), {"minimum_sets": 4}),
    (({"max_set_laps": 25}, {"minimum_sets": 3}),
     {"max_set_laps": 25, "minimum_sets": 3}),
    (({"max_set_laps": 25, "minimum_sets": 3}, {}, None,
      {"max_set_laps": 40, "minimum_sets": 2}),
     {"max_set_laps": 25, "minimum_sets": 3}),
])
def test_merged_constraints_keep_the_stricter_independent_requirements(values, expected):
    assert merge_race_tire_rules(*values) == expected
    assert merge_race_tire_rules(*reversed(values)) == expected


def test_merge_is_detached_and_rejects_invalid_additional_constraints():
    source = {"max_set_laps": 25, "minimum_sets": 3}
    merged = merge_race_tire_rules(source, {})
    merged["max_set_laps"] = 18
    assert source == {"max_set_laps": 25, "minimum_sets": 3}
    with pytest.raises(ValueError, match="positive integer"):
        merge_race_tire_rules(source, {"minimum_sets": False})


@pytest.mark.parametrize("value", [[], "25", 25, False, {"unknown": 3}, {1: 3}])
def test_rules_reject_nonobjects_and_unknown_fields(value):
    with pytest.raises(ValueError, match="race_tire_rules"):
        validate_race_tire_rules(value)


@pytest.mark.parametrize("field", ["max_set_laps", "minimum_sets"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 25.0, "25", None, float("nan")])
def test_rules_require_strict_positive_integers(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        validate_race_tire_rules({field: value})


def test_snapshot_restores_exact_rule_policy_and_copies_values():
    snapshot = {"schema_version": 16, "race_tire_rules": {"minimum_sets": 3},
                "race_tire_rules_policy": RACE_TIRE_RULES_POLICY}
    rules = validate_race_tire_rules_snapshot(snapshot)
    assert rules == {"minimum_sets": 3}
    rules["minimum_sets"] = 4
    assert snapshot["race_tire_rules"] == {"minimum_sets": 3}


@pytest.mark.parametrize("version", range(1, 16))
def test_older_snapshots_stay_disabled_and_reject_new_rule_fields(version):
    assert validate_race_tire_rules_snapshot({"schema_version": version}) == {}
    for fields in ({"race_tire_rules": {"max_set_laps": 25}},
                   {"race_tire_rules": {}},
                   {"race_tire_rules_policy": RACE_TIRE_RULES_POLICY}):
        with pytest.raises(ValueError, match="require schema 16"):
            validate_race_tire_rules_snapshot({"schema_version": version, **fields})


@pytest.mark.parametrize("fields", [
    {},
    {"race_tire_rules": {"minimum_sets": 3}},
    {"race_tire_rules_policy": RACE_TIRE_RULES_POLICY},
    {"race_tire_rules": {"minimum_sets": 3}, "race_tire_rules_policy": "future_policy"},
    {"race_tire_rules": {}, "race_tire_rules_policy": RACE_TIRE_RULES_POLICY},
    {"race_tire_rules": None, "race_tire_rules_policy": RACE_TIRE_RULES_POLICY},
    {"race_tire_rules": {"minimum_sets": 1}, "race_tire_rules_policy": RACE_TIRE_RULES_POLICY},
])
def test_schema16_cannot_omit_policy_or_disable_the_enabled_feature(fields):
    with pytest.raises(ValueError, match="Schema 16"):
        validate_race_tire_rules_snapshot({"schema_version": 16, **fields})


def test_unknown_snapshot_version_cannot_reinterpret_rule_policy():
    with pytest.raises(ValueError, match="require schema 16"):
        validate_race_tire_rules_snapshot({"schema_version": 17,
                                           "race_tire_rules": {"max_set_laps": 25},
                                           "race_tire_rules_policy": RACE_TIRE_RULES_POLICY})


@pytest.mark.parametrize("value", [None, [], "snapshot"])
def test_snapshot_requires_an_object(value):
    with pytest.raises(ValueError, match="snapshot must be an object"):
        validate_race_tire_rules_snapshot(value)
