"""Explicit tyre pools preserve physical set identity, wear and damage."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from f1sim.simulation.tire_inventory import (
    TireInventory,
    parse_tire_inventory_spec,
    validate_tire_inventory,
)


@pytest.mark.parametrize("value", [[], {"": []}, {1: []}, {"A": []},
                                   {"A": [{}] * 21}, {"A": {}}, {"A": [None]},
                                   {"A": [{"compound": "soft", "extra": 1}]},
                                   {"A": [{"compound": "SOFT"}]}, {"A": [{}]},
                                   {"A": [{"compound": 1}]},
                                   {"A": [{"compound": "soft", "id": " "}]},
                                   {"A": [{"compound": "soft", "id": "set-2"},
                                          {"compound": "hard"}]}])
def test_malformed_pools_rejected(value):
    with pytest.raises(ValueError, match="tire_inventory"):
        validate_tire_inventory(value)


@pytest.mark.parametrize("age", [True, -1, 1001, 2.0, "2", None, float("nan")])
def test_strict_opening_age(age):
    with pytest.raises(ValueError, match="tire_inventory"):
        validate_tire_inventory({"A": [{"compound": "soft", "age": age}]})


def test_exact_opening_match_roster_and_copy():
    original = {"A": [{"compound": "soft", "age": 5}, {"compound": "hard"}]}
    result = validate_tire_inventory(original, {"A": "soft"}, {"A": 5}, ["A", "B"])
    assert result["A"][0] == {"id": "set-1", "compound": "soft", "age": 5}
    result["A"][0]["age"] = 9
    assert original["A"][0]["age"] == 5
    for openings, ages in [({"A": "soft"}, None), ({"A": "hard"}, {"A": 5})]:
        with pytest.raises(ValueError, match="exact opening"):
            validate_tire_inventory(original, openings, ages)
    with pytest.raises(ValueError, match="Unknown"):
        validate_tire_inventory(original, driver_ids=["B"])
    assert validate_tire_inventory(original, {"B": "wet"})
    assert validate_tire_inventory(None) == validate_tire_inventory({}) == {}


@pytest.mark.parametrize("text", ["A=soft;", ";A=soft", "A=soft;;B=hard",
                                  "A=soft,", "A=", "=soft", "A=soft;A=hard",
                                  "A=soft@-1", "A=soft@1.5", "A=soft@1001"])
def test_parser_rejects_empty_or_duplicate_entries(text):
    with pytest.raises(ValueError):
        parse_tire_inventory_spec(text)


def test_parser_accepts_multiple_drivers_and_distinct_duplicate_sets():
    result = parse_tire_inventory_spec(" A=soft@5,soft,hard;B=wet\nC=medium ")
    assert [item["age"] for item in result["A"]] == [5, 0, 0]
    assert [item["id"] for item in result["A"]] == ["set-1", "set-2", "set-3"]
    assert parse_tire_inventory_spec(" ") == {}


def test_reuse_damage_snapshot_and_no_aliasing():
    records = [{"id": "used", "compound": "soft", "age": 5}, {"compound": "hard"}]
    inventory = TireInventory.from_sets(records)
    records[0]["age"] = 100
    assert inventory.fit("used").age == 5
    with pytest.raises(FrozenInstanceError):
        inventory.sets["used"].age = 6
    assert inventory.fit("used", current_age=7).age == 7
    assert inventory.fit("set-2", current_age=9).age == 0
    assert inventory.replacements()[0].age == 9
    assert inventory.fit("used", current_age=0).age == 9  # Unrun removed set is reusable.
    before = deepcopy(inventory.__dict__)
    snapshot = inventory.snapshot(current_age=11)
    assert snapshot[0] == {"id": "used", "compound": "soft", "age": 11,
                           "current": True, "available": False, "unavailable": False}
    snapshot[0]["age"] = 999
    assert inventory.__dict__ == before
    inventory.mark_current_unavailable(11)
    inventory.fit("set-2", current_age=11)
    assert inventory.replacements() == ()
    assert inventory.snapshot()[0]["available"] is False
    assert inventory.snapshot()[0]["unavailable"] is True
    with pytest.raises(ValueError, match="unavailable"):
        inventory.fit("used")


@pytest.mark.parametrize("identifier,age", [("missing", 50), ([], 50),
                                          ("set-2", True), ("set-2", -1)])
def test_invalid_fit_is_atomic(identifier, age):
    inventory = TireInventory.from_sets([{"compound": "soft"}, {"compound": "hard"}])
    inventory.fit("set-1")
    before = deepcopy(inventory.__dict__)
    with pytest.raises(ValueError):
        inventory.fit(identifier, current_age=age)
    assert inventory.__dict__ == before


def test_reported_current_wear_cannot_reverse_existing_use():
    inventory = TireInventory.from_sets([{"compound": "soft", "age": 8},
                                         {"compound": "hard"}])
    inventory.fit("set-1")
    before = deepcopy(inventory.__dict__)
    for operation in (lambda: inventory.fit("set-2", 7),
                      lambda: inventory.mark_current_unavailable(7),
                      lambda: inventory.snapshot(7)):
        with pytest.raises(ValueError, match="cannot decrease"):
            operation()
        assert inventory.__dict__ == before


def test_runtime_wear_can_exceed_opening_input_limit():
    inventory = TireInventory.from_sets([{"compound": "soft", "age": 1000}])
    inventory.fit("set-1")
    assert inventory.fit("set-1", current_age=1001).age == 1001
    inventory.mark_current_unavailable(1002)
    before = deepcopy(inventory.__dict__)
    with pytest.raises(ValueError):
        inventory.fit("set-1", current_age=0)
    assert inventory.__dict__ == before
