"""Public validation and shorthand parsing for explicit pit plans."""

import pytest

from f1sim.simulation.pit_plans import parse_pit_plan_spec, validate_pit_plans


def test_validate_pit_plans_copies_and_preserves_explicit_empty_driver():
    source = {
        "A": [{"lap": 18, "compound": "medium"}],
        "B": [],
    }
    normalized = validate_pit_plans(source, driver_ids=["A", "B"], total_laps=50)
    assert normalized == source
    assert normalized is not source
    assert normalized["A"][0] is not source["A"][0]
    normalized["A"][0]["lap"] = 20
    assert source["A"][0]["lap"] == 18


def test_parser_supports_groups_and_none():
    assert parse_pit_plan_spec("DRIVER=18:medium,36:hard;NOR=24:hard") == {
        "DRIVER": [
            {"lap": 18, "compound": "medium"},
            {"lap": 36, "compound": "hard"},
        ],
        "NOR": [{"lap": 24, "compound": "hard"}],
    }
    assert parse_pit_plan_spec("A=none\nB=24:soft") == {
        "A": [], "B": [{"lap": 24, "compound": "soft"}],
    }
    assert parse_pit_plan_spec("  ") == {}


@pytest.mark.parametrize("value", [
    {"A": [{"lap": 2.0, "compound": "soft"}]},
    {"A": [{"lap": True, "compound": "soft"}]},
    {"A": [{"lap": 1, "compound": "soft"}]},
    {"A": [{"lap": 4, "compound": "soft", "extra": 1}]},
    {"A": [{"lap": 4, "compound": "soft"}, {"lap": 4, "compound": "hard"}]},
])
def test_validation_rejects_ambiguous_or_malformed_instructions(value):
    with pytest.raises(ValueError):
        validate_pit_plans(value, driver_ids=["A"], total_laps=10)


def test_validation_rejects_unknown_driver_and_out_of_pool_compound():
    with pytest.raises(ValueError, match="Unknown"):
        validate_pit_plans({"X": []}, driver_ids=["A"])
    with pytest.raises(ValueError, match="absent"):
        validate_pit_plans(
            {"A": [{"lap": 4, "compound": "hard"}]},
            driver_ids=["A"],
            tire_inventory={"A": [{"id": "M", "compound": "medium", "age": 0}]},
        )


def test_parser_rejects_duplicate_drivers_and_tokens():
    with pytest.raises(ValueError):
        parse_pit_plan_spec("A=4:soft;A=8:hard")
    with pytest.raises(ValueError):
        parse_pit_plan_spec("A=4:soft,,8:hard")
    with pytest.raises(ValueError):
        parse_pit_plan_spec("A=4.0:soft")
