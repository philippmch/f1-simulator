"""Year, venue, source timing and ownership contracts for special race tyre rules."""

from datetime import datetime, timedelta, timezone

import pytest

from f1sim.data.event_tire_rules import get_event_tire_rules


@pytest.mark.parametrize(
    "event",
    [
        {"circuit_id": "monaco"},
        {"circuit_name": "Circuit de Monaco"},
        {"circuit_name": "  CIRCUIT-DE-MONACO  "},
        {"circuit_id": "monte_carlo"},
    ],
)
def test_2025_monaco_requires_three_sets(event):
    rules, sources = get_event_tire_rules(2025, event)
    assert rules == {"minimum_sets": 3}
    source = sources["minimum_sets"]
    assert source["published_at"] == "2025-02-26"
    assert source["source_url"].endswith("issue_4_-_2025-02-26.pdf")
    assert source["source_section"] == "Article 30.5(m), page 38"
    assert any("Wet tyre use does not remove" in item for item in source["limitations"])


@pytest.mark.parametrize("year", [2023, 2024, 2026, 2027])
@pytest.mark.parametrize("circuit", ["monaco", "losail"])
def test_special_rules_do_not_carry_to_other_seasons(year, circuit):
    assert get_event_tire_rules(year, {"circuit_id": circuit}) == ({}, {})


@pytest.mark.parametrize(
    "event",
    [
        {"circuit_id": "losail"},
        {"circuit_id": "lusail"},
        {"circuit_name": "Lusail International Circuit"},
        {"circuit_name": "Losail International Circuit"},
    ],
)
def test_2025_qatar_cap_preserves_weekend_usage_limitations(event):
    rules, sources = get_event_tire_rules(2025, event)
    assert rules == {"max_set_laps": 25}
    source = sources["max_set_laps"]
    assert source["source_url"] == (
        "https://press.pirelli.com/a-maximum-of-25-laps-per-tyre-set-in-qatar/"
    )
    assert source["published_at"] == "2025-11-17T09:36:00Z"
    assert "cumulative" in source["scope"]
    assert any("necessary upper bound" in item for item in source["limitations"])
    assert any("remaining_laps" in item for item in source["limitations"])
    assert any("not reconstructed" in item for item in source["limitations"])
    assert any("Started but incomplete" in item for item in source["limitations"])


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"circuit_id": "silverstone", "race": "Monaco Grand Prix"},
        {"country": "Qatar"},
        {"circuit_name": "Fake Circuit de Monaco Annex"},
        {"circuit_id": "monaco", "circuit_name": "Lusail International Circuit"},
        {"circuit_id": "monaco", "date": "2026-06-01"},
        {"circuit_id": "monaco", "season": 2026},
    ],
)
def test_unknown_ambiguous_and_wrong_year_events_have_no_rule(event):
    assert get_event_tire_rules(2025, event) == ({}, {})


def test_date_only_monaco_source_requires_end_of_published_day():
    event = {"circuit_id": "monaco", "date": "2025-05-25"}
    assert get_event_tire_rules(2025, event, cutoff="2025-02-26T23:59:59Z") == ({}, {})
    rules, source = get_event_tire_rules(2025, event, cutoff="2025-02-27T00:00:00Z")
    assert rules == {"minimum_sets": 3}
    assert source["minimum_sets"]["publication_precision"] == "day"


def test_qatar_source_timestamp_and_timezone_cutoff_are_conservative():
    event = {"circuit_id": "losail", "date": "2025-11-30"}
    assert get_event_tire_rules(2025, event, cutoff="2025-11-17T09:36:59Z") == ({}, {})
    local = datetime(2025, 11, 17, 10, 37, tzinfo=timezone(timedelta(hours=1)))
    assert get_event_tire_rules(2025, event, cutoff=local)[0] == {"max_set_laps": 25}
    assert get_event_tire_rules(2025, event, cutoff="2025-11-29T18:00:00Z")[0] == {
        "max_set_laps": 25
    }


@pytest.mark.parametrize("cutoff", [datetime(2025, 11, 30), "2025-11-30", "invalid", 3])
def test_invalid_or_naive_cutoffs_are_rejected(cutoff):
    with pytest.raises(ValueError):
        get_event_tire_rules(2025, {"circuit_id": "losail"}, cutoff=cutoff)


@pytest.mark.parametrize("year", [True, 2025.0, "2025"])
def test_non_integer_years_are_rejected(year):
    with pytest.raises(ValueError, match="integer season"):
        get_event_tire_rules(year, {"circuit_id": "monaco"})


def test_returned_rules_and_nested_provenance_are_detached():
    event = {"circuit_id": "monaco"}
    rules, sources = get_event_tire_rules(2025, event)
    rules["minimum_sets"] = 99
    sources["minimum_sets"]["limitations"].clear()
    sources["minimum_sets"]["source_url"] = "wrong"
    fresh_rules, fresh_sources = get_event_tire_rules(2025, event)
    assert fresh_rules == {"minimum_sets": 3}
    assert fresh_sources["minimum_sets"]["limitations"]
    assert fresh_sources["minimum_sets"]["source_url"] != "wrong"
    assert event == {"circuit_id": "monaco"}
