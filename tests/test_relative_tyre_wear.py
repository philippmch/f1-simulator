"""Contract tests for offline relative tyre-wear evaluation."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from f1sim.analysis.relative_tyre_wear import (
    RelativeTyreWearInputError,
    evaluate_relative_tyre_wear,
)

_FEEDS = ("TimingData", "TimingAppData", "TrackStatus", "WeatherData", "SessionStatus")
_BY_COMPOUND = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.02}


def _clock(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.000"


def _report(
    *,
    event_key: int = 101,
    season: int = 2026,
    drivers_per_compound: int = 4,
    lap_count: int = 20,
    slopes: dict[str, float] | None = None,
    common_shift=None,
    stint_offsets=None,
    noise: bool = False,
    prior_wear: int | None = None,
) -> dict:
    slopes = _BY_COMPOUND if slopes is None else slopes
    compounds = [
        "SOFT",
        "MEDIUM",
        "HARD",
    ]
    laps = []
    for driver_index in range(drivers_per_compound * len(compounds)):
        compound = compounds[driver_index // drivers_per_compound]
        driver = driver_index + 1
        for lap_number in range(1, lap_count + 1):
            common = 7.0 * lap_number + 0.003 * lap_number**2
            shift = common_shift(lap_number) if common_shift else 0.0
            stint_offset = stint_offsets(driver, lap_number) if stint_offsets else 0.0
            perturbation = 0.17 * math.sin(driver * 1.73 + lap_number * 0.91) if noise else 0.0
            duration = (
                90.0
                + driver_index * 0.21
                + common
                + slopes[compound] * lap_number
                + shift
                + stint_offset
                + perturbation
            )
            start = _clock((lap_number - 1) * 92)
            end = _clock(lap_number * 92)
            laps.append(
                {
                    "driver_number": driver,
                    "lap_number": lap_number,
                    "duration_seconds": duration,
                    "reported_at": end,
                    "observed_start": start,
                    "observed_end": end,
                    "stint": 0,
                    "compound": compound,
                    "prior_wear": prior_wear,
                    "start_stint": 0,
                    "end_stint": 0,
                    "start_compound": compound,
                    "end_compound": compound,
                    "eligible": True,
                    "exclusions": [],
                }
            )
    summary = _summary(laps)
    return {
        "source": f"https://livetiming.formula1.com/static/{season}/2026-05-01_Test/2026-05-01_Race/",
        "source_feeds": list(_FEEDS),
        "decoded_feed_sha256": {feed: chr(97 + index) * 64 for index, feed in enumerate(_FEEDS)},
        "season": season,
        "meeting": f"Test {event_key}",
        "session_key": event_key,
        "retrieved_at": "2026-05-01T23:00:00+00:00",
        "evidence": {
            "normalizer_version": 1,
            "laps": laps,
            "summary": summary,
            "coverage": {
                "session_started_at": "00:00:00.000",
                "feeds": {
                    feed: {
                        "present": True,
                        "rows": 1,
                        "first": "00:00:00.000",
                        "last": "00:01:00.000",
                    }
                    for feed in _FEEDS
                },
            },
        },
    }


def _summary(laps: list[dict]) -> dict:
    reason_counts: dict[str, int] = {}
    for lap in laps:
        for reason in lap["exclusions"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "total_laps": len(laps),
        "eligible_laps": sum(lap["eligible"] for lap in laps),
        "duration_observations": sum(lap["duration_seconds"] is not None for lap in laps),
        "missing_duration": reason_counts.get("missing_duration", 0),
        "stale_duration": reason_counts.get("stale_duration", 0),
        "ambiguous_laps": sum(
            any("ambiguous" in reason or "duplicate" in reason for reason in lap["exclusions"])
            for lap in laps
        ),
        "duplicate_identical_observations": 0,
        "conflicting_duplicate_observations": 0,
        "missing_preceding_crossing": reason_counts.get("missing_preceding_crossing", 0),
        "unpaired_duration_observations": 0,
        "nonempty_unpaired_duration_observations": 0,
        "invalid_duration_observations": 0,
        "invalid_lap_count_observations": 0,
        "stale_duration_observations": 0,
        "exclusion_counts": reason_counts,
    }


def _synchronize_compounds_by_lap(report: dict) -> dict:
    for lap in report["evidence"]["laps"]:
        phase = (lap["lap_number"] - 1) // 6
        compound = ("HARD", "MEDIUM", "SOFT")[phase]
        lap["compound"] = compound
        lap["start_compound"] = compound
        lap["end_compound"] = compound
        lap["stint"] = phase
        lap["start_stint"] = phase
        lap["end_stint"] = phase
    report["evidence"]["summary"] = _summary(report["evidence"]["laps"])
    return report


def _retain_compounds(report: dict, compounds: set[str]) -> dict:
    for lap in report["evidence"]["laps"]:
        if lap["compound"] not in compounds:
            lap["eligible"] = False
            lap["exclusions"] = ["pit_affected"]
    report["evidence"]["summary"] = _summary(report["evidence"]["laps"])
    return report


def _manual_single_event_intervals(report: dict) -> dict[str, tuple[float, float]]:
    laps = report["evidence"]["laps"]
    drivers = sorted({lap["driver_number"] for lap in laps})
    lap_numbers = sorted({lap["lap_number"] for lap in laps})
    stints = sorted({(lap["driver_number"], lap["stint"]) for lap in laps})
    stint_index = {key: index for index, key in enumerate(stints)}
    lap_index = {value: index for index, value in enumerate(lap_numbers)}
    y = np.asarray([lap["duration_seconds"] for lap in laps])
    centered = np.asarray([lap["lap_number"] for lap in laps], dtype=float)
    centered -= centered.mean()
    x = np.column_stack(
        (
            np.asarray([lap["compound"] == "SOFT" for lap in laps]) * centered,
            np.asarray([lap["compound"] == "MEDIUM" for lap in laps]) * centered,
        )
    )
    nuisance = np.zeros((len(laps), len(stints) + len(lap_numbers)))
    for row, lap in enumerate(laps):
        nuisance[row, stint_index[(lap["driver_number"], lap["stint"])]] = 1
        nuisance[row, len(stints) + lap_index[lap["lap_number"]]] = 1
    nuisance_x = nuisance @ np.linalg.lstsq(nuisance, x, rcond=None)[0]
    residual_x = x - nuisance_x
    full = np.column_stack((nuisance, x))
    full_beta, _, full_rank, _ = np.linalg.lstsq(full, y, rcond=None)
    residual_y = y - full @ full_beta
    bread = np.linalg.inv(residual_x.T @ residual_x)
    score = np.zeros((len(drivers), 2))
    driver_index = {driver: index for index, driver in enumerate(drivers)}
    for row, lap in enumerate(laps):
        score[driver_index[lap["driver_number"]]] += residual_x[row] * residual_y[row]
    covariance = (
        len(drivers)
        / (len(drivers) - 1)
        * (len(laps) - 1)
        / (len(laps) - full_rank)
        * bread
        @ (score.T @ score)
        @ bread
    )
    estimates = (full_beta[-2], full_beta[-1], full_beta[-2] - full_beta[-1])
    contrasts = (np.array([1, 0]), np.array([0, 1]), np.array([1, -1]))
    names = ("soft_minus_hard", "medium_minus_hard", "soft_minus_medium")
    return {
        name: (
            estimates[index] - 1.96 * math.sqrt(contrasts[index] @ covariance @ contrasts[index]),
            estimates[index] + 1.96 * math.sqrt(contrasts[index] @ covariance @ contrasts[index]),
        )
        for index, name in enumerate(names)
    }


def _manual_partial_soft_medium(report: dict) -> tuple[float, float, float]:
    """Independent full-design oracle for the estimable SOFT-MEDIUM contrast."""
    laps = [lap for lap in report["evidence"]["laps"] if lap["eligible"]]
    drivers = sorted({lap["driver_number"] for lap in laps})
    lap_numbers = sorted({lap["lap_number"] for lap in laps})
    stints = sorted({(lap["driver_number"], lap["stint"]) for lap in laps})
    stint_index = {key: index for index, key in enumerate(stints)}
    lap_index = {value: index for index, value in enumerate(lap_numbers)}
    y = np.asarray([lap["duration_seconds"] for lap in laps])
    centered_lap = np.asarray([lap["lap_number"] for lap in laps], dtype=float)
    centered_lap -= centered_lap.mean()
    nuisance = np.zeros((len(laps), len(stints) + len(lap_numbers)))
    contrast_regressor = np.zeros(len(laps))
    for row, lap in enumerate(laps):
        nuisance[row, stint_index[(lap["driver_number"], lap["stint"])]] = 1
        nuisance[row, len(stints) + lap_index[lap["lap_number"]]] = 1
        sign = 1.0 if lap["compound"] == "SOFT" else -1.0
        contrast_regressor[row] = 0.5 * sign * centered_lap[row]
    full = np.column_stack((nuisance, contrast_regressor))
    beta, _, full_rank, _ = np.linalg.lstsq(full, y, rcond=None)
    residual = y - full @ beta
    projected_contrast = contrast_regressor - nuisance @ np.linalg.lstsq(
        nuisance, contrast_regressor, rcond=None
    )[0]
    bread = 1 / float(projected_contrast @ projected_contrast)
    driver_index = {driver: index for index, driver in enumerate(drivers)}
    scores = np.zeros(len(drivers))
    for row, lap in enumerate(laps):
        scores[driver_index[lap["driver_number"]]] += projected_contrast[row] * residual[row]
    df = len(y) - full_rank
    correction = len(drivers) / (len(drivers) - 1) * (len(y) - 1) / df
    variance = correction * bread**2 * float(scores @ scores)
    return float(beta[-1]), math.sqrt(variance), df


def test_recovers_relative_slopes_and_serializes_finite_json() -> None:
    result = evaluate_relative_tyre_wear(_report())
    event = result["events"][0]
    assert event["status"] == "available"
    assert event["estimates_seconds_per_lap"]["soft_minus_hard"] == pytest.approx(0.06)
    assert event["estimates_seconds_per_lap"]["medium_minus_hard"] == pytest.approx(0.03)
    assert event["estimates_seconds_per_lap"]["soft_minus_medium"] == pytest.approx(0.03)
    assert event["intervals_seconds_per_lap"]["soft_minus_hard"] is None
    assert event["uncertainty"]["reason"] == "fewer_than_five_driver_event_clusters_for_a_compound"
    assert result["coverage"]["driver_event_clusters_by_compound"] == {
        "SOFT": 4,
        "MEDIUM": 4,
        "HARD": 4,
    }
    assert event["identifiable_contrasts"]["soft_minus_medium"][
        "estimate_seconds_per_lap"
    ] == pytest.approx(0.03)
    assert event["identifiable_contrasts"]["soft_minus_medium"][
        "standard_error_seconds_per_lap"
    ] is None
    assert event["identifiable_contrasts"]["soft_minus_medium"][
        "uncertainty_reason"
    ] == "fewer_than_five_driver_event_clusters_for_a_compound"
    json.dumps(result, allow_nan=False)


def test_full_rank_legacy_fields_keep_their_existing_contract() -> None:
    event = evaluate_relative_tyre_wear(_report())["events"][0]
    assert event["status"] == "available"
    assert event["rank"] == {
        "nuisance": 31,
        "contrast": 2,
        "full": 33,
        "residual_degrees_of_freedom": 207,
    }
    assert event["estimates_seconds_per_lap"] == pytest.approx(
        {
            "soft_minus_hard": 0.06,
            "medium_minus_hard": 0.03,
            "soft_minus_medium": 0.03,
        }
    )
    assert event["intervals_seconds_per_lap"] == {
        "soft_minus_hard": None,
        "medium_minus_hard": None,
        "soft_minus_medium": None,
    }
    assert event["uncertainty"]["reason"] == (
        "fewer_than_five_driver_event_clusters_for_a_compound"
    )


@pytest.mark.parametrize(
    ("compounds", "expected_name", "expected_estimate", "unidentified_name"),
    [
        ({"SOFT", "MEDIUM"}, "soft_minus_medium", 0.03, "soft_minus_hard"),
        ({"SOFT", "HARD"}, "soft_minus_hard", 0.06, "medium_minus_hard"),
        ({"MEDIUM", "HARD"}, "medium_minus_hard", 0.03, "soft_minus_hard"),
    ],
)
def test_partial_compound_sets_report_only_estimable_contrasts(
    compounds, expected_name, expected_estimate, unidentified_name
) -> None:
    event = evaluate_relative_tyre_wear(
        _retain_compounds(_report(drivers_per_compound=6), compounds)
    )["events"][0]
    assert event["status"] == "unavailable"
    assert event["reason"] == "minimum_compound_coverage"
    diagnostics = event["identifiable_contrasts"]
    assert diagnostics[expected_name]["status"] == "available"
    assert diagnostics[expected_name]["reason"] is None
    assert diagnostics[expected_name]["estimate_seconds_per_lap"] == pytest.approx(
        expected_estimate
    )
    assert diagnostics[expected_name]["rank"]["contrast"] == 1
    assert diagnostics[unidentified_name]["status"] == "unavailable"
    assert diagnostics[unidentified_name]["reason"] == "contrast_not_estimable"
    assert diagnostics[unidentified_name]["estimate_seconds_per_lap"] is None


def test_partial_soft_medium_cr1_matches_independent_reparameterized_full_design() -> None:
    report = _retain_compounds(_report(drivers_per_compound=6, noise=True), {"SOFT", "MEDIUM"})
    diagnostic = evaluate_relative_tyre_wear(report)["events"][0]["identifiable_contrasts"][
        "soft_minus_medium"
    ]
    expected_estimate, expected_se, expected_df = _manual_partial_soft_medium(report)
    assert diagnostic["status"] == "available"
    assert diagnostic["rank"]["contrast"] == 1
    assert diagnostic["rank"]["residual_degrees_of_freedom"] == expected_df
    assert diagnostic["estimate_seconds_per_lap"] == pytest.approx(expected_estimate, abs=1e-9)
    assert diagnostic["standard_error_seconds_per_lap"] == pytest.approx(expected_se, abs=1e-9)
    assert diagnostic["interval_seconds_per_lap"]["lower"] == pytest.approx(
        expected_estimate - 1.96 * expected_se, abs=1e-9
    )
    assert diagnostic["interval_seconds_per_lap"]["upper"] == pytest.approx(
        expected_estimate + 1.96 * expected_se, abs=1e-9
    )


def test_partial_contrast_minimum_coverage_and_interval_guards_are_separate() -> None:
    one_driver = _retain_compounds(_report(drivers_per_compound=1), {"SOFT", "MEDIUM"})
    insufficient_point = evaluate_relative_tyre_wear(one_driver)["events"][0][
        "identifiable_contrasts"
    ]["soft_minus_medium"]
    assert insufficient_point["rank"]["estimable"] is True
    assert insufficient_point["status"] == "unavailable"
    assert insufficient_point["reason"] == "minimum_compound_coverage"
    assert insufficient_point["estimate_seconds_per_lap"] is None

    eight_clusters = _retain_compounds(
        _report(drivers_per_compound=4), {"SOFT", "MEDIUM"}
    )
    low_cluster_interval = evaluate_relative_tyre_wear(eight_clusters)["events"][0][
        "identifiable_contrasts"
    ]["soft_minus_medium"]
    assert low_cluster_interval["status"] == "available"
    assert low_cluster_interval["estimate_seconds_per_lap"] == pytest.approx(0.03)
    assert low_cluster_interval["standard_error_seconds_per_lap"] is None
    assert low_cluster_interval["interval_seconds_per_lap"] is None
    assert low_cluster_interval["uncertainty_reason"] == (
        "fewer_than_ten_driver_event_clusters"
    )

    ten_clusters = _retain_compounds(
        _report(drivers_per_compound=6, noise=True), {"SOFT", "MEDIUM"}
    )
    for lap in ten_clusters["evidence"]["laps"]:
        if lap["driver_number"] in {11, 12}:
            lap["eligible"] = False
            lap["exclusions"] = ["pit_affected"]
    ten_clusters["evidence"]["summary"] = _summary(ten_clusters["evidence"]["laps"])
    low_pair_interval = evaluate_relative_tyre_wear(ten_clusters)["events"][0][
        "identifiable_contrasts"
    ]["soft_minus_medium"]
    assert low_pair_interval["coverage"]["driver_event_clusters"] == 10
    assert low_pair_interval["coverage"]["driver_event_clusters_by_compound"] == {
        "SOFT": 6,
        "MEDIUM": 4,
        "HARD": 0,
    }
    assert low_pair_interval["status"] == "available"
    assert low_pair_interval["estimate_seconds_per_lap"] is not None
    assert low_pair_interval["standard_error_seconds_per_lap"] is None
    assert low_pair_interval["uncertainty_reason"] == (
        "fewer_than_five_driver_event_clusters_for_a_compound"
    )


def test_driver_trend_can_remove_an_otherwise_estimable_partial_contrast() -> None:
    report = _retain_compounds(_report(drivers_per_compound=6), {"SOFT", "MEDIUM"})
    result = evaluate_relative_tyre_wear(report)
    base = result["events"][0]["identifiable_contrasts"]["soft_minus_medium"]
    trend = result["driver_trend_sensitivity"]["events"][0]["identifiable_contrasts"][
        "soft_minus_medium"
    ]
    assert base["status"] == "available"
    assert base["rank"]["contrast"] == 1
    assert trend["status"] == "unavailable"
    assert trend["reason"] == "contrast_not_estimable"
    assert trend["rank"]["contrast"] == 0


def test_rank_zero_fit_reports_no_identifiable_contrasts() -> None:
    report = _synchronize_compounds_by_lap(_report(lap_count=18, drivers_per_compound=6))
    event = evaluate_relative_tyre_wear(report)["events"][0]
    assert event["status"] == "unavailable"
    for diagnostic in event["identifiable_contrasts"].values():
        assert diagnostic["status"] == "unavailable"
        assert diagnostic["reason"] == "contrast_not_estimable"
        assert diagnostic["estimate_seconds_per_lap"] is None
        assert diagnostic["rank"]["contrast"] == 0


def test_estimates_ignore_common_lap_trend_and_stint_origin_offsets() -> None:
    baseline = evaluate_relative_tyre_wear(_report())["pooled"]["estimates_seconds_per_lap"]
    altered = _report(
        common_shift=lambda lap: -4.0 * lap + 0.02 * lap**2,
        stint_offsets=lambda driver, lap: (
            (driver % 7) * 0.8 + _BY_COMPOUND[("SOFT", "MEDIUM", "HARD")[(driver - 1) // 4]] * 37
        ),
        prior_wear=None,
    )
    shifted = evaluate_relative_tyre_wear(altered)["pooled"]["estimates_seconds_per_lap"]
    assert shifted == pytest.approx(baseline)


def test_driver_trend_sensitivity_can_report_nonidentifiability() -> None:
    result = evaluate_relative_tyre_wear(_report())
    assert result["pooled_event_blocks"][0]["contrast_rank"] == 2
    assert result["driver_trend_sensitivity"]["pooled"]["status"] == "unavailable"
    assert result["driver_trend_sensitivity"]["pooled"]["reason"] == "insufficient_events"
    assert (
        result["driver_trend_sensitivity"]["events"][0]["reason"]
        == "compound_contrast_not_identifiable"
    )
    assert result["driver_trend_sensitivity"]["pooled_event_blocks"] == []
    assert (
        result["driver_trend_sensitivity"]["pooled_event_exclusions"][0]["reason"]
        == "no_compound_lap_contrast_after_fixed_effects"
    )


def test_missing_compound_coverage_is_structured_unavailable() -> None:
    report = _report()
    for lap in report["evidence"]["laps"]:
        if lap["compound"] == "HARD":
            lap["eligible"] = False
            lap["exclusions"] = ["pit_affected"]
    report["evidence"]["summary"] = _summary(report["evidence"]["laps"])
    result = evaluate_relative_tyre_wear(report)
    assert result["events"][0]["status"] == "unavailable"
    assert result["events"][0]["reason"] == "minimum_compound_coverage"
    assert result["pooled"]["status"] == "unavailable"
    assert result["coverage"]["ineligible_laps"] == 4 * 20
    assert result["coverage"]["exclusion_counts"] == {"pit_affected": 4 * 20}


def test_pooled_fit_keeps_usable_partial_event_blocks() -> None:
    partial = _report(event_key=101)
    for lap in partial["evidence"]["laps"]:
        if lap["compound"] == "HARD":
            lap["eligible"] = False
            lap["exclusions"] = ["pit_affected"]
    partial["evidence"]["summary"] = _summary(partial["evidence"]["laps"])
    complete = _report(event_key=102, drivers_per_compound=8)

    result = evaluate_relative_tyre_wear([partial, complete])
    assert result["events"][0]["status"] == "unavailable"
    assert result["events"][0]["reason"] == "minimum_compound_coverage"
    assert result["pooled"]["status"] == "available"
    assert len(result["pooled"]["contributing_events"]) == 2
    assert result["pooled_event_exclusions"] == []
    assert result["pooled"]["coverage"]["exclusion_counts"] == {"pit_affected": 4 * 20}


def test_simultaneous_compound_changes_are_rank_deficient_not_zero() -> None:
    report = _synchronize_compounds_by_lap(_report(lap_count=18))
    result = evaluate_relative_tyre_wear(report)
    assert result["events"][0]["status"] == "unavailable"
    assert result["events"][0]["reason"] == "compound_contrast_not_identifiable"
    assert result["events"][0]["estimates_seconds_per_lap"] is None


def test_rank_zero_event_is_excluded_without_changing_pooled_uncertainty() -> None:
    informative = _report(event_key=101, drivers_per_compound=8, noise=True)
    zero_information = _synchronize_compounds_by_lap(
        _report(event_key=102, drivers_per_compound=8, lap_count=18)
    )
    reference = evaluate_relative_tyre_wear(informative)["pooled"]
    combined = evaluate_relative_tyre_wear([informative, zero_information])
    actual = combined["pooled"]

    assert combined["pooled_event_blocks"] == [
        {"event": {"season": 2026, "session_key": 101, "meeting": "Test 101"}, "contrast_rank": 2}
    ]
    assert combined["pooled_event_exclusions"][0]["event"]["session_key"] == 102
    assert (
        combined["pooled_event_exclusions"][0]["reason"]
        == "no_compound_lap_contrast_after_fixed_effects"
    )
    assert actual["estimates_seconds_per_lap"] == pytest.approx(
        reference["estimates_seconds_per_lap"]
    )
    assert actual["intervals_seconds_per_lap"] == reference["intervals_seconds_per_lap"]
    assert actual["uncertainty"] == reference["uncertainty"]
    assert actual["identifiable_contrasts"] == reference["identifiable_contrasts"]


def test_rank_one_event_blocks_can_identify_pooled_contrasts_collectively() -> None:
    soft_hard = _report(event_key=101, drivers_per_compound=6)
    medium_hard = _report(event_key=102, drivers_per_compound=6)
    for lap in soft_hard["evidence"]["laps"]:
        if lap["compound"] == "MEDIUM":
            lap["eligible"] = False
            lap["exclusions"] = ["pit_affected"]
    soft_hard["evidence"]["summary"] = _summary(soft_hard["evidence"]["laps"])
    for lap in medium_hard["evidence"]["laps"]:
        if lap["compound"] == "SOFT":
            lap["eligible"] = False
            lap["exclusions"] = ["pit_affected"]
    medium_hard["evidence"]["summary"] = _summary(medium_hard["evidence"]["laps"])

    result = evaluate_relative_tyre_wear([soft_hard, medium_hard])
    assert [block["contrast_rank"] for block in result["pooled_event_blocks"]] == [1, 1]
    assert result["pooled_event_exclusions"] == []
    assert result["pooled"]["status"] == "available"
    assert result["pooled"]["rank"]["contrast"] == 2
    assert all(
        contrast["rank"]["estimable"]
        for contrast in result["pooled"]["identifiable_contrasts"].values()
    )


def test_equal_event_weighting_and_leave_one_event_out() -> None:
    first = _report(event_key=101, drivers_per_compound=4)
    second = _report(
        event_key=102,
        drivers_per_compound=8,
        slopes={"SOFT": 0.11, "MEDIUM": 0.07, "HARD": 0.02},
    )
    result = evaluate_relative_tyre_wear([second, first])
    first_fit = result["events"][0]["estimates_seconds_per_lap"]
    second_fit = result["events"][1]["estimates_seconds_per_lap"]
    pooled = result["pooled"]["estimates_seconds_per_lap"]
    assert pooled["soft_minus_hard"] == pytest.approx(
        (first_fit["soft_minus_hard"] + second_fit["soft_minus_hard"]) / 2
    )
    assert pooled["medium_minus_hard"] == pytest.approx(
        (first_fit["medium_minus_hard"] + second_fit["medium_minus_hard"]) / 2
    )
    assert len(result["leave_one_event_out"]) == 2
    assert all(item["status"] == "available" for item in result["leave_one_event_out"])


def test_cluster_cr1_intervals_match_independent_full_design_calculation() -> None:
    report = _report(drivers_per_compound=8, noise=True)
    result = evaluate_relative_tyre_wear(report)["events"][0]
    expected = _manual_single_event_intervals(report)
    for name, interval in expected.items():
        actual = result["intervals_seconds_per_lap"][name]
        assert actual is not None
        assert actual["lower"] == pytest.approx(interval[0], abs=1e-8)
        assert actual["upper"] == pytest.approx(interval[1], abs=1e-8)


def test_event_order_is_deterministic_and_inputs_are_not_mutated() -> None:
    reports = [_report(event_key=102), _report(event_key=101)]
    original = copy.deepcopy(reports)
    result_a = evaluate_relative_tyre_wear(reports)
    result_b = evaluate_relative_tyre_wear(list(reversed(reports)))
    assert reports == original
    assert result_a == result_b


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report: report["evidence"].pop("normalizer_version"), "normalizer_version"),
        (
            lambda report: report["evidence"]["laps"][0].__setitem__("driver_number", True),
            "driver_number",
        ),
        (
            lambda report: report["evidence"]["laps"][0].__setitem__(
                "duration_seconds", float("nan")
            ),
            "duration_seconds",
        ),
        (
            lambda report: report["evidence"]["summary"].__setitem__("eligible_laps", 1),
            "eligible_laps",
        ),
        (lambda report: report.__setitem__("decoded_feed_sha256", {}), "decoded_feed_sha256"),
    ],
)
def test_invalid_versions_types_nonfinite_values_and_summaries_are_rejected(
    mutate, message
) -> None:
    report = _report()
    mutate(report)
    with pytest.raises(RelativeTyreWearInputError, match=message):
        evaluate_relative_tyre_wear(report)


def test_duplicate_event_and_duplicate_driver_lap_are_rejected() -> None:
    report = _report()
    with pytest.raises(RelativeTyreWearInputError, match="duplicate event identity"):
        evaluate_relative_tyre_wear([report, copy.deepcopy(report)])

    report = _report()
    report["evidence"]["laps"].append(copy.deepcopy(report["evidence"]["laps"][0]))
    report["evidence"]["summary"] = _summary(report["evidence"]["laps"])
    with pytest.raises(RelativeTyreWearInputError, match="duplicates a driver lap"):
        evaluate_relative_tyre_wear(report)


def test_null_prior_wear_is_counted_as_unknown() -> None:
    result = evaluate_relative_tyre_wear(_report(prior_wear=None))
    assert result["coverage"]["unknown_prior_wear_laps"] == 12 * 20


@pytest.mark.parametrize("all_excluded", [False, True])
def test_no_eligible_laps_returns_unavailable_without_runtime_warnings(all_excluded) -> None:
    report = _report(lap_count=20 if all_excluded else 0)
    for lap in report["evidence"]["laps"]:
        lap["eligible"] = False
        lap["exclusions"] = ["track_status_not_green"]
    report["evidence"]["summary"] = _summary(report["evidence"]["laps"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = evaluate_relative_tyre_wear(report)
    fits = result["events"] + [result["pooled"]]
    fits += result["driver_trend_sensitivity"]["events"]
    fits += [result["driver_trend_sensitivity"]["pooled"]]
    for fit in fits:
        assert fit["status"] == "unavailable"
        for contrast in fit["identifiable_contrasts"].values():
            assert contrast["status"] == "unavailable"
            assert contrast["estimate_seconds_per_lap"] is None
            assert contrast["interval_seconds_per_lap"] is None
            assert contrast["coverage"]["eligible_laps"] == 0


def test_cli_reads_bom_json_and_emits_nothing_for_bad_input(tmp_path, capsys) -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "evaluate_relative_tyre_wear.py"
    spec = importlib.util.spec_from_file_location("relative_tyre_wear_cli", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evidence = tmp_path / "evidence.json"
    evidence.write_text("\ufeff" + json.dumps(_report()), encoding="utf-8")
    assert module.main([str(evidence)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["normalizer_version_required"] == 1

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        module.main([str(invalid)])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Could not evaluate relative tyre wear" in captured.err
