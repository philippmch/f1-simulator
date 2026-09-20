"""Synthetic contract tests for official timing archive normalization."""

from __future__ import annotations

import json

import pytest

from f1sim.data.timing_evidence import TimingEvidenceError, normalize_timing_evidence


def _stream(*rows: tuple[str, dict]) -> str:
    return "\n".join(timestamp + json.dumps(payload) for timestamp, payload in rows) + "\n"


def _feeds(
    timing: str,
    *,
    app: str | None = None,
    track: str | None = None,
    weather: str | None = None,
    session: str | None = None,
) -> dict[str, str]:
    return {
        "TimingData": timing,
        "TimingAppData": app
        or _stream(
            (
                "00:00:00.000",
                {"Lines": {"1": {"Stints": [{"Compound": "SOFT", "StartLaps": 0}]}}},
            )
        ),
        "TrackStatus": track or _stream(("00:00:00.000", {"Status": "1"})),
        "WeatherData": weather or _stream(("00:00:00.000", {"Rainfall": "0"})),
        "SessionStatus": session or _stream(("00:00:00.000", {"Status": "Started"})),
    }


def _timing(*updates: tuple[str, dict]) -> str:
    return _stream(*updates)


def _driver(lap: int | str | None = None, duration: str | None = None, **extra: object) -> dict:
    value: dict[str, object] = dict(extra)
    value.setdefault("InPit", False)
    if lap is not None:
        value["NumberOfLaps"] = lap
    if duration is not None:
        value["LastLapTime"] = {"Value": duration}
    return {"Lines": {"1": value}}


def _lap(report: dict, number: int) -> dict:
    return next(item for item in report["laps"] if item["lap_number"] == number)


def test_paired_sparse_updates_keep_crossing_evidence_and_ignore_flag_only_updates() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1)),
        ("00:01:30.000", _driver(2, "1:30.500")),
        ("00:01:31.000", _driver(Speeds={"FL": 201})),
        ("00:01:32.000", _driver(Speeds={"FL": 200})),
    )
    report = normalize_timing_evidence(_feeds(timing))

    lap = _lap(report, 2)
    assert lap["duration_seconds"] == pytest.approx(90.5)
    assert lap["observed_start"] == "00:00:00.000"
    assert lap["observed_end"] == "00:01:30.000"
    assert lap["eligible"] is True
    assert report["summary"]["unpaired_duration_observations"] == 0
    assert report["summary"]["missing_duration"] == 1


def test_duration_without_same_delta_lap_count_is_unpaired_and_flag_update_does_not_fill() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1)),
        ("00:01:30.000", _driver(2)),
        ("00:01:31.000", _driver(duration="1:30.500")),
    )
    report = normalize_timing_evidence(_feeds(timing))

    lap = _lap(report, 2)
    assert lap["duration_seconds"] is None
    assert "missing_duration" in lap["exclusions"]
    assert report["summary"]["unpaired_duration_observations"] == 1


def test_stale_paired_lap_time_after_crossing_is_excluded() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1)),
        ("00:01:30.000", _driver(2)),
        ("00:01:31.000", _driver(2, "1:30.500")),
    )
    report = normalize_timing_evidence(_feeds(timing))

    lap = _lap(report, 2)
    assert lap["duration_seconds"] is None
    assert "stale_duration" in lap["exclusions"]
    assert report["summary"]["stale_duration_observations"] == 1


def test_missing_lap_count_and_invalid_duration_are_explicit() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(duration="1:30.000")),
        ("00:00:01.000", _driver(1, "0:00.000")),
    )
    report = normalize_timing_evidence(_feeds(timing))

    lap = _lap(report, 1)
    assert lap["duration_seconds"] is None
    assert "invalid_duration" in lap["exclusions"]
    assert report["summary"]["unpaired_duration_observations"] == 1
    assert report["summary"]["invalid_duration_observations"] == 1


def test_conflicting_duplicate_driver_lap_is_ambiguous() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1, "1:30.000")),
        ("00:00:00.000", _driver(1, "1:31.000")),
    )
    report = normalize_timing_evidence(_feeds(timing))

    lap = _lap(report, 1)
    assert lap["duration_seconds"] is None
    assert "ambiguous_duplicate_duration" in lap["exclusions"]
    assert report["summary"]["conflicting_duplicate_observations"] == 1


def test_stint_list_then_indexed_patch_uses_start_laps_as_reported_metadata() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1)),
        ("00:01:30.000", _driver(2, "1:30.000")),
        ("00:02:00.000", _driver(2)),
        ("00:03:00.000", _driver(3, "1:30.000")),
        ("00:04:30.000", _driver(4, "1:30.000")),
    )
    app = _stream(
        (
            "00:00:00.000",
            {"Lines": {"1": {"Stints": [{"Compound": "SOFT", "New": False, "StartLaps": 3}]}}},
        ),
        (
            "00:02:00.000",
            {
                "Lines": {
                    "1": {
                        "Stints": {"1": {"Compound": "MEDIUM", "New": True, "StartLaps": 0}}
                    }
                }
            },
        ),
    )
    report = normalize_timing_evidence(_feeds(timing, app=app))

    lap2 = _lap(report, 2)
    lap3 = _lap(report, 3)
    assert lap2["compound"] == "SOFT"
    assert lap2["prior_wear"] == 3
    assert lap3["compound"] == "MEDIUM"
    assert lap3["prior_wear"] == 0
    assert "stint_change_mid_lap" in lap3["exclusions"]


@pytest.mark.parametrize("correction,late_only,lap_three_compound", [
    ({"Compound": "M"}, False, "MEDIUM"),
    ({"StartLaps": 3}, False, "HARD"),
    ({"Compound": "M", "StartLaps": 3}, False, "MEDIUM"),
    ({"Compound": "M", "StartLaps": 3}, True, "HARD"),
])
def test_repeated_stint_index_corrections_exclude_touching_laps_retrospectively(
    correction, late_only, lap_three_compound,
) -> None:
    def line(lap: int, duration: str | None = None) -> dict:
        return _driver(lap, duration)["Lines"]["1"]

    timing = _timing(
        ("00:00:00.000", {"Lines": {"1": line(1), "2": line(1)}}),
        ("00:01:30.000", {"Lines": {"1": line(2, "1:30.000"),
                                      "2": line(2, "1:30.000")}}),
        ("00:03:00.000", {"Lines": {"1": line(3, "1:30.000"),
                                      "2": line(3, "1:30.000")}}),
        ("00:04:30.000", {"Lines": {"1": line(4, "1:30.000"),
                                      "2": line(4, "1:30.000")}}),
        ("00:06:00.000", {"Lines": {"1": line(5, "1:30.000"),
                                      "2": line(5, "1:30.000")}}),
        ("00:07:30.000", {"Lines": {"1": line(6, "1:30.000"),
                                      "2": line(6, "1:30.000")}}),
        ("00:09:00.000", {"Lines": {"1": line(7, "1:30.000"),
                                      "2": line(7, "1:30.000")}}),
        ("00:10:30.000", {"Lines": {"1": line(8, "1:30.000"),
                                      "2": line(8, "1:30.000")}}),
    )
    app = _stream(
        (
            "00:00:00.000",
            {
                "Lines": {
                    "1": {"Stints": [{"Compound": "H", "StartLaps": 0}]},
                    "2": {"Stints": [{"Compound": "H", "StartLaps": 0}]},
                }
            },
        ),
        (
            "00:01:30.000",
            {
                "Lines": {
                    "1": {"Stints": {"0": {"TotalLaps": 1}}},
                    "2": {"Stints": {"0": {"Compound": "HARD", "StartLaps": 0}}},
                }
            },
        ),
        (
            "00:03:00.000",
            {"Lines": {"1": {"Stints": {"0": {} if late_only else correction}}}},
        ),
        (
            "00:04:30.000",
            {"Lines": {"1": {"Stints": {"1": {"Compound": "S"}}}}},
        ),
        (
            "00:05:00.000",
            {"Lines": {"1": {"Stints": {"1": {"StartLaps": 0}}}}},
        ),
        (
            "00:06:00.000",
            {"Lines": {"1": {"Stints": {"0": correction if late_only else {
                "Compound": "H", "StartLaps": 0,
            }}}}},
        ),
        (
            "00:07:30.000",
            {"Lines": {"1": {"Stints": {"2": {"Compound": "S", "StartLaps": 0}}}}},
        ),
    )
    report = normalize_timing_evidence(_feeds(timing, app=app))
    driver_one = [lap for lap in report["laps"] if lap["driver_number"] == 1]
    driver_two = [lap for lap in report["laps"] if lap["driver_number"] == 2]

    # Even a correction first arriving after stint 1 becomes active excludes
    # all earlier laps touching index 0. A later reversion cannot undo it.
    assert all(
        "stint_metadata_corrected" in _lap({"laps": driver_one}, lap_number)["exclusions"]
        for lap_number in (1, 2, 3, 4)
    )
    assert _lap({"laps": driver_one}, 2)["compound"] == "HARD"
    assert _lap({"laps": driver_one}, 3)["compound"] == lap_three_compound
    assert _lap({"laps": driver_one}, 5)["eligible"] is True
    assert "stint_metadata_corrected" not in _lap({"laps": driver_one}, 5)["exclusions"]
    assert "stint_metadata_corrected" not in _lap({"laps": driver_one}, 6)["exclusions"]
    assert _lap({"laps": driver_one}, 7)["eligible"] is True

    # The unrelated driver's alias-only repeat is harmless, and its index 0
    # never inherits driver 1's correction.
    assert all("stint_metadata_corrected" not in lap["exclusions"] for lap in driver_two)
    assert _lap({"laps": driver_two}, 2)["eligible"] is True
    assert report["summary"]["exclusion_counts"]["stint_metadata_corrected"] == 4
    assert any("before the correction" in text for text in report["limitations"])


def test_pit_in_and_pit_out_inside_interval_exclude_lap() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1, InPit=False)),
        ("00:00:20.000", _driver(InPit=True)),
        ("00:00:50.000", _driver(InPit=False, PitOut=True)),
        ("00:01:30.000", _driver(2, "1:30.000")),
    )
    report = normalize_timing_evidence(_feeds(timing))

    lap = _lap(report, 2)
    assert "pit_affected" in lap["exclusions"]
    assert lap["eligible"] is False


def test_sparse_same_timestamp_pit_fields_do_not_raise_and_invalid_inpit_is_unknown() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1, InPit=False)),
        ("00:00:30.000", {"Lines": {"1": {"InPit": False}}}),
        ("00:00:30.000", {"Lines": {"1": {"PitOut": True}}}),
        ("00:01:00.000", {"Lines": {"1": {"InPit": "unknown"}}}),
        ("00:01:30.000", _driver(2, "1:30.000")),
    )
    report = normalize_timing_evidence(_feeds(timing))

    lap = _lap(report, 2)
    assert "pit_affected" in lap["exclusions"]
    assert "pit_status_unknown" in lap["exclusions"]


def test_invalid_pitout_remains_unknown_until_a_valid_pitout_update() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1, InPit=False)),
        ("00:00:30.000", {"Lines": {"1": {"PitOut": "unknown"}}}),
        ("00:01:30.000", _driver(2, "1:30.000")),
        ("00:03:00.000", _driver(3, "1:30.000", PitOut=False)),
        ("00:04:30.000", _driver(4, "1:30.000")),
    )
    report = normalize_timing_evidence(_feeds(timing))

    assert "pit_status_unknown" in _lap(report, 2)["exclusions"]
    assert "pit_status_unknown" in _lap(report, 3)["exclusions"]
    assert "pit_status_unknown" not in _lap(report, 4)["exclusions"]


def test_count_gap_only_excludes_affected_crossing() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1)),
        ("00:01:30.000", _driver(2, "1:30.000")),
        ("00:03:00.000", _driver(4, "1:30.000")),
    )
    report = normalize_timing_evidence(_feeds(timing))

    assert _lap(report, 2)["eligible"] is True
    assert "missing_preceding_crossing" in _lap(report, 4)["exclusions"]
    assert "nonconsecutive_or_regressed_lap_count" in _lap(report, 4)["exclusions"]


def test_count_regression_does_not_poison_prior_clean_interval() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1)),
        ("00:01:30.000", _driver(2, "1:30.000")),
        ("00:03:00.000", _driver(3, "1:30.000")),
        ("00:03:10.000", _driver(2)),
        ("00:04:30.000", _driver(4, "1:30.000")),
    )
    report = normalize_timing_evidence(_feeds(timing))

    assert _lap(report, 3)["eligible"] is True
    assert "nonconsecutive_or_regressed_lap_count" in _lap(report, 4)["exclusions"]


def test_track_status_and_rainfall_changes_inside_interval_fail_closed() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1)),
        ("00:01:30.000", _driver(2, "1:30.000")),
    )
    track = _stream(
        ("00:00:00.000", {"Status": "1"}),
        ("00:00:30.000", {"Status": "2"}),
    )
    weather = _stream(
        ("00:00:00.000", {"Rainfall": "0"}),
        ("00:00:40.000", {"Rainfall": "1"}),
    )
    report = normalize_timing_evidence(_feeds(timing, track=track, weather=weather))

    lap = _lap(report, 2)
    assert "track_status_not_green" in lap["exclusions"]
    assert "rainfall_reported" in lap["exclusions"]


def test_weather_metadata_delta_preserves_zero_but_null_is_unknown() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1)),
        ("00:01:30.000", _driver(2, "1:30.000")),
    )
    weather = _stream(
        ("00:00:00.000", {"Rainfall": "0"}),
        ("00:00:30.000", {"AirTemp": "20.0"}),
        ("00:00:40.000", {"Rainfall": None}),
    )
    report = normalize_timing_evidence(_feeds(timing, weather=weather))

    lap = _lap(report, 2)
    assert "weather_rainfall_unknown" in lap["exclusions"]


def test_unknown_metadata_and_prerace_lap_are_excluded() -> None:
    timing = _timing(
        ("00:00:00.000", _driver(1)),
        ("00:01:30.000", _driver(2, "1:30.000")),
    )
    session = _stream(("00:00:30.000", {"Status": "Started"}))
    report = normalize_timing_evidence(
        {
            "TimingData": timing,
            "TimingAppData": _stream(("00:00:00.000", {"Lines": {}})),
            "TrackStatus": "",
            "WeatherData": "",
            "SessionStatus": session,
        }
    )

    lap = _lap(report, 2)
    assert "before_session_start" in lap["exclusions"]
    assert "missing_stint_compound" in lap["exclusions"]
    assert "track_status_unknown" in lap["exclusions"]
    assert "weather_rainfall_unknown" in lap["exclusions"]


def test_utf8_bom_and_prefixed_stream_are_supported() -> None:
    timing = (
        "\ufeff00:00:00.000"
        + json.dumps(_driver(1))
        + "\n"
        + "00:01:30.000"
        + json.dumps(_driver(2, "1:30.000"))
    )
    report = normalize_timing_evidence(_feeds(timing))
    assert report["coverage"]["feeds"]["TimingData"]["rows"] == 2


@pytest.mark.parametrize(
    "timing, message",
    [
        ("00:00:00.000{bad-json}\n", "malformed JSON"),
        (
            "00:01:00.000{}\n00:00:59.000{}\n",
            "earlier than the preceding row",
        ),
    ],
)
def test_malformed_and_nonmonotonic_streams_fail_with_feed_and_line(
    timing: str, message: str
) -> None:
    with pytest.raises(TimingEvidenceError, match=message):
        normalize_timing_evidence({"TimingData": timing})
