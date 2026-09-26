"""Offline estimates of relative slick-tyre lap-time trends."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np

from f1sim.data.timing_evidence import NORMALIZER_VERSION

_FEEDS = ("TimingData", "TimingAppData", "TrackStatus", "WeatherData", "SessionStatus")
_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
_ESTIMATE_NAMES = ("soft_minus_hard", "medium_minus_hard", "soft_minus_medium")
_CONDITION_LIMIT = 1e12
_MAX_EVENTS = 100
_MAX_TOTAL_LAPS = 100_000
_MAX_EVENT_ROWS = 10_000
_MAX_EVENT_DESIGN_CELLS = 5_000_000


class RelativeTyreWearInputError(ValueError):
    """Raised when saved timing evidence is incomplete or inconsistent."""


class _Unavailable(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def evaluate_relative_tyre_wear(reports: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
    """Evaluate corrected archive reports without network access or input mutation.

    ``reports`` is one archive wrapper returned by ``check_tyre_evidence.py`` or a
    sequence of those wrappers. Reports must contain individual lap observations
    produced by the current timing-evidence normalizer.
    """

    events = _validate_reports(reports)
    pooled_events, pooled_blocks, pooled_exclusions = _select_pooled_events(
        events, driver_trend=False
    )
    event_results: list[dict[str, Any]] = []
    for event in events:
        coverage = _coverage([event])
        try:
            _require_coverage([event])
            fitted = _fit([event], weighting="single_event", driver_trend=False)
        except _Unavailable as exc:
            event_results.append({"event": _event_identity(event), **_unavailable(exc, coverage)})
            continue
        event_results.append({"event": _event_identity(event), **fitted})

    pooled = _pooled_result(pooled_events, weighting="equal_event_total", driver_trend=False)
    leave_one_out: list[dict[str, Any]] = []
    for omitted in pooled_events:
        remaining = [event for event in pooled_events if event is not omitted]
        if not remaining:
            result = _unavailable(
                _Unavailable("no_events_after_omission", "No contributing events remain."),
                _coverage([]),
            )
        else:
            result = _pooled_result(remaining, weighting="equal_event_total", driver_trend=False)
        leave_one_out.append({"omitted_event": _event_identity(omitted), **result})

    trend_events: list[dict[str, Any]] = []
    for event in events:
        try:
            fitted = _fit([event], weighting="single_event", driver_trend=True)
        except _Unavailable as exc:
            fitted = _unavailable(exc, _coverage([event]))
        trend_events.append({"event": _event_identity(event), **fitted})
    trend_pool_events, trend_pool_blocks, trend_pool_exclusions = _select_pooled_events(
        events, driver_trend=True
    )
    driver_trend = {
        "description": (
            "Sensitivity fit adds one linear lap trend per driver within each event; "
            "it is descriptive and may be unidentifiable."
        ),
        "events": trend_events,
        "pooled": _pooled_result(
            trend_pool_events,
            weighting="equal_event_total",
            driver_trend=True,
        ),
        "pooled_event_blocks": trend_pool_blocks,
        "pooled_event_exclusions": trend_pool_exclusions,
    }

    return {
        "schema_version": 1,
        "method": "event_driver_stint_and_event_lap_fixed_effects_v1",
        "normalizer_version_required": NORMALIZER_VERSION,
        "interpretation": (
            "Relative descriptive lap-time slopes only. A common lap trend is absorbed "
            "by event-lap effects; coefficients are not absolute wear rates or causal effects."
        ),
        "provenance": {"events": [_public_provenance(event) for event in events]},
        "coverage": _coverage(events),
        "events": event_results,
        "pooled": pooled,
        "pooled_event_blocks": pooled_blocks,
        "pooled_event_exclusions": pooled_exclusions,
        "leave_one_event_out": leave_one_out,
        "driver_trend_sensitivity": driver_trend,
    }


def _validate_reports(reports: Mapping[str, Any] | Sequence[Any]) -> list[dict[str, Any]]:
    if isinstance(reports, Mapping):
        rows = [reports]
    elif isinstance(reports, Sequence) and not isinstance(reports, (str, bytes, bytearray)):
        rows = list(reports)
    else:
        raise RelativeTyreWearInputError(
            "reports must be an archive report or a flat list of reports"
        )
    if not rows:
        raise RelativeTyreWearInputError("at least one archive report is required")
    if len(rows) > _MAX_EVENTS:
        raise RelativeTyreWearInputError(f"at most {_MAX_EVENTS} events can be evaluated")

    events: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    total_laps = 0
    for report_index, report in enumerate(rows):
        event = _validate_report(report, report_index)
        identity = (event["season"], event["session_key"])
        if identity in identities:
            raise RelativeTyreWearInputError(
                f"duplicate event identity season={identity[0]}, session_key={identity[1]}"
            )
        identities.add(identity)
        total_laps += len(event["laps"])
        if total_laps > _MAX_TOTAL_LAPS:
            raise RelativeTyreWearInputError(
                f"reports exceed the {_MAX_TOTAL_LAPS}-lap analysis limit"
            )
        events.append(event)
    return sorted(events, key=lambda item: (item["season"], item["session_key"]))


def _validate_report(report: Any, index: int) -> dict[str, Any]:
    prefix = f"report {index + 1}"
    if not isinstance(report, Mapping):
        raise RelativeTyreWearInputError(f"{prefix} must be an object")
    season = _positive_int(report.get("season"), f"{prefix}.season")
    session_key = _positive_int(report.get("session_key"), f"{prefix}.session_key")
    meeting = report.get("meeting")
    if not isinstance(meeting, str) or not meeting.strip():
        raise RelativeTyreWearInputError(f"{prefix}.meeting must be a nonempty string")
    source = report.get("source")
    try:
        parsed_source = urlparse(source) if isinstance(source, str) else None
        source_host = parsed_source.hostname if parsed_source is not None else None
    except ValueError:
        parsed_source = None
        source_host = None
    if (
        parsed_source is None
        or parsed_source.scheme != "https"
        or source_host != "livetiming.formula1.com"
        or parsed_source.netloc != "livetiming.formula1.com"
        or parsed_source.query
        or parsed_source.fragment
        or parsed_source.username
        or parsed_source.password
        or not re.fullmatch(rf"/static/{season}/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/", parsed_source.path)
    ):
        raise RelativeTyreWearInputError(f"{prefix}.source is not a canonical season archive path")
    try:
        retrieved = datetime.fromisoformat(
            str(report.get("retrieved_at", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise RelativeTyreWearInputError(f"{prefix}.retrieved_at must be an ISO timestamp") from exc
    if retrieved.utcoffset() is None:
        raise RelativeTyreWearInputError(f"{prefix}.retrieved_at must include a timezone")

    source_feeds = report.get("source_feeds")
    if (
        not isinstance(source_feeds, list)
        or len(source_feeds) != len(_FEEDS)
        or any(not isinstance(feed, str) for feed in source_feeds)
        or set(source_feeds) != set(_FEEDS)
    ):
        raise RelativeTyreWearInputError(f"{prefix}.source_feeds must list all five archive feeds")
    hashes = report.get("decoded_feed_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(_FEEDS):
        raise RelativeTyreWearInputError(f"{prefix}.decoded_feed_sha256 must hash all five feeds")
    for feed in _FEEDS:
        if not isinstance(hashes[feed], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", hashes[feed]):
            raise RelativeTyreWearInputError(f"{prefix} has an invalid SHA-256 for {feed}")

    evidence = report.get("evidence")
    if not isinstance(evidence, Mapping):
        raise RelativeTyreWearInputError(
            f"{prefix}.evidence must be an object with lap observations"
        )
    version = evidence.get("normalizer_version")
    if type(version) is not int or version != NORMALIZER_VERSION:
        raise RelativeTyreWearInputError(
            f"{prefix} requires normalizer_version={NORMALIZER_VERSION}; regenerate this report"
        )
    _validate_feed_coverage(evidence.get("coverage"), prefix)
    laps = evidence.get("laps")
    if not isinstance(laps, list):
        raise RelativeTyreWearInputError(
            f"{prefix}.evidence.laps is missing; collect the report with --include-laps"
        )

    normalized_laps: list[dict[str, Any]] = []
    seen_laps: set[tuple[tuple[str, int | str], int]] = set()
    reason_counts: Counter[str] = Counter()
    durations = 0
    eligible_count = 0
    for lap_index, lap in enumerate(laps):
        lap_prefix = f"{prefix}.evidence.laps[{lap_index}]"
        if not isinstance(lap, Mapping):
            raise RelativeTyreWearInputError(f"{lap_prefix} must be an object")
        driver = _driver_id(lap.get("driver_number"), f"{lap_prefix}.driver_number")
        driver_key = _driver_key(driver)
        lap_number = _positive_int(lap.get("lap_number"), f"{lap_prefix}.lap_number")
        unique = (driver_key, lap_number)
        if unique in seen_laps:
            raise RelativeTyreWearInputError(f"{lap_prefix} duplicates a driver lap")
        seen_laps.add(unique)

        duration = lap.get("duration_seconds")
        if duration is not None:
            duration = _positive_finite_number(duration, f"{lap_prefix}.duration_seconds")
            durations += 1
        eligible = lap.get("eligible")
        if type(eligible) is not bool:
            raise RelativeTyreWearInputError(f"{lap_prefix}.eligible must be a boolean")
        exclusions = lap.get("exclusions")
        if (
            not isinstance(exclusions, list)
            or any(not isinstance(reason, str) or not reason for reason in exclusions)
            or len(set(exclusions)) != len(exclusions)
            or (eligible and exclusions)
            or (not eligible and not exclusions)
        ):
            raise RelativeTyreWearInputError(
                f"{lap_prefix}.eligible and exclusions are inconsistent"
            )
        reason_counts.update(exclusions)

        stint = lap.get("stint")
        prior_wear = lap.get("prior_wear")
        if prior_wear is not None and (type(prior_wear) is not int or prior_wear < 0):
            raise RelativeTyreWearInputError(
                f"{lap_prefix}.prior_wear must be null or a nonnegative integer"
            )
        if eligible:
            eligible_count += 1
            if duration is None:
                raise RelativeTyreWearInputError(f"{lap_prefix} is eligible without a duration")
            if type(stint) is not int or stint < 0:
                raise RelativeTyreWearInputError(
                    f"{lap_prefix}.stint must be a nonnegative integer"
                )
            if (
                type(lap.get("start_stint")) is not int
                or type(lap.get("end_stint")) is not int
                or lap["start_stint"] != stint
                or lap["end_stint"] != stint
            ):
                raise RelativeTyreWearInputError(f"{lap_prefix} crosses or omits stint metadata")
            compound = lap.get("compound")
            if compound not in _COMPOUNDS:
                raise RelativeTyreWearInputError(
                    f"{lap_prefix} is eligible with an unsupported slick compound"
                )
            if lap.get("start_compound") != compound or lap.get("end_compound") != compound:
                raise RelativeTyreWearInputError(f"{lap_prefix} has inconsistent compound metadata")
            if not _clock_text(lap.get("observed_start")) or not _clock_text(
                lap.get("observed_end")
            ):
                raise RelativeTyreWearInputError(
                    f"{lap_prefix} is eligible without paired crossings"
                )
            normalized_laps.append(
                {
                    "driver": driver,
                    "driver_key": driver_key,
                    "lap_number": lap_number,
                    "duration_seconds": duration,
                    "stint": stint,
                    "compound": compound,
                    "prior_wear": prior_wear,
                }
            )

    _validate_summary(
        evidence.get("summary"), laps, eligible_count, durations, reason_counts, prefix
    )
    return {
        "season": season,
        "session_key": session_key,
        "meeting": meeting.strip(),
        "source": source,
        "retrieved_at": report["retrieved_at"],
        "source_feeds": list(source_feeds),
        "decoded_feed_sha256": {feed: hashes[feed] for feed in _FEEDS},
        "laps": sorted(
            normalized_laps,
            key=lambda row: (_driver_sort(row["driver"]), row["lap_number"]),
        ),
        "all_lap_count": len(laps),
        "exclusion_counts": dict(reason_counts),
    }


def _validate_feed_coverage(value: Any, prefix: str) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get("feeds"), Mapping):
        raise RelativeTyreWearInputError(f"{prefix}.evidence.coverage.feeds is missing")
    feeds = value["feeds"]
    if set(feeds) != set(_FEEDS):
        raise RelativeTyreWearInputError(f"{prefix}.evidence.coverage must describe all five feeds")
    for feed in _FEEDS:
        row = feeds[feed]
        if (
            not isinstance(row, Mapping)
            or row.get("present") is not True
            or type(row.get("rows")) is not int
            or row["rows"] < 0
        ):
            raise RelativeTyreWearInputError(f"{prefix}.evidence.coverage is incomplete for {feed}")


def _validate_summary(
    summary: Any,
    laps: list[Any],
    eligible_count: int,
    durations: int,
    reason_counts: Counter[str],
    prefix: str,
) -> None:
    if not isinstance(summary, Mapping):
        raise RelativeTyreWearInputError(f"{prefix}.evidence.summary is missing")
    expected = {
        "total_laps": len(laps),
        "eligible_laps": eligible_count,
        "duration_observations": durations,
        "missing_duration": reason_counts.get("missing_duration", 0),
        "stale_duration": reason_counts.get("stale_duration", 0),
        "ambiguous_laps": sum(
            1
            for lap in laps
            if any("ambiguous" in reason or "duplicate" in reason for reason in lap["exclusions"])
        ),
        "missing_preceding_crossing": reason_counts.get("missing_preceding_crossing", 0),
    }
    for key, count in expected.items():
        if type(summary.get(key)) is not int or summary[key] != count:
            raise RelativeTyreWearInputError(
                f"{prefix}.evidence.summary.{key} does not match the laps"
            )
    observed = summary.get("exclusion_counts")
    if not isinstance(observed, Mapping) or dict(observed) != dict(reason_counts):
        raise RelativeTyreWearInputError(
            f"{prefix}.evidence.summary.exclusion_counts does not match the lap exclusions"
        )
    for key in (
        "duplicate_identical_observations",
        "conflicting_duplicate_observations",
        "unpaired_duration_observations",
        "nonempty_unpaired_duration_observations",
        "invalid_duration_observations",
        "invalid_lap_count_observations",
        "stale_duration_observations",
    ):
        if type(summary.get(key)) is not int or summary[key] < 0:
            raise RelativeTyreWearInputError(
                f"{prefix}.evidence.summary.{key} must be a nonnegative integer"
            )
    if (
        summary["unpaired_duration_observations"]
        != summary["nonempty_unpaired_duration_observations"]
    ):
        raise RelativeTyreWearInputError(
            f"{prefix}.evidence summary has inconsistent unpaired counts"
        )


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise RelativeTyreWearInputError(f"{field} must be a positive integer")
    return value


def _driver_id(value: Any, field: str) -> int | str:
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise RelativeTyreWearInputError(f"{field} must be a nonempty string or positive integer")


def _driver_key(value: int | str) -> tuple[str, int | str]:
    return ("int", value) if isinstance(value, int) else ("str", value)


def _driver_sort(value: int | str) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def _positive_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RelativeTyreWearInputError(f"{field} must be a finite positive number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise RelativeTyreWearInputError(f"{field} must be a finite positive number")
    return numeric


def _clock_text(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"\d{2}:\d{2}:\d{2}\.\d{3}", value))


def _event_identity(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "season": event["season"],
        "session_key": event["session_key"],
        "meeting": event["meeting"],
    }


def _public_provenance(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_event_identity(event),
        "source": event["source"],
        "retrieved_at": event["retrieved_at"],
        "source_feeds": list(event["source_feeds"]),
        "decoded_feed_sha256": dict(event["decoded_feed_sha256"]),
    }


def _coverage(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible_laps = [lap for event in events for lap in event["laps"]]
    exclusions: Counter[str] = Counter()
    for event in events:
        exclusions.update(event["exclusion_counts"])
    per_compound: dict[str, set[tuple[int, int, tuple[str, int | str]]]] = {
        compound: set() for compound in _COMPOUNDS
    }
    for event in events:
        for lap in event["laps"]:
            per_compound[lap["compound"]].add(
                (event["season"], event["session_key"], lap["driver_key"])
            )
    return {
        "event_count": len(events),
        "eligible_laps": len(eligible_laps),
        "all_reported_laps": sum(event["all_lap_count"] for event in events),
        "ineligible_laps": sum(event["all_lap_count"] for event in events) - len(eligible_laps),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "distinct_event_laps": sum(
            len({lap["lap_number"] for lap in event["laps"]}) for event in events
        ),
        "driver_event_clusters": len(
            {
                (event["season"], event["session_key"], lap["driver_key"])
                for event in events
                for lap in event["laps"]
            }
        ),
        "unknown_prior_wear_laps": sum(lap["prior_wear"] is None for lap in eligible_laps),
        "driver_event_clusters_by_compound": {
            compound: len(per_compound[compound]) for compound in _COMPOUNDS
        },
    }


def _pooled_event_exclusion(event: Mapping[str, Any]) -> dict[str, str] | None:
    laps = event["laps"]
    if len({lap["lap_number"] for lap in laps}) < 2:
        return {
            "reason": "insufficient_lap_variation",
            "detail": "At least two distinct event laps are required for a pooled event block.",
        }
    if len({lap["compound"] for lap in laps}) < 2:
        return {
            "reason": "insufficient_within_event_compound_variation",
            "detail": "A pooled event block must include at least two compounds.",
        }
    return None


def _select_pooled_events(
    events: Sequence[dict[str, Any]], *, driver_trend: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for event in events:
        exclusion = _pooled_event_exclusion(event)
        if exclusion is not None:
            exclusions.append({"event": _event_identity(event), **exclusion})
            continue
        try:
            _, residualized_x, original_x, _ = _residualize_event(event, driver_trend=driver_trend)
            rank = _contrast_rank(residualized_x, original_x)
        except _Unavailable as exc:
            exclusions.append(
                {
                    "event": _event_identity(event),
                    "reason": exc.reason,
                    "detail": exc.detail,
                }
            )
            continue
        if rank == 0:
            exclusions.append(
                {
                    "event": _event_identity(event),
                    "reason": "no_compound_lap_contrast_after_fixed_effects",
                    "detail": (
                        "The event's residualized compound-by-lap regressors have rank zero "
                        "after its fixed effects."
                    ),
                }
            )
            continue
        selected.append(event)
        blocks.append({"event": _event_identity(event), "contrast_rank": rank})
    return selected, blocks, exclusions


def _contrast_rank(residualized_x: np.ndarray, original_x: np.ndarray) -> int:
    try:
        singular_values = np.linalg.svd(residualized_x, compute_uv=False)
        original_values = np.linalg.svd(original_x, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise _Unavailable(
            "numerical_failure", "The event contrast rank did not converge."
        ) from exc
    reference_scale = max(float(original_values[0]) if len(original_values) else 0.0, 1.0)
    tolerance = 10.0 * np.finfo(np.float64).eps * max(residualized_x.shape) * reference_scale
    return int(np.count_nonzero(singular_values > tolerance))


def _require_coverage(events: Sequence[Mapping[str, Any]], *, pooled: bool = False) -> None:
    if not events:
        raise _Unavailable("insufficient_events", "No events meet the minimum compound coverage.")
    if pooled:
        for event in events:
            exclusion = _pooled_event_exclusion(event)
            if exclusion is not None:
                raise _Unavailable(exclusion["reason"], exclusion["detail"])
        for compound in _COMPOUNDS:
            drivers = {
                (event["season"], event["session_key"], lap["driver_key"])
                for event in events
                for lap in event["laps"]
                if lap["compound"] == compound
            }
            if len(drivers) < 2:
                raise _Unavailable(
                    "minimum_compound_coverage",
                    f"At least two drivers must have eligible {compound.lower()} laps "
                    "across the pooled events.",
                )
        return
    for event in events:
        laps = event["laps"]
        if len({lap["lap_number"] for lap in laps}) < 2:
            raise _Unavailable(
                "insufficient_lap_variation", "At least two distinct event laps are required."
            )
        for compound in _COMPOUNDS:
            drivers = {lap["driver_key"] for lap in laps if lap["compound"] == compound}
            if len(drivers) < 2:
                raise _Unavailable(
                    "minimum_compound_coverage",
                    f"At least two drivers must have eligible {compound.lower()} laps.",
                )


def _unavailable(exc: _Unavailable, coverage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": exc.reason,
        "detail": exc.detail,
        "coverage": dict(coverage),
        "rank": None,
        "condition_number": None,
        "estimates_seconds_per_lap": None,
        "intervals_seconds_per_lap": None,
    }


def _pooled_result(
    events: Sequence[dict[str, Any]], *, weighting: str, driver_trend: bool
) -> dict[str, Any]:
    coverage = _coverage(events)
    if not events:
        return _unavailable(
            _Unavailable("insufficient_events", "No events meet the minimum compound coverage."),
            coverage,
        )
    try:
        _require_coverage(events, pooled=weighting == "equal_event_total")
        return _fit(events, weighting=weighting, driver_trend=driver_trend)
    except _Unavailable as exc:
        return _unavailable(exc, coverage)


def _fit(events: Sequence[dict[str, Any]], *, weighting: str, driver_trend: bool) -> dict[str, Any]:
    pooled = weighting == "equal_event_total"
    _require_coverage(events, pooled=pooled)
    prepared: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, int]] = []
    for event in events:
        y, x, original_x, nuisance_rank = _residualize_event(event, driver_trend=driver_trend)
        prepared.append((event, y, x, original_x, nuisance_rank))

    y_parts: list[np.ndarray] = []
    x_parts: list[np.ndarray] = []
    original_x_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    cluster_parts: list[np.ndarray] = []
    nuisance_rank = 0
    cluster_labels: dict[tuple[int, int, tuple[str, int | str]], int] = {}
    for event, y, x, original_x, event_nuisance_rank in prepared:
        nuisance_rank += event_nuisance_rank
        y_parts.append(y)
        x_parts.append(x)
        original_x_parts.append(original_x)
        if weighting == "equal_event_total":
            weight_parts.append(np.full(len(y), 1.0 / len(y), dtype=np.float64))
        else:
            weight_parts.append(np.ones(len(y), dtype=np.float64))
        labels: list[int] = []
        for lap in event["laps"]:
            cluster = (event["season"], event["session_key"], lap["driver_key"])
            if cluster not in cluster_labels:
                cluster_labels[cluster] = len(cluster_labels)
            labels.append(cluster_labels[cluster])
        cluster_parts.append(np.asarray(labels, dtype=np.int64))

    y_all = np.concatenate(y_parts)
    x_all = np.concatenate(x_parts)
    original_x_all = np.concatenate(original_x_parts)
    weights = np.concatenate(weight_parts)
    clusters = np.concatenate(cluster_parts)
    root_weights = np.sqrt(weights)
    weighted_x = x_all * root_weights[:, None]
    weighted_y = y_all * root_weights
    try:
        beta, _, _, _ = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)
        _, singular_values, right_vectors = np.linalg.svd(weighted_x, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise _Unavailable(
            "numerical_failure", "The contrast regression did not converge."
        ) from exc
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    original_singular_values = np.linalg.svd(
        original_x_all * root_weights[:, None], compute_uv=False
    )
    reference_scale = max(
        float(original_singular_values[0]) if len(original_singular_values) else 0.0,
        1.0,
    )
    # Comparing only the two residualized singular values can mistake a pair of
    # rounding-noise columns for a well-conditioned, estimable contrast.
    tolerance = 10.0 * np.finfo(np.float64).eps * max(weighted_x.shape) * reference_scale
    numeric_rank = int(np.count_nonzero(singular_values > tolerance))
    if numeric_rank < 2:
        raise _Unavailable(
            "compound_contrast_not_identifiable",
            "The residualized compound-by-lap contrasts are rank deficient after fixed effects.",
        )
    condition_number = largest / float(singular_values[-1])
    if not math.isfinite(condition_number) or condition_number > _CONDITION_LIMIT:
        raise _Unavailable(
            "compound_contrast_ill_conditioned",
            f"Residualized compound contrasts have condition number {condition_number:.6g}.",
        )
    if not np.all(np.isfinite(beta)):
        raise _Unavailable(
            "numerical_failure", "The contrast regression produced non-finite estimates."
        )

    residual = y_all - x_all @ beta
    cluster_count = len(cluster_labels)
    full_rank = nuisance_rank + numeric_rank
    residual_df = len(y_all) - full_rank
    covariance = None
    interval_reason: str | None = None
    compound_clusters = _coverage(events)["driver_event_clusters_by_compound"]
    if cluster_count < 10:
        interval_reason = "fewer_than_ten_driver_event_clusters"
    elif residual_df <= 0:
        interval_reason = "no_residual_degrees_of_freedom"
    elif min(compound_clusters.values(), default=0) < 5:
        interval_reason = "fewer_than_five_driver_event_clusters_for_a_compound"
    else:
        try:
            inverse_squares = 1.0 / np.square(singular_values)
            bread = (right_vectors.T * inverse_squares) @ right_vectors
            scores = np.zeros((cluster_count, 2), dtype=np.float64)
            np.add.at(scores, clusters, (weights * residual)[:, None] * x_all)
            correction = (cluster_count / (cluster_count - 1)) * ((len(y_all) - 1) / residual_df)
            covariance = correction * bread @ (scores.T @ scores) @ bread
            if not np.all(np.isfinite(covariance)):
                covariance = None
                interval_reason = "non_finite_cluster_covariance"
        except np.linalg.LinAlgError:
            interval_reason = "singular_contrast_covariance"

    estimates = {
        _ESTIMATE_NAMES[0]: float(beta[0]),
        _ESTIMATE_NAMES[1]: float(beta[1]),
        _ESTIMATE_NAMES[2]: float(beta[0] - beta[1]),
    }
    if not all(math.isfinite(value) for value in estimates.values()):
        raise _Unavailable(
            "numerical_failure", "The contrast regression produced non-finite estimates."
        )
    intervals: dict[str, dict[str, float] | None] = {name: None for name in _ESTIMATE_NAMES}
    if covariance is not None:
        contrasts = (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, -1.0]))
        for name, contrast in zip(_ESTIMATE_NAMES, contrasts):
            variance = float(contrast @ covariance @ contrast)
            if variance < 0 and variance > -1e-12:
                variance = 0.0
            if variance < 0 or not math.isfinite(variance):
                intervals[name] = None
                interval_reason = "invalid_cluster_variance"
                continue
            half_width = 1.96 * math.sqrt(variance)
            lower = estimates[name] - half_width
            upper = estimates[name] + half_width
            if math.isfinite(lower) and math.isfinite(upper):
                intervals[name] = {"lower": lower, "upper": upper}
            else:
                interval_reason = "non_finite_cluster_interval"

    coverage = _coverage(events)
    return {
        "status": "available",
        "weighting": "equal_total_per_event"
        if weighting == "equal_event_total"
        else "single_event_unweighted",
        "driver_trend_adjusted": driver_trend,
        "contributing_events": [_event_identity(event) for event in events],
        "coverage": coverage,
        "rank": {
            "nuisance": nuisance_rank,
            "contrast": numeric_rank,
            "full": full_rank,
            "residual_degrees_of_freedom": residual_df,
        },
        "condition_number": condition_number,
        "estimates_seconds_per_lap": estimates,
        "intervals_seconds_per_lap": intervals,
        "uncertainty": {
            "method": "CR1 driver-event clustered sandwich with 1.96 normal multiplier",
            "driver_event_clusters": cluster_count,
            "driver_event_clusters_by_compound": compound_clusters,
            "available": covariance is not None,
            "reason": interval_reason,
        },
    }


def _residualize_event(
    event: Mapping[str, Any], *, driver_trend: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    laps = event["laps"]
    if len(laps) > _MAX_EVENT_ROWS:
        raise _Unavailable("resource_limit", f"An event exceeds {_MAX_EVENT_ROWS} eligible laps.")
    stint_keys = sorted(
        {(lap["driver_key"], lap["stint"]) for lap in laps},
        key=lambda key: (_driver_sort(key[0][1]), key[1]),
    )
    lap_numbers = sorted({lap["lap_number"] for lap in laps})
    driver_keys = sorted({lap["driver_key"] for lap in laps}, key=lambda key: _driver_sort(key[1]))
    column_count = len(stint_keys) + len(lap_numbers) + (len(driver_keys) if driver_trend else 0)
    if len(laps) * column_count > _MAX_EVENT_DESIGN_CELLS:
        raise _Unavailable(
            "resource_limit",
            "The event fixed-effect design exceeds the bounded dense-matrix allocation.",
        )

    stint_index = {key: index for index, key in enumerate(stint_keys)}
    lap_index = {number: index for index, number in enumerate(lap_numbers)}
    driver_index = {key: index for index, key in enumerate(driver_keys)}
    row_indices = np.arange(len(laps), dtype=np.int64)
    lap_values = np.asarray([lap["lap_number"] for lap in laps], dtype=np.float64)
    y = np.asarray([lap["duration_seconds"] for lap in laps], dtype=np.float64)
    centered_lap = lap_values - float(np.mean(lap_values))
    compounds = [lap["compound"] for lap in laps]
    x = np.column_stack(
        (
            np.asarray([compound == "SOFT" for compound in compounds], dtype=np.float64)
            * centered_lap,
            np.asarray([compound == "MEDIUM" for compound in compounds], dtype=np.float64)
            * centered_lap,
        )
    )
    nuisance = np.zeros((len(laps), column_count), dtype=np.float64)
    stint_codes = np.fromiter(
        (stint_index[(lap["driver_key"], lap["stint"])] for lap in laps),
        dtype=np.int64,
        count=len(laps),
    )
    lap_codes = np.fromiter(
        (lap_index[lap["lap_number"]] for lap in laps), dtype=np.int64, count=len(laps)
    )
    nuisance[row_indices, stint_codes] = 1.0
    nuisance[row_indices, len(stint_keys) + lap_codes] = 1.0
    if driver_trend:
        driver_codes = np.fromiter(
            (driver_index[lap["driver_key"]] for lap in laps), dtype=np.int64, count=len(laps)
        )
        driver_means = np.asarray(
            [np.mean(centered_lap[driver_codes == code]) for code in range(len(driver_keys))],
            dtype=np.float64,
        )
        nuisance[row_indices, len(stint_keys) + len(lap_numbers) + driver_codes] = (
            centered_lap - driver_means[driver_codes]
        )
    try:
        residualized, _, rank, _ = np.linalg.lstsq(nuisance, np.column_stack((y, x)), rcond=None)
        residuals = np.column_stack((y, x)) - nuisance @ residualized
    except np.linalg.LinAlgError as exc:
        raise _Unavailable(
            "numerical_failure", "The event fixed effects did not converge."
        ) from exc
    if not np.all(np.isfinite(residuals)):
        raise _Unavailable(
            "numerical_failure", "Fixed-effect residualization produced non-finite values."
        )
    return residuals[:, 0], residuals[:, 1:], x, int(rank)


__all__ = ["RelativeTyreWearInputError", "evaluate_relative_tyre_wear"]
