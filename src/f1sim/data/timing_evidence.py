"""Normalize official live-timing archive deltas into lap evidence.

The archive feeds are intentionally treated as observations rather than as a
complete race model.  In particular, a value that happens to be present in a
later delta is never used to fill an earlier delta.  This keeps the adapter
useful for auditing a downloaded archive without turning missing information
into synthetic precision.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

_FEED_NAMES = ("TimingData", "TimingAppData", "TrackStatus", "WeatherData", "SessionStatus")
_TIME_RE = re.compile(
    r"^(?P<hours>\d{2}):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)\.(?P<millis>\d{3})"
)
_DURATION_RE = re.compile(r"^(?P<minutes>\d+):(?P<seconds>[0-5]\d)\.(?P<millis>\d{3})$")
_MISSING = object()
_AMBIGUOUS = object()


class TimingEvidenceError(ValueError):
    """Raised when an archive stream cannot be parsed safely."""


@dataclass(frozen=True)
class _Row:
    seconds: float
    display_time: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class _Crossing:
    seconds: float
    display_time: str


@dataclass
class _DurationObservation:
    duration: float | None
    seconds: float
    display_time: str
    invalid: bool = False
    stale: bool = False
    conflict: bool = False
    duplicate_count: int = 0


@dataclass
class _DriverTiming:
    crossings: dict[int, _Crossing] = field(default_factory=dict)
    durations: dict[int, _DurationObservation] = field(default_factory=dict)
    count_issues: dict[int, set[str]] = field(default_factory=dict)
    pending_count_issues: set[str] = field(default_factory=set)
    previous_lap: int | None = None
    invalid_count_observations: int = 0
    unpaired_duration_observations: int = 0
    stale_duration_observations: int = 0


@dataclass(frozen=True)
class _StintState:
    index: int
    compound: str | None
    prior_wear: int | None


@dataclass
class _StintSnapshot:
    seconds: float
    display_time: str
    states: dict[int, dict[str, Any]]
    active_index: int | None


@dataclass(frozen=True)
class _SeriesEvent:
    seconds: float
    value: Any


def _lookup(mapping: Mapping[str, Any], name: str, default: Any = _MISSING) -> Any:
    """Look up one exact official feed field."""

    return mapping.get(name, default)


def _timestamp(value: Any, feed: str, line_number: int) -> tuple[float, str]:
    if not isinstance(value, str):
        raise TimingEvidenceError(
            f"feed {feed!r} line {line_number}: timestamp must be HH:MM:SS.mmm text"
        )
    text = value.strip()
    match = _TIME_RE.fullmatch(text)
    if match:
        hours = int(match.group("hours"))
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        millis = int(match.group("millis"))
        total = hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0
        return total, text
    raise TimingEvidenceError(
        f"feed {feed!r} line {line_number}: invalid timestamp {text!r}; "
        "expected HH:MM:SS.mmm"
    )


def _parse_stream(feed: str, text: str | bytes) -> list[_Row]:
    if isinstance(text, bytes):
        try:
            decoded = text.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise TimingEvidenceError(f"feed {feed!r}: stream is not valid UTF-8") from exc
    elif isinstance(text, str):
        decoded = text.removeprefix("\ufeff")
    else:
        raise TypeError(f"feed {feed!r}: expected UTF-8 text or bytes")

    rows: list[_Row] = []
    previous_seconds: float | None = None
    for line_number, raw_line in enumerate(decoded.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        line = line.removeprefix("\ufeff")
        match = _TIME_RE.match(line)
        if match:
            display_time = match.group(0)
            seconds, _ = _timestamp(display_time, feed, line_number)
            json_text = line[match.end() :].strip()
        else:
            raise TimingEvidenceError(
                f"feed {feed!r} line {line_number}: missing HH:MM:SS.mmm timestamp prefix"
            )
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise TimingEvidenceError(
                f"feed {feed!r} line {line_number}: malformed JSON ({exc.msg})"
            ) from exc
        if not isinstance(payload, Mapping):
            raise TimingEvidenceError(
                f"feed {feed!r} line {line_number}: JSON delta must be an object"
            )
        if previous_seconds is not None and seconds < previous_seconds:
            raise TimingEvidenceError(
                f"feed {feed!r} line {line_number}: timestamp {display_time!r} is earlier "
                f"than the preceding row"
            )
        previous_seconds = seconds
        rows.append(_Row(seconds, display_time, payload))
    return rows


def _value(mapping: Mapping[str, Any], name: str) -> Any:
    value = _lookup(mapping, name)
    if isinstance(value, Mapping):
        nested = _lookup(value, "Value")
        if nested is not _MISSING:
            return nested
    return value


def _parse_lap_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _last_lap_value(mapping: Mapping[str, Any]) -> Any:
    """Return only the official LastLapTime.Value field.

    TimingData also carries ``PersonalFastest`` and ``OverallFastest``
    metadata under LastLapTime.  Those values describe records, not the
    completed lap attached to this delta, and must not become observations.
    """

    value = _lookup(mapping, "LastLapTime")
    if value is _MISSING:
        return _MISSING
    if isinstance(value, Mapping):
        return _lookup(value, "Value")
    return _MISSING


def _parse_duration(value: Any) -> float | None:
    """Parse the official ``M:SS.mmm`` lap duration representation."""

    if not isinstance(value, str):
        return None
    match = _DURATION_RE.fullmatch(value.strip())
    if not match:
        return None
    parsed = (
        int(match.group("minutes")) * 60.0
        + int(match.group("seconds"))
        + int(match.group("millis")) / 1000.0
    )
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _driver_lines(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    lines = _lookup(payload, "Lines")
    if isinstance(lines, Mapping):
        return lines
    return {}


def _record_timing(rows: Sequence[_Row]) -> tuple[dict[str, _DriverTiming], dict[str, Any]]:
    drivers: dict[str, _DriverTiming] = {}
    total_nonempty_duration = 0
    total_invalid_duration = 0
    total_invalid_count = 0
    total_unpaired = 0

    for row in rows:
        payload = row.payload
        lines = _driver_lines(payload)
        for raw_driver, raw_line in lines.items():
            if not isinstance(raw_line, Mapping):
                continue
            driver = str(raw_driver)
            state = drivers.setdefault(driver, _DriverTiming())
            count_value = _lookup(raw_line, "NumberOfLaps")
            count_present = count_value is not _MISSING
            lap_count = (
                _parse_lap_count(_value(raw_line, "NumberOfLaps"))
                if count_present
                else None
            )
            if count_present and lap_count is None:
                state.invalid_count_observations += 1
                total_invalid_count += 1

            last_lap = _lookup(raw_line, "LastLapTime")
            last_value = _last_lap_value(raw_line) if last_lap is not _MISSING else _MISSING
            has_duration_value = (
                last_value is not _MISSING
                and last_value is not None
                and str(last_value).strip() != ""
            )
            if has_duration_value:
                total_nonempty_duration += 1

            if lap_count is not None:
                previous_lap = state.previous_lap
                current_issues: set[str] = set()
                if previous_lap is not None:
                    if lap_count < previous_lap:
                        if lap_count in state.crossings:
                            state.pending_count_issues.add("lap_count_regression")
                        else:
                            current_issues.add("lap_count_regression")
                    elif lap_count > previous_lap + 1:
                        current_issues.add("nonconsecutive_lap_count")
                state.previous_lap = lap_count
                if lap_count > 0 and lap_count not in state.crossings:
                    current_issues.update(state.pending_count_issues)
                    state.pending_count_issues.clear()
                    if current_issues:
                        state.count_issues[lap_count] = current_issues
                    state.crossings[lap_count] = _Crossing(row.seconds, row.display_time)

            if not has_duration_value:
                continue
            if lap_count is None:
                state.unpaired_duration_observations += 1
                total_unpaired += 1
                continue
            duration = _parse_duration(last_value)
            invalid = duration is None
            if invalid:
                total_invalid_duration += 1
            existing = state.durations.get(lap_count)
            if existing is None:
                state.durations[lap_count] = _DurationObservation(
                    duration, row.seconds, row.display_time, invalid=invalid
                )
                continue
            if duration is not None and existing.duration == duration:
                existing.duplicate_count += 1
                continue
            if duration is None and existing.invalid:
                existing.duplicate_count += 1
                continue
            existing.conflict = True

    counts = {
        "nonempty_duration_observations": total_nonempty_duration,
        "invalid_duration_observations": total_invalid_duration,
        "invalid_lap_count_observations": total_invalid_count,
        "unpaired_duration_observations": total_unpaired,
        "stale_duration_observations": 0,
    }
    return drivers, counts


def _normalise_compound(value: Any) -> str | None:
    if value is None or value is _MISSING:
        return None
    text = re.sub(r"[^A-Z0-9]+", "", str(value).upper())
    aliases = {
        "S": "SOFT",
        "SOFT": "SOFT",
        "M": "MEDIUM",
        "MED": "MEDIUM",
        "MEDIUM": "MEDIUM",
        "H": "HARD",
        "HARD": "HARD",
        "HS": "HYPERSOFT",
        "HYPERSOFT": "HYPERSOFT",
        "US": "ULTRASOFT",
        "ULTRASOFT": "ULTRASOFT",
        "SH": "SUPERHARD",
        "SUPERHARD": "SUPERHARD",
        "I": "INTERMEDIATE",
        "INTER": "INTERMEDIATE",
        "INTERMEDIATE": "INTERMEDIATE",
        "W": "WET",
        "WET": "WET",
        "FULLWET": "WET",
    }
    return aliases.get(text)


def _start_laps(value: Any) -> int | None:
    parsed = _parse_lap_count(value)
    return parsed


def _merge_stints(states: dict[int, dict[str, Any]], value: Any) -> set[int]:
    changed: set[int] = set()
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                target = states.setdefault(index, {})
                target.update({str(key): val for key, val in item.items()})
                changed.add(index)
        return changed
    if not isinstance(value, Mapping):
        return changed
    if any(str(key).casefold() in {"compound", "new", "startlaps", "totallaps"} for key in value):
        target = states.setdefault(0, {})
        target.update({str(key): val for key, val in value.items()})
        changed.add(0)
        return changed
    for raw_index, item in value.items():
        try:
            index = int(str(raw_index))
        except (TypeError, ValueError):
            continue
        if index < 0 or not isinstance(item, Mapping):
            continue
        target = states.setdefault(index, {})
        target.update({str(key): val for key, val in item.items()})
        changed.add(index)
    return changed


def _record_stints(rows: Sequence[_Row]) -> dict[str, list[_StintSnapshot]]:
    snapshots: dict[str, list[_StintSnapshot]] = {}
    states_by_driver: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        payload = row.payload
        lines = _driver_lines(payload)
        for raw_driver, raw_line in lines.items():
            if not isinstance(raw_line, Mapping):
                continue
            value = _lookup(raw_line, "Stints")
            if value is _MISSING:
                value = _lookup(raw_line, "TyreStints")
            if value is _MISSING:
                continue
            driver = str(raw_driver)
            states = states_by_driver.setdefault(driver, {})
            _merge_stints(states, value)
            active_index = max(states) if states else None
            snapshots.setdefault(driver, []).append(
                _StintSnapshot(
                    row.seconds,
                    row.display_time,
                    {index: dict(item) for index, item in states.items()},
                    active_index,
                )
            )
    return snapshots


def _stint_at(
    snapshots: Sequence[_StintSnapshot], seconds: float
) -> _StintState | None:
    selected: _StintSnapshot | None = None
    for snapshot in snapshots:
        if snapshot.seconds <= seconds:
            selected = snapshot
        else:
            break
    if selected is None or selected.active_index is None:
        return None
    raw = selected.states.get(selected.active_index)
    if raw is None:
        return None
    compound = _normalise_compound(_lookup(raw, "Compound"))
    prior_wear = _start_laps(_lookup(raw, "StartLaps"))
    return _StintState(selected.active_index, compound, prior_wear)


def _series_events(rows: Sequence[_Row], feed_name: str) -> list[_SeriesEvent]:
    events: list[_SeriesEvent] = []
    for row in rows:
        payload = row.payload
        if feed_name == "TrackStatus":
            value = _value(payload, "Status")
            if value is _MISSING:
                # Feed deltas omit unchanged fields.  An omitted Status does
                # not erase the last known status.
                continue
            value = str(value).strip() if value is not None else None
        else:
            value = _value(payload, "Rainfall")
            if value is _MISSING:
                # WeatherData is also a delta stream; retain the prior
                # Rainfall value through metadata-only updates.
                continue
            if value is not None:
                value = _rainfall(value)
        events.append(_SeriesEvent(row.seconds, value))
    return events


def _rainfall(value: Any) -> bool | object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if value in (0, 1):
            return bool(value)
        return _AMBIGUOUS
    text = str(value).strip().casefold()
    if text in {"0", "0.0"}:
        return False
    if text in {"1", "1.0"}:
        return True
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return _AMBIGUOUS
    if math.isfinite(numeric) and numeric in (0, 1):
        return bool(numeric)
    return _AMBIGUOUS


def _series_segments(
    events: Sequence[_SeriesEvent], start: float, end: float
) -> list[tuple[str, Any]]:
    """Return every piecewise-constant state covering ``[start, end)``."""

    if start >= end or not events:
        return [("unknown", None)]
    ordered = sorted(events, key=lambda item: item.seconds)
    groups: list[tuple[float, Any]] = []
    index = 0
    while index < len(ordered):
        event_time = ordered[index].seconds
        values: list[Any] = []
        while index < len(ordered) and ordered[index].seconds == event_time:
            values.append(ordered[index].value)
            index += 1
        first = values[0]
        value = first if all(item == first for item in values[1:]) else _AMBIGUOUS
        groups.append((event_time, value))

    at_start: Any = _MISSING
    for event_time, value in groups:
        if event_time > start:
            break
        at_start = value
    if at_start is _MISSING:
        return [("unknown", None)]

    segments: list[tuple[str, Any]] = []
    current = at_start
    for event_time, value in groups:
        if event_time <= start:
            continue
        if event_time >= end:
            break
        if current is _AMBIGUOUS:
            segments.append(("ambiguous", None))
        elif current is None or current is _MISSING:
            segments.append(("unknown", None))
        else:
            segments.append(("known", current))
        if value is _AMBIGUOUS:
            current = _AMBIGUOUS
        else:
            current = value
    if current is _AMBIGUOUS:
        segments.append(("ambiguous", None))
    elif current is None or current is _MISSING:
        segments.append(("unknown", None))
    else:
        segments.append(("known", current))
    return segments


def _status_result(events: Sequence[_SeriesEvent], start: float, end: float) -> str:
    segments = _series_segments(events, start, end)
    if any(state == "ambiguous" for state, _ in segments):
        return "ambiguous"
    if any(state == "unknown" for state, _ in segments):
        return "unknown"
    return "green" if all(str(value).strip() == "1" for _, value in segments) else "non_green"


def _record_pit_events(
    rows: Sequence[_Row],
) -> dict[str, list[tuple[float, bool | None, bool | None, bool, bool]]]:
    events_by_driver: dict[
        str, list[tuple[float, bool | None, bool | None, bool, bool]]
    ] = {}
    for row in rows:
        payload = row.payload
        for raw_driver, raw_line in _driver_lines(payload).items():
            if not isinstance(raw_line, Mapping):
                continue
            in_pit_raw = _lookup(raw_line, "InPit")
            pit_out_raw = _lookup(raw_line, "PitOut")
            in_pit_present = in_pit_raw is not _MISSING
            pit_out_present = pit_out_raw is not _MISSING
            in_pit = _parse_bool(_value(raw_line, "InPit")) if in_pit_present else None
            pit_out = (
                _parse_bool(_value(raw_line, "PitOut"))
                if pit_out_present
                else None
            )
            if in_pit_present or pit_out_present:
                events_by_driver.setdefault(str(raw_driver), []).append(
                    (row.seconds, in_pit, pit_out, in_pit_present, pit_out_present)
                )
    return events_by_driver


def _pit_affected(
    events_by_driver: Mapping[
        str, Sequence[tuple[float, bool | None, bool | None, bool, bool]]
    ],
    driver: str,
    start: float,
    end: float,
) -> tuple[bool, bool, bool]:
    events = events_by_driver.get(driver, ())
    if not events or start >= end:
        return False, False, True
    ordered = sorted(events, key=lambda item: item[0])
    in_pit: bool | None = None
    pit_out: bool | object = _MISSING
    ambiguous = False
    affected = False
    known_at_start = False
    unknown_segment = False
    last_in_time: float | None = None
    last_in_value: bool | None = None
    last_out_time: float | None = None
    last_out_value: bool | None | object = _MISSING

    def apply_pit_out(seconds: float, value: bool | None) -> None:
        nonlocal ambiguous, last_out_time, last_out_value, pit_out
        if (
            value is not None
            and last_out_time == seconds
            and last_out_value not in {_MISSING, None, _AMBIGUOUS}
            and last_out_value != value
        ):
            ambiguous = True
            pit_out = _AMBIGUOUS
        elif value is None:
            pit_out = None
        else:
            pit_out = value
        last_out_time = seconds
        last_out_value = value

    for seconds, in_value, out_value, in_present, out_present in ordered:
        if seconds > end:
            break
        at_interval_start = seconds == start
        in_interval = start < seconds < end
        if seconds <= start and in_present:
            if (
                in_value is not None
                and last_in_time == seconds
                and last_in_value is not None
                and last_in_value != in_value
            ):
                ambiguous = True
            in_pit = in_value
            if in_value is not None:
                known_at_start = True
            elif at_interval_start:
                unknown_segment = True
            last_in_time = seconds
            last_in_value = in_value
        elif in_interval and in_present:
            if in_pit is True:
                affected = True
            if in_value:
                affected = True
            in_pit = in_value
            if in_value is None:
                unknown_segment = True
            if (
                in_value is not None
                and last_in_time == seconds
                and last_in_value is not None
                and last_in_value != in_value
            ):
                ambiguous = True
            last_in_time = seconds
            last_in_value = in_value
        if (seconds <= start or in_interval) and out_present:
            apply_pit_out(seconds, out_value)
        if (at_interval_start or in_interval) and out_present:
            if out_value is True:
                affected = True
            elif out_value is None:
                unknown_segment = True
    if in_pit is True:
        affected = True
    if pit_out is True:
        affected = True
    if pit_out is _AMBIGUOUS:
        ambiguous = True
    elif pit_out is None:
        unknown_segment = True
    return affected, ambiguous, not known_at_start or in_pit is None or unknown_segment


def _driver_number(driver: str) -> int | str:
    return int(driver) if re.fullmatch(r"\d+", driver) else driver


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _session_start(rows: Sequence[_Row]) -> float | None:
    started: list[float] = []
    for row in rows:
        payload = row.payload
        status = _value(payload, "Status")
        if status is _MISSING:
            continue
        if str(status).strip().casefold() == "started":
            started.append(row.seconds)
    return min(started) if started else None


def normalize_timing_evidence(feeds: Mapping[str, str | bytes]) -> dict[str, Any]:
    """Normalize official timing archive streams into JSON-serializable evidence.

    ``feeds`` maps the official feed names to newline-delimited JSON streams.
    Each row begins with an ``HH:MM:SS.mmm`` prefix immediately followed by
    its JSON delta.  Unknown data is represented by explicit exclusions and
    never silently filled from a later update.
    """

    if not isinstance(feeds, Mapping):
        raise TypeError("feeds must be a mapping of feed name to UTF-8 JSON stream")
    unknown_feeds = [name for name in feeds if name not in _FEED_NAMES]
    if unknown_feeds:
        raise TimingEvidenceError(
            f"unsupported feed(s): {', '.join(repr(str(name)) for name in unknown_feeds)}"
        )
    parsed: dict[str, list[_Row]] = {
        feed: _parse_stream(feed, feeds[feed]) for feed in _FEED_NAMES if feed in feeds
    }

    timing_rows = parsed.get("TimingData", [])
    app_rows = parsed.get("TimingAppData", [])
    track_rows = parsed.get("TrackStatus", [])
    weather_rows = parsed.get("WeatherData", [])
    session_rows = parsed.get("SessionStatus", [])
    drivers, data_quality = _record_timing(timing_rows)
    pit_events = _record_pit_events(timing_rows)
    stint_snapshots = _record_stints(app_rows)
    track_events = _series_events(track_rows, "TrackStatus")
    weather_events = _series_events(weather_rows, "WeatherData")
    session_start = _session_start(session_rows)

    laps: list[dict[str, Any]] = []
    duplicate_identical = 0
    conflicting_duplicates = 0
    for driver, state in sorted(drivers.items()):
        snapshots = stint_snapshots.get(driver, [])
        for lap_number, crossing in sorted(state.crossings.items()):
            if lap_number <= 0:
                continue
            reasons: list[str] = []
            duration_observation = state.durations.get(lap_number)
            duration: float | None = None
            if duration_observation is None:
                _append_reason(reasons, "missing_duration")
            elif duration_observation.conflict:
                _append_reason(reasons, "ambiguous_duplicate_duration")
                duration_observation.conflict = True
                conflicting_duplicates += 1
            elif duration_observation.invalid:
                _append_reason(reasons, "invalid_duration")
            elif duration_observation.seconds != crossing.seconds:
                _append_reason(reasons, "stale_duration")
                data_quality["stale_duration_observations"] += 1
            else:
                duration = duration_observation.duration
            if duration_observation is not None:
                duplicate_identical += duration_observation.duplicate_count

            previous = state.crossings.get(lap_number - 1)
            if previous is None:
                _append_reason(reasons, "missing_preceding_crossing")
                observed_start = None
            else:
                observed_start = previous.display_time
                if previous.seconds >= crossing.seconds:
                    _append_reason(reasons, "invalid_crossing_interval")
            observed_end = crossing.display_time
            if lap_number in state.count_issues:
                _append_reason(reasons, "nonconsecutive_or_regressed_lap_count")
            if session_start is None:
                _append_reason(reasons, "session_start_unknown")
            elif crossing.seconds < session_start or (
                previous is not None and previous.seconds < session_start
            ):
                _append_reason(reasons, "before_session_start")

            start_seconds = previous.seconds if previous is not None else None
            end_seconds = crossing.seconds
            start_stint = _stint_at(snapshots, start_seconds) if start_seconds is not None else None
            end_stint = _stint_at(snapshots, end_seconds)
            compound = end_stint.compound if end_stint is not None else None
            prior_wear = end_stint.prior_wear if end_stint is not None else None
            stint = end_stint.index if end_stint is not None else None
            if start_stint is None or end_stint is None:
                _append_reason(reasons, "missing_stint_compound")
            else:
                if start_stint.index != end_stint.index:
                    _append_reason(reasons, "stint_change_mid_lap")
                if start_stint.compound is None or end_stint.compound is None:
                    _append_reason(reasons, "missing_stint_compound")
                elif start_stint.compound != end_stint.compound:
                    _append_reason(reasons, "compound_change_mid_lap")
                if end_stint.compound in {"INTERMEDIATE", "WET"}:
                    _append_reason(reasons, "non_slick_compound")

            if start_seconds is None:
                track_result = "unknown"
                weather_result = "unknown"
                pit_affected, pit_ambiguous, pit_unknown = False, False, True
            else:
                track_result = _status_result(track_events, start_seconds, end_seconds)
                weather_segments = _series_segments(weather_events, start_seconds, end_seconds)
                if any(state == "ambiguous" for state, _ in weather_segments):
                    weather_result = "ambiguous"
                elif any(state == "unknown" for state, _ in weather_segments):
                    weather_result = "unknown"
                elif any(value for _, value in weather_segments):
                    weather_result = "wet"
                else:
                    weather_result = "dry"
                pit_affected, pit_ambiguous, pit_unknown = _pit_affected(
                    pit_events, driver, start_seconds, end_seconds
                )
            if track_result == "unknown":
                _append_reason(reasons, "track_status_unknown")
            elif track_result == "ambiguous":
                _append_reason(reasons, "track_status_ambiguous")
            elif track_result != "green":
                _append_reason(reasons, "track_status_not_green")
            if weather_result == "unknown":
                _append_reason(reasons, "weather_rainfall_unknown")
            elif weather_result == "ambiguous":
                _append_reason(reasons, "weather_rainfall_ambiguous")
            elif weather_result == "wet":
                _append_reason(reasons, "rainfall_reported")
            if pit_affected:
                _append_reason(reasons, "pit_affected")
            if pit_ambiguous:
                _append_reason(reasons, "pit_observation_ambiguous")
            if pit_unknown:
                _append_reason(reasons, "pit_status_unknown")

            eligible = not reasons
            laps.append(
                {
                    "driver_number": _driver_number(driver),
                    "lap_number": lap_number,
                    "duration_seconds": duration,
                    "reported_at": observed_end,
                    "observed_start": observed_start,
                    "observed_end": observed_end,
                    "stint": stint,
                    "compound": compound,
                    "prior_wear": prior_wear,
                    "start_stint": start_stint.index if start_stint is not None else None,
                    "end_stint": end_stint.index if end_stint is not None else None,
                    "start_compound": start_stint.compound if start_stint is not None else None,
                    "end_compound": end_stint.compound if end_stint is not None else None,
                    "eligible": eligible,
                    "exclusions": reasons,
                }
            )

    laps.sort(key=lambda item: (str(item["driver_number"]), item["lap_number"]))
    reason_counts: dict[str, int] = {}
    for lap in laps:
        for reason in lap["exclusions"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary: dict[str, Any] = {
        "total_laps": len(laps),
        "eligible_laps": sum(1 for lap in laps if lap["eligible"]),
        "duration_observations": sum(1 for lap in laps if lap["duration_seconds"] is not None),
        "missing_duration": reason_counts.get("missing_duration", 0),
        "stale_duration": reason_counts.get("stale_duration", 0),
        "ambiguous_laps": sum(
            1
            for lap in laps
            if any("ambiguous" in reason or "duplicate" in reason for reason in lap["exclusions"])
        ),
        "duplicate_identical_observations": duplicate_identical,
        "conflicting_duplicate_observations": conflicting_duplicates,
        "missing_preceding_crossing": reason_counts.get("missing_preceding_crossing", 0),
        "unpaired_duration_observations": data_quality["unpaired_duration_observations"],
        "nonempty_unpaired_duration_observations": data_quality[
            "unpaired_duration_observations"
        ],
        "invalid_duration_observations": data_quality["invalid_duration_observations"],
        "invalid_lap_count_observations": data_quality["invalid_lap_count_observations"],
        "stale_duration_observations": data_quality["stale_duration_observations"],
        "exclusion_counts": reason_counts,
    }
    coverage: dict[str, Any] = {
        "session_started_at": (
            _display_seconds(session_start) if session_start is not None else None
        ),
        "feeds": {},
    }
    for feed in _FEED_NAMES:
        rows = parsed.get(feed, [])
        coverage["feeds"][feed] = {
            "present": feed in parsed,
            "rows": len(rows),
            "first": rows[0].display_time if rows else None,
            "last": rows[-1].display_time if rows else None,
        }
    return {
        "laps": laps,
        "summary": summary,
        "coverage": coverage,
        "limitations": [
            (
                "This is observational timing evidence; it makes no thermal-clean claim "
                "and estimates no causal warmup penalty."
            ),
            (
                "Surface wetness is not established by Rainfall=0; traffic, fuel load, "
                "energy deployment, deleted laps, and thermal state remain unmeasured."
            ),
            (
                "Only laps with explicit duration, consecutive crossing evidence, known "
                "unchanged stint compound, green status, zero reported rainfall, and no "
                "pit exposure are eligible."
            ),
        ],
    }


def _display_seconds(seconds: float) -> str:
    """Format relative seconds for coverage output."""

    if seconds < 0:
        return str(seconds)
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000.0))
    if millis == 1000:
        whole += 1
        millis = 0
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


__all__ = ["TimingEvidenceError", "normalize_timing_evidence"]
