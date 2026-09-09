"""Pure evidence summaries of reported race distances, stints, and pit entries.

Coverage describes reported laps, never feed completeness or tyre durability.
Pit entries are not interpreted as paid tyre changes or matched to stints.
"""

import json
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from math import isfinite
from numbers import Integral, Real

KNOWN_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}


def _integer(value, minimum=0):
    return not isinstance(value, bool) and isinstance(value, Integral) and value >= minimum


def _feed(rows, name, session):
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{name} must be a list of mappings")
    unique = {}
    for row in rows:
        if not _integer(row.get("session_key"), 1) or row["session_key"] != session:
            raise ValueError(f"{name} contains an invalid or different session_key")
        key = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
        unique[key] = row
    return [unique[key] for key in sorted(unique)], {
        "available": bool(rows), "input_rows": len(rows), "unique_rows": len(unique),
        "exact_duplicate_rows": len(rows) - len(unique),
    }


def _identity_rows(rows, fields, name):
    seen = {}
    for row in rows:
        identity = tuple(row[field] for field in fields)
        if identity in seen:
            raise ValueError(f"conflicting {name} identity: {identity}")
        seen[identity] = row


def _coverage(ranges, distance):
    # Sweep interval endpoints; memory is proportional to rows, not lap values.
    changes = Counter()
    outside = 0
    for start, end in ranges:
        outside += max(0, end - max(distance, start - 1))
        end = min(end, distance)
        if start <= end:
            changes[start] += 1
            changes[end + 1] -= 1
    covered = overlap = 0
    active = 0
    previous = 1
    for lap, change in sorted(changes.items()):
        width = lap - previous
        covered += width if active else 0
        overlap += width if active > 1 else 0
        active += change
        previous = lap
    return covered, overlap, outside


def _duration(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        return float(value) if isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def observed_strategy_summary(result_rows, stint_rows, pit_rows, session_key):
    """Summarize one session without inferring unreported stops or stint endings.

    Invalid ancillary identities/ranges are counted and excluded. Conflicting
    valid identities and wrong sessions fail rather than selecting one record.
    Empty ranges (end = start - 1) report no completed laps; they do not
    establish that a tyre was unused or fresh.
    Outside laps count reported exposures (overlapping outside ranges may count
    twice); overlap_laps counts distinct multiply-covered result laps.
    """
    if not _integer(session_key, 1):
        raise ValueError("session_key must be a positive integer")
    results, result_feed = _feed(result_rows, "results", session_key)
    stints, stint_feed = _feed(stint_rows, "stints", session_key)
    pits, pit_feed = _feed(pit_rows, "pits", session_key)
    if not results:
        raise ValueError("results must be nonempty")
    for row in results:
        if not _integer(row.get("driver_number"), 1):
            raise ValueError("result driver_number must be a positive integer")
        if not _integer(row.get("number_of_laps")):
            raise ValueError("result number_of_laps must be a nonnegative integer")
        if any(row.get(flag) is not None and not isinstance(row[flag], bool)
               for flag in ("dnf", "dns", "dsq")):
            raise ValueError("result status flags must be boolean or null")
    _identity_rows(results, ("driver_number",), "result")
    drivers = {}
    for row in results:
        drivers[row["driver_number"]] = {
            "driver_number": int(row["driver_number"]),
            "completed_laps": int(row["number_of_laps"]),
            **{flag: row.get(flag) for flag in ("dnf", "dns", "dsq")},
            "stints": [], "invalid_range_rows": 0, "empty_range_rows": 0,
            "invalid_stint_identity_rows": 0, "unknown_compound_rows": 0,
            "invalid_age_rows": 0, "pit_entries": [],
        }
    exclusions = {key: 0 for key in (
        "stint_invalid_identity_rows", "stint_unknown_driver_rows", "stint_invalid_range_rows",
        "stint_empty_range_rows", "stint_unknown_compound_rows", "stint_invalid_age_rows",
        "pit_invalid_identity_rows", "pit_unknown_driver_rows", "pit_invalid_lap_rows",
        "pit_invalid_lane_duration_rows", "pit_invalid_stop_duration_rows",
        "pit_deprecated_duration_rows",
    )}
    valid_stints = []
    for row in stints:
        number = row.get("driver_number")
        if not _integer(number, 1) or not _integer(row.get("stint_number"), 1):
            exclusions["stint_invalid_identity_rows"] += 1
            if _integer(number, 1) and number in drivers:
                drivers[number]["invalid_stint_identity_rows"] += 1
            continue
        valid_stints.append(row)
    _identity_rows(valid_stints, ("driver_number", "stint_number"), "stint")
    for row in valid_stints:
        if row["driver_number"] not in drivers:
            exclusions["stint_unknown_driver_rows"] += 1
            continue
        driver = drivers[row["driver_number"]]
        start, end = row.get("lap_start"), row.get("lap_end")
        if _integer(start, 1) and _integer(end) and end == start - 1:
            driver["empty_range_rows"] += 1
            exclusions["stint_empty_range_rows"] += 1
            continue
        if not _integer(start, 1) or not _integer(end, 1) or end < start:
            driver["invalid_range_rows"] += 1
            exclusions["stint_invalid_range_rows"] += 1
            continue
        raw = row.get("compound")
        compound = raw.upper() if isinstance(raw, str) and raw.upper() in KNOWN_COMPOUNDS else None
        if compound is None:
            driver["unknown_compound_rows"] += 1
            exclusions["stint_unknown_compound_rows"] += 1
        age = row.get("tyre_age_at_start")
        if age is not None and not _integer(age):
            driver["invalid_age_rows"] += 1
            exclusions["stint_invalid_age_rows"] += 1
            age = None
        driver["stints"].append({
            "stint_number": int(row["stint_number"]), "lap_start": int(start),
            "lap_end": int(end), "reported_laps": int(end - start + 1),
            "compound": compound, "raw_compound": raw if isinstance(raw, str) else None,
            "tyre_age_at_start": int(age) if age is not None else None, "end_reason": "unknown",
        })
    valid_pits = []
    for row in pits:
        try:
            date = datetime.fromisoformat(row.get("date", "").replace("Z", "+00:00"))
            valid = date.tzinfo is not None and _integer(row.get("driver_number"), 1)
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            exclusions["pit_invalid_identity_rows"] += 1
            continue
        normalized = dict(row, date=date.astimezone(timezone.utc).isoformat())
        valid_pits.append(normalized)
    # Equivalent instants are semantic duplicates, distinct from byte/value
    # duplicates already counted in feed metadata before normalization.
    normalized_unique = {}
    for row in valid_pits:
        key = json.dumps(row, sort_keys=True, separators=(",", ":"))
        normalized_unique[key] = row
    pit_feed["equivalent_instant_duplicate_rows"] = len(valid_pits) - len(normalized_unique)
    valid_pits = [normalized_unique[key] for key in sorted(normalized_unique)]
    _identity_rows(valid_pits, ("driver_number", "date"), "pit")
    for row in valid_pits:
        if row["driver_number"] not in drivers:
            exclusions["pit_unknown_driver_rows"] += 1
            continue
        entry = {"date": row["date"]}
        lap = row.get("lap_number")
        if not _integer(lap, 1):
            exclusions["pit_invalid_lap_rows"] += 1
            lap = None
        entry["lap_number"] = int(lap) if lap is not None else None
        for key in ("lane_duration", "stop_duration"):
            value = _duration(row.get(key))
            if row.get(key) is not None and value is None:
                exclusions[f"pit_invalid_{key}_rows"] += 1
            entry[key] = value
        exclusions["pit_deprecated_duration_rows"] += int("pit_duration" in row)
        drivers[row["driver_number"]]["pit_entries"].append(entry)
    totals = Counter()
    for driver in drivers.values():
        driver["stints"].sort(key=lambda s: (s["lap_start"], s["stint_number"]))
        driver["pit_entries"].sort(key=lambda p: p["date"])
        ranges = [(s["lap_start"], s["lap_end"]) for s in driver["stints"]]
        covered, overlap, outside = _coverage(ranges, driver["completed_laps"])
        known, _, _ = _coverage([(s["lap_start"], s["lap_end"]) for s in driver["stints"]
                                 if s["compound"] is not None], driver["completed_laps"])
        complete = (bool(driver["completed_laps"]) and covered == driver["completed_laps"]
                    and not (overlap or outside or driver["invalid_range_rows"]
                             or driver["invalid_stint_identity_rows"]
                             or driver["unknown_compound_rows"]))
        if not driver["completed_laps"]:
            complete = None
        driver["coverage"] = {
            "covered_laps": covered, "known_compound_laps": known,
            "missing_laps": driver["completed_laps"] - covered,
            "overlap_laps": overlap, "outside_result_lap_exposures": outside,
            "complete_lap_coverage": complete,
        }
        driver["pit_evidence"] = {"entries": len(driver["pit_entries"]),
                                  "lane_duration_count": sum(p["lane_duration"] is not None
                                                             for p in driver["pit_entries"]),
                                  "stop_duration_count": sum(p["stop_duration"] is not None
                                                             for p in driver["pit_entries"])}
        totals["result_drivers"] += 1
        totals["positive_lap_drivers"] += int(driver["completed_laps"] > 0)
        totals["completed_laps"] += driver["completed_laps"]
        totals["complete_lap_coverage_drivers"] += int(complete is True)
        for key, value in driver["coverage"].items():
            if key != "complete_lap_coverage":
                totals[key] += value
        for key, value in driver["pit_evidence"].items():
            totals[f"pit_{key}"] += value
        for flag in ("dnf", "dns", "dsq"):
            totals[f"{flag}_true_drivers"] += int(driver[flag] is True)
            totals[f"{flag}_unknown_drivers"] += int(driver[flag] is None)
    return {"session_key": int(session_key),
            "feeds": {"results": result_feed, "stints": stint_feed, "pits": pit_feed},
            "exclusions": exclusions, "totals": dict(sorted(totals.items())),
            "drivers": [drivers[number] for number in sorted(drivers)]}
