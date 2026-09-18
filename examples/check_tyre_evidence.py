"""Inspect current-season official timing archives without fitting tyre physics.

Explicit network diagnostic; source responses stay in memory. The output is
observational evidence, not a measured warm-up penalty or a strategy optimum.
"""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone

from f1sim.data.current import CurrentSeasonDataLoader
from f1sim.data.timing_evidence import normalize_timing_evidence

BASE_URL = "https://livetiming.formula1.com/static/"
FEEDS = ("TimingData", "TimingAppData", "TrackStatus", "WeatherData", "SessionStatus")


def collect_archive_evidence(meeting, *, now=None, http_get=None):
    """Fetch one completed race in the current UTC season, with bounded reads."""
    now = now or datetime.now(timezone.utc)
    if now.utcoffset() is None:
        raise ValueError("Archive evidence requires a timezone-aware current date")
    year = now.astimezone(timezone.utc).year
    getter = http_get or CurrentSeasonDataLoader._default_http_get
    deadline = time.monotonic() + 60

    def fetch(path):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Archive evidence exceeded the 60-second fetch budget")
        raw = getter(BASE_URL + path, headers={"User-Agent": "f1sim-tyre-evidence"},
                     timeout=min(20, remaining))
        if time.monotonic() > deadline:
            raise RuntimeError("Archive evidence exceeded the 60-second fetch budget")
        return raw.decode("utf-8-sig")

    def fetch_object(path):
        value = json.loads(fetch(path))
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected an official archive object: {path}")
        return value

    index = fetch_object(f"{year}/Index.json")
    meetings = index.get("Meetings")
    if not isinstance(meetings, list):
        raise RuntimeError("Official season index is missing its meetings")
    query = meeting.strip().casefold()
    if not query:
        raise ValueError("Specify a meeting name")
    candidates = [row for row in meetings if isinstance(row, dict)
                  and query in str(row.get("Name", "")).casefold()]
    exact = [row for row in candidates if row.get("Name", "").casefold() == query]
    candidates = exact or candidates
    if len(candidates) != 1:
        raise ValueError(f"Meeting must uniquely match the {year} index: {meeting!r}")
    selected = candidates[0]
    available = selected.get("Sessions")
    if not isinstance(available, list) or any(not isinstance(row, dict) for row in available):
        raise RuntimeError("Official meeting has invalid session records")
    sessions = [row for row in available if row.get("Name") == "Race" and row.get("Path")]
    if len(sessions) != 1:
        raise RuntimeError("Meeting does not have one available race archive")
    path = sessions[0]["Path"]
    # Accept only relative archive directory names from this season. Never
    # interpret provider paths as hosts, queries, or parent-directory links.
    if not isinstance(path, str) or not re.fullmatch(
        rf"{year}/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/", path,
    ):
        raise RuntimeError("Official index returned an invalid current-season race path")
    info = fetch_object(path + "SessionInfo.json")
    archive_status = info.get("ArchiveStatus")
    try:
        session_year = datetime.fromisoformat(info["StartDate"]).year
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Race archive has no valid session start date") from exc
    if (info.get("Path") != path or info.get("Type") != "Race"
            or session_year != year
            or info.get("Name") != "Race" or info.get("SessionStatus") != "Finalised"
            or not isinstance(archive_status, dict) or archive_status.get("Status") != "Complete"
            or type(info.get("Key")) is not int or info["Key"] <= 0
            or (sessions[0].get("Key") is not None and sessions[0]["Key"] != info["Key"])):
        raise RuntimeError("Race archive is incomplete or its session identity does not match")
    feeds = {name: fetch(path + name + ".jsonStream") for name in FEEDS}
    report = normalize_timing_evidence(feeds)
    return {
        "source": BASE_URL + path,
        "source_feeds": list(FEEDS),
        "decoded_feed_sha256": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in feeds.items()
        },
        "season": year,
        "meeting": selected["Name"],
        "session_key": info["Key"],
        "retrieved_at": now.isoformat(),
        "evidence": report,
        "calibration": "No thermal or wear coefficients fitted; simulation presets unchanged.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting", required=True,
                        help="Unique name from the current season, e.g. Canadian")
    parser.add_argument("--include-laps", action="store_true",
                        help="Include individual normalized lap observations in JSON")
    args = parser.parse_args()
    try:
        report = collect_archive_evidence(args.meeting)
        if not args.include_laps:
            report["evidence"].pop("laps", None)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Could not collect tyre evidence: {exc}\n")


if __name__ == "__main__":
    main()
