"""Verify the sealed normalized published-grid coverage without live refetching."""

import hashlib
import json
from pathlib import Path

from f1sim.simulation.execution import validate_pit_lane_starters, validate_starting_grid


def verify(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {key: value for key, value in record.items() if key != "content_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    if digest.hexdigest() != record.get("content_sha256"):
        raise ValueError("Published-grid coverage seal does not match")
    if (record.get("year") != 2026 or record.get("prospectively_recorded") is not False
            or record.get("independent_untouched_test") is not False
            or record.get("grid_sources_fetched_after_races") is not True):
        raise ValueError("Grid coverage is retrospective current-season source evidence")
    events = record["events"]
    if [e["round"] for e in events] != list(range(1, 17)):
        raise ValueError("Published-grid cohort is incomplete or duplicated")
    for event in events:
        ids = [d["id"] for d in event["entrants"]]
        if len(ids) != 22 or len(set(ids)) != 22:
            raise ValueError("Grid coverage must retain the complete known entrant field")
        grid = validate_starting_grid(event["starting_grid"], ids)
        validate_pit_lane_starters(event["pit_lane_starters"], grid)
        if (event["year"] != record["year"]
                or event["qualifying_observation_round"] != event["round"]
                or event["target_race_performance_used"] is not False
                or any(type(n) is not int or not 0 < n < event["round"]
                       for n in event["earlier_race_performance_rounds"])):
            raise ValueError("Grid coverage has inconsistent event or performance boundaries")
        if (not event["source_url"].startswith("https://www.formula1.com/en/results/2026/races/")
                or not event["source_url"].endswith("/starting-grid")
                or len(event["source_document_sha256"]) != 64):
            raise ValueError("Published-grid source provenance is incomplete")
    summary = {"events": len(events), "complete_fields": len(events),
               "pit_lane_events": sum(bool(e["pit_lane_starters"]) for e in events),
               "pit_lane_starters": sum(len(e["pit_lane_starters"]) for e in events)}
    if record["summary"] != summary:
        raise ValueError("Published-grid coverage summary does not match its event rows")
    return summary


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "evidence/published-grid-coverage-2026.json"
    print(json.dumps(verify(path), indent=2))
