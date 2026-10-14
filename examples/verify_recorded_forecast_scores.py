"""Recompute sealed recorded-forecast scores offline, without fetching or simulating.

The saved publication receipts were checked online when evidence was collected.
This verifier checks their timing and file digests; it cannot authenticate GitHub
or the observed result feed without fetching them again.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from f1sim.analysis.recorded_forecast import load_recorded_forecast, score_recorded_forecast
from f1sim.data import CurrentSeasonDataLoader

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/recorded-forecast-scores-2026-10-14.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Evidence timestamps must include a timezone")
    return parsed


class _OfflineLoader(CurrentSeasonDataLoader):
    def __init__(self, event, provenance):
        self.event = event
        self._saved_provenance = provenance

    def _event_for_race(self, year, race):
        if (year, race) != (self.event["year"], self.event["round"]):
            raise ValueError("Observed event must match the forecast")
        return self.event

    def get_provenance(self):
        return self._saved_provenance


def assess(evidence, *, root=ROOT):
    body = {key: value for key, value in evidence.items() if key != "content_sha256"}
    if evidence.get("format_version") != 1 or evidence.get("content_sha256") != digest(body):
        raise ValueError("Recorded-score evidence format or seal is invalid")
    if evidence["forecast_stage"] != "pre_qualifying":
        raise ValueError("These records must retain their pre-qualifying interpretation")
    event = evidence["observed_event"]
    if (event["year"], event["round"]) != (2026, 17):
        raise ValueError("Evidence must identify Singapore 2026 round 17")
    records = evidence["records"]
    expected = {"forecasts/2026-round-17-dry.json", "forecasts/2026-round-17-dry-clock.json",
                "forecasts/2026-round-17-dry-teammate-points.json"}
    if len(records) != 3 or {row["path"] for row in records} != expected:
        raise ValueError("Evidence must contain each of the three original forecasts once")
    results = []
    for saved in records:
        path = root / saved["path"]
        forecast = load_recorded_forecast(path)
        if forecast["schema_version"] not in (1, 2):
            raise ValueError("An after-qualifying forecast cannot replace these original records")
        if forecast["content_sha256"] != saved["canonical_content_sha256"]:
            raise ValueError("Original forecast canonical seal changed")
        data = path.read_bytes()
        lf = data.replace(b"\r\n", b"\n")
        variants = {data, lf, lf.replace(b"\n", b"\r\n")}
        receipt = saved["publication_receipt"]
        matching = [value for value in variants if hashlib.sha256(value).hexdigest()
                    == saved["published_byte_sha256"]]
        if (not matching or saved["local_byte_sha256_at_scoring"] not in {
                hashlib.sha256(value).hexdigest() for value in variants}):
            raise ValueError("Forecast bytes changed beyond checkout line endings")
        published = matching[0]
        blob = hashlib.sha1(b"blob " + str(len(published)).encode() + b"\0" + published).hexdigest()
        if blob != receipt["blob_sha"]:
            raise ValueError("Published Git blob digest disagrees with the original file")
        api = "https://api.github.com/repos/philippmch/f1-simulator/contents/"
        if receipt["contents_api_url"] != api + saved["path"] + "?ref=" + receipt["commit"]:
            raise ValueError("Publication receipt must identify the exact forecast and commit")
        run_api = "https://api.github.com/repos/philippmch/f1-simulator/actions/runs/"
        run_page = "https://github.com/philippmch/f1-simulator/actions/runs/"
        receipt_api_url = receipt["receipt_api_url"]
        run_id = receipt_api_url.removeprefix(run_api)
        if (not receipt_api_url.startswith(run_api) or not run_id.isascii()
                or not run_id.isdecimal() or receipt["receipt_url"] != run_page + run_id
                or receipt["receipt_head_sha"] != receipt["commit"]
                or receipt["receipt_event"] != "push"):
            raise ValueError(
                "Publication receipt must be a push of that commit in the same repository",
            )
        if timestamp(receipt["created_at"]) >= timestamp(forecast["qualifying_starts_at"]):
            raise ValueError("Dated publication receipt must precede qualifying")
        report = saved["score_report"]
        provenance = report["observations_provenance"]
        if not provenance["fresh_fetch"] or provenance["cache"] != "disabled":
            raise ValueError("Score evidence must record a fresh result collection")
        if timestamp(provenance["fetched_at"]).date() <= timestamp(forecast["recorded_at"]).date():
            raise ValueError("Observation collection must follow the saved forecast")
        event_date = datetime.fromisoformat(event["date"]).date()
        if timestamp(provenance["fetched_at"]).date() < event_date:
            raise ValueError("Observed race collection must not precede the Grand Prix date")
        roster = {row["id"] for row in forecast["roster"]}
        for rows, source in ((evidence["observed_results"], evidence["result_evidence"]),
                             (evidence["observed_qualifying"], evidence["qualifying_evidence"])):
            if (len(rows) != len(roster) or {row["Driver"]["code"] for row in rows} != roster
                    or any(row["round"] != event["round"] for row in rows)
                    or source["coverage"] != len(rows) or not source["fresh_fetch"]
                    or source["cache"] != "disabled"
                    or timestamp(source["fetched_at"]).date() < event_date):
                raise ValueError("Observed evidence must cover the complete original field")
        loader = _OfflineLoader(event, provenance)
        rebuilt = score_recorded_forecast(
            forecast, loader, evidence["observed_results"], evidence["observed_qualifying"],
        )
        if rebuilt != report or rebuilt["observed_winner"] != evidence["observed_winner"]:
            raise ValueError(
                "Saved scores disagree with original counts, policy or observed results",
            )
        modeled = forecast.get("winner_estimate", forecast["winner_forecast"])
        results.append({"path": saved["path"], "forecast_stage": "pre_qualifying",
                        "winner_policy": report.get("winner_policy", "native_simulation"),
                        "winner_brier": rebuilt["winner_score"]["brier_score"],
                        "native_winner_brier": rebuilt.get(
                            "native_winner_score", rebuilt["winner_score"])["brier_score"],
                        "probability_actual_winner": modeled["drivers"][
                            rebuilt["observed_winner"]]["probability"]})
    return {"events": 1, "forecast_stage": "pre_qualifying",
            "after_qualifying_grid_reference_evidence": False,
            "observed_winner": evidence["observed_winner"], "scores": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="?", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    try:
        result = assess(json.loads(args.evidence.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
