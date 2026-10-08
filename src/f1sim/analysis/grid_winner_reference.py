"""Fixed published-grid references frozen before the race, after GP qualifying."""

import math
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from f1sim.analysis.race_probability_scores import score_winner_probabilities
from f1sim.simulation.execution import validate_pit_lane_starters, validate_starting_grid

GRID_REFERENCE_SCALES = (6, 12, 18)


def _instant(value):
    if not isinstance(value, str):
        raise ValueError("Grid reference timestamps must include a timezone")
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Grid reference timestamps must include a timezone")
    return instant.astimezone(timezone.utc)


def build_grid_winner_references(
    entrant_ids, evidence, *, year, target_round, recorded_at, qualifying_starts_at, race_starts_at,
):
    """Use fixed exp(scale * normalized grid rank), with no race-outcome fitting."""
    ids = validate_starting_grid(list(entrant_ids))
    if len(ids) < 2 or not isinstance(evidence, dict):
        raise ValueError("Grid references require a complete field and published evidence")
    if (evidence.get("mode") != "published"
            or type(evidence.get("year")) is not int or evidence["year"] != year
            or type(evidence.get("round")) is not int or evidence["round"] != target_round
            or not isinstance(evidence.get("source_url"), str)
            or not re.fullmatch(rf"https://www\.formula1\.com/en/results/{year}/races/"
                                r"\d+/[a-z0-9-]+/starting-grid", evidence["source_url"])):
        raise ValueError("Grid references require published starting-grid evidence for this event")
    grid = validate_starting_grid(evidence.get("starting_grid"), ids)
    if grid is None:
        raise ValueError("Grid references require a supplied starting grid")
    validate_pit_lane_starters(evidence.get("pit_lane_starters"), grid)
    fetched = _instant(evidence.get("fetched_at"))
    qualifying, race, recorded = map(_instant, (qualifying_starts_at, race_starts_at, recorded_at))
    if (qualifying.year != year or race.year != year
            or _instant(evidence.get("qualifying_started_at")) != qualifying
            or not qualifying + timedelta(hours=1) <= fetched <= recorded < race):
        raise ValueError("Grid reference must be fetched after GP qualifying and before the race")
    references = {}
    for scale in GRID_REFERENCE_SCALES:
        # Subtract the maximum log weight to keep all probabilities finite.
        weights = [math.exp(-scale * index / (len(ids) - 1)) for index in range(len(ids))]
        denominator = math.fsum(weights)
        references[f"grid_rank_softmax_{scale}_v1"] = {
            "scale": scale,
            "probabilities": dict(zip(grid, (w / denominator for w in weights), strict=True)),
            "no_classified_winner_probability": 0.0,
        }
    return {"evidence": deepcopy(evidence), "references": references}


def score_saved_grid_references(
    saved, entrant_ids, *, year, target_round, recorded_at, qualifying_starts_at, race_starts_at,
    observed_winner=None,
):
    """Rebuild from saved grid evidence without consulting a later published grid."""
    if not isinstance(saved, dict):
        raise ValueError("Recorded grid references are missing")
    rebuilt = build_grid_winner_references(
        entrant_ids, saved.get("evidence"), year=year, target_round=target_round,
        recorded_at=recorded_at, qualifying_starts_at=qualifying_starts_at,
        race_starts_at=race_starts_at,
    )
    # Saved scores are deliberately ignored; the frozen formulas are verified.
    references = saved.get("references")
    if not isinstance(references, dict) or set(references) != set(rebuilt["references"]):
        raise ValueError("Recorded grid reference policies differ from their fixed definitions")
    for name, reference in rebuilt["references"].items():
        candidate = references[name]
        if (not isinstance(candidate, dict)
                or type(candidate.get("scale")) is not int
                or candidate["scale"] != reference["scale"]
                or candidate.get("probabilities") != reference["probabilities"]
                or type(candidate.get("no_classified_winner_probability")) not in (int, float)
                or candidate["no_classified_winner_probability"] != 0.0):
            raise ValueError("Recorded grid probabilities differ from their saved grid")
        if observed_winner is not None:
            reference["score"] = score_winner_probabilities(
                reference["probabilities"], 0.0, observed_winner,
            )
    return rebuilt
