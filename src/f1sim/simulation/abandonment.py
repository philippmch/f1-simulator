"""Historical completed crossings for an explicitly abandoned race.

Recording is opt-in for scenarios containing an abandonment request. Snapshots
contain result data, never mutable physics models or future strategy inputs.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite
from numbers import Real

from f1sim.simulation.race_points import (
    RacePointsContext,
    points_for_classification,
    race_scoring_context,
)


@dataclass(frozen=True)
class RaceAbandonment:
    announcement_lap: int
    signal_leader_lap: int
    countback_lap: int
    signal_time: float
    countback_time: float
    decision_time: float
    policy: str = "after_crossing_penultimate_lap_2026_v1"


@dataclass(frozen=True)
class AbandonmentTireRule:
    used_compounds: tuple[str, ...]
    penalty_seconds: int
    reason: str
    policy: str = "two_dry_specs_abandoned_2026_v1"


def abandonment_tire_rule(compounds, countback_lap):
    used = frozenset(compounds)
    reason = ("no_countback_distance" if not countback_lap else
              "no_completed_tyre_use" if not used else
              "wet_tyre_used" if used & {"intermediate", "wet"} else
              "two_dry_compounds" if len(used & {"soft", "medium", "hard"}) >= 2 else
              "dry_compound_requirement")
    return AbandonmentTireRule(tuple(sorted(used)),
                               30 if reason == "dry_compound_requirement" else 0, reason)


def serialize_abandonment_tire_rule(value):
    if isinstance(value, AbandonmentTireRule):
        value = asdict(value)
    if not isinstance(value, dict) or value.get("policy") != "two_dry_specs_abandoned_2026_v1":
        return None
    used = value.get("used_compounds")
    if (not isinstance(used, (list, tuple)) or any(not isinstance(item, str) for item in used)
            or list(used) != sorted(set(used))
            or not set(used) <= {"soft", "medium", "hard", "intermediate", "wet"}
            or type(value.get("penalty_seconds")) is not int):
        return None
    expected = abandonment_tire_rule(used, value.get("reason") != "no_countback_distance")
    if (value.get("reason") != expected.reason
            or value["penalty_seconds"] != expected.penalty_seconds):
        return None
    return asdict(expected) | {"used_compounds": list(used)}


def serialize_abandonment(value):
    """Normalize recorded evidence; malformed clocks or policies stay unknown."""
    if isinstance(value, RaceAbandonment):
        value = asdict(value)
    if not isinstance(value, dict):
        return None
    integers = ("announcement_lap", "signal_leader_lap", "countback_lap")
    clocks = ("signal_time", "countback_time", "decision_time")
    if (value.get("policy") != "after_crossing_penultimate_lap_2026_v1"
            or any(type(value.get(key)) is not int for key in integers)
            or not 1 <= value["announcement_lap"] <= 1000
            or not 2 <= value["signal_leader_lap"] <= 1001
            or value["countback_lap"] != value["signal_leader_lap"] - 2
            or any(not _finite_clock(value.get(key)) for key in clocks)
            or not 0 <= value["countback_time"] <= value["signal_time"] <= value["decision_time"]):
        return None
    result = {key: value[key] for key in (*integers, *clocks, "policy")}
    result["description"] = (
        f"Race abandoned after leading crossing {value['announcement_lap']} "
        f"(leader entering lap {value['signal_leader_lap']}). "
        + (f"Countback classification: end of lap {value['countback_lap']} "
           f"at {value['countback_time']:.3f} seconds."
           if value["countback_lap"] else "No completed countback lap: no race result.")
    )
    return result


def _finite_clock(value):
    try:
        return not isinstance(value, bool) and isinstance(value, Real) and isfinite(value)
    except OverflowError:
        return False


def race_abandonment_context(results):
    """Require consistent historical evidence and scoring across the whole field."""
    try:
        rows = list(results)
    except (TypeError, ValueError):
        return None
    if not rows:
        return None
    evidence = serialize_abandonment(getattr(rows[0], "race_abandonment", None))
    if evidence is None:
        return None
    for row in rows:
        if serialize_abandonment(getattr(row, "race_abandonment", None)) != evidence:
            return None
        laps = getattr(row, "laps_completed", None)
        if type(laps) is not int or not 0 <= laps <= evidence["countback_lap"]:
            return None
        if (evidence["countback_lap"] == 0
                and getattr(getattr(row, "status", None), "value", None) != "no_result"):
            return None
        rule = serialize_abandonment_tire_rule(getattr(row, "abandonment_tire_rule", None))
        if rule is None or ((rule["reason"] == "no_countback_distance")
                            != (evidence["countback_lap"] == 0)):
            return None
    scoring = race_scoring_context(rows)
    if scoring is None or scoring["winner_laps"] != (evidence["countback_lap"] or None):
        return None
    return evidence


class CountbackHistory:
    """Keep completed own-lap outcomes and the original leading crossing clocks."""

    def __init__(self):
        self.rows = {}
        self.leaders = {}

    def capture(self, state, fastest, inventory_fields, *, at=None):
        from f1sim.simulation.race import DriverStatus, RaceResult

        status = DriverStatus.FINISHED if state.status == DriverStatus.RACING else state.status
        row = RaceResult(
            state.driver.id, state.driver.name, state.car.team_name, state.position,
            state.total_time, 0., state.pit_stops, fastest, status,
            dnf_reason=state.dnf_reason, strategy=list(state.tire_compound_history),
            laps_completed=state.laps_completed, pit_laps=list(state.pit_laps),
            pit_stop_details=deepcopy(state.pit_stop_details),
            pit_plan_history=deepcopy(state.pit_plan_history),
            overtake_attempts=state.overtake_attempts, overtake_successes=state.overtake_successes,
            overtake_contacts=state.overtake_contacts, **inventory_fields,
        )
        self.rows.setdefault(state.driver.id, []).append(
            (state.total_time if at is None else at, row),
        )

    def leading_crossing(self, driver_id, lap, time, green_pair):
        # A lapped successor can repeat an earlier leader distance. That must
        # not replace the original race's historical end of this lap.
        self.leaders.setdefault(lap, (driver_id, time, green_pair))

    def classify(self, *, scheduled_laps, announcement_lap, signal_leader_lap,
                 signal_time, decision_time, suspension_seconds, used_compounds):
        from f1sim.simulation.race import DriverStatus

        # Scheduled red flags are announced after a completed leading crossing:
        # the leader is entering the following lap when the signal is given.
        lap = max(0, signal_leader_lap - 2)
        if lap:
            _, cutoff, green_pair = self.leaders[lap]
        else:
            cutoff, green_pair = 0., False
        evidence = RaceAbandonment(announcement_lap, signal_leader_lap, lap, signal_time,
                                   cutoff, decision_time)
        rows = []
        for history in self.rows.values():
            # The first own crossing at/after the historical leading flag
            # completes that car's current lap, including a lapped follower.
            # A prior retirement keeps its last completed distance instead.
            row = (next((row for time, row in history if time >= cutoff), history[-1][1])
                   if lap else history[0][1])
            row = deepcopy(row)
            rule = abandonment_tire_rule(used_compounds[row.driver_id], lap)
            rows.append(replace(row, total_time=row.total_time + rule.penalty_seconds,
                                abandonment_tire_rule=rule))
        rows.sort(key=lambda row: (-row.laps_completed, row.total_time, row.position))
        winner_laps = rows[0].laps_completed if lap and rows else None
        context = RacePointsContext(scheduled_laps, winner_laps, green_pair)
        leader_time = rows[0].total_time if lap and rows else 0.
        results = []
        for position, row in enumerate(rows, 1):
            classified = (winner_laps is not None and row.laps_completed > 0
                          and row.laps_completed >= winner_laps * 9 // 10)
            for item in row.pit_plan_history or []:
                if item["status"] is None:
                    item.update(status="not_reached", reason="race_abandoned",
                                actual_compound=None, actual_set_id=None)
            results.append(replace(
                row, position=position, classified=classified,
                status=row.status if lap else DriverStatus.NO_RESULT,
                gap_to_leader=(max(0., row.total_time - leader_time)
                               if lap and row.status == DriverStatus.FINISHED else 0.),
                points_awarded=points_for_classification(
                    position, classified, winner_laps, scheduled_laps, green_pair,
                ),
                race_points_context=context, race_suspension_seconds=suspension_seconds,
                race_abandonment=evidence,
            ))
        return results, context, evidence
