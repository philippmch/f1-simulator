"""Optional race-only tyre usage constraints, separate from physical inventory."""

from collections.abc import Mapping

RACE_TIRE_RULES_POLICY = "completed_race_laps_and_used_sets_v1"
RACE_TIRE_RULES_SCHEMA_VERSION = 16


def validate_race_tire_rules(value):
    """Copy positive integer rules; absent/default rules preserve ordinary races.

    ``max_set_laps`` limits completed own race laps on one physical set. It is
    independent of prior wear and any explicit inventory's remaining allowance.
    ``minimum_sets`` counts distinct sets actually run, including free refits.
    These inputs cannot reconstruct a set's unknown earlier weekend usage.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("race_tire_rules must be an object")
    if set(value) - {"max_set_laps", "minimum_sets"}:
        raise ValueError("race_tire_rules allows only max_set_laps and minimum_sets")
    result = {}
    for name in ("max_set_laps", "minimum_sets"):
        if name not in value:
            continue
        number = value[name]
        if type(number) is not int or number <= 0:
            raise ValueError(f"race_tire_rules {name} must be a positive integer")
        if name != "minimum_sets" or number != 1:
            result[name] = number
    return result


def merge_race_tire_rules(*values):
    """Combine independent constraints without weakening an earlier requirement."""
    cap = None
    minimum = 1
    for value in values:
        rules = validate_race_tire_rules(value)
        if "max_set_laps" in rules:
            cap = rules["max_set_laps"] if cap is None else min(cap, rules["max_set_laps"])
        minimum = max(minimum, rules.get("minimum_sets", 1))
    result = {}
    if cap is not None:
        result["max_set_laps"] = cap
    if minimum > 1:
        result["minimum_sets"] = minimum
    return result


def validate_race_tire_rules_snapshot(snapshot):
    """Restore enabled schema16 rules without reinterpreting older snapshots."""
    if not isinstance(snapshot, Mapping):
        raise ValueError("race_tire_rules snapshot must be an object")
    has_rules = "race_tire_rules" in snapshot
    has_policy = "race_tire_rules_policy" in snapshot
    if snapshot.get("schema_version") == RACE_TIRE_RULES_SCHEMA_VERSION:
        if not has_rules or snapshot.get("race_tire_rules_policy") != RACE_TIRE_RULES_POLICY:
            raise ValueError("Schema 16 requires race_tire_rules and its supported policy")
        rules = validate_race_tire_rules(snapshot["race_tire_rules"])
        if not rules:
            raise ValueError("Schema 16 requires nonempty race_tire_rules")
        return rules
    if has_rules or has_policy:
        raise ValueError("race_tire_rules and its policy require schema 16")
    return {}


class RaceTireUsage:
    """Keep race-only physical-set usage without inventing a finite tyre pool.

    Explicit inventory identities retain their cumulative race use when refitted.
    An omitted identity represents a fresh replacement from the ordinary unlimited
    pool. Prior wear and explicit inventory allowances remain separate constraints.
    A started but interrupted lap marks the set used, but earns no completed lap.
    """

    def __init__(self, rules):
        self.rules = validate_race_tire_rules(rules)
        self.current_set_id = None
        self.used_set_ids = set()
        self.completed_laps_by_set = {}
        self._next_fresh_id = 0
        self._lap_in_progress = False

    @staticmethod
    def _validate_id(set_id):
        if not isinstance(set_id, str) or not set_id.strip():
            raise ValueError("race_tire_rules set ID must be a nonempty string")
        return set_id

    def remaining_laps(self, set_id=None):
        """Return remaining race-only allowance; unknown candidate IDs are unused."""
        if set_id is not None:
            self._validate_id(set_id)
        cap = self.rules.get("max_set_laps")
        if cap is None:
            return None
        identity = self.current_set_id if set_id is None else set_id
        return cap - self.completed_laps_by_set.get(identity, 0)

    def can_fit(self, set_id):
        """Inspect a physical candidate without fitting it or consuming allowance."""
        if set_id is None:
            return True  # A newly allocated unlimited-pool set has no race use.
        remaining = self.remaining_laps(self._validate_id(set_id))
        return remaining is None or remaining > 0

    def fit(self, set_id=None):
        """Select a physical set; free and paid fittings use the same accounting."""
        if set_id is None:
            while True:
                self._next_fresh_id += 1
                set_id = f"unlimited-fresh-{self._next_fresh_id}"
                if set_id not in self.completed_laps_by_set:
                    break
        else:
            self._validate_id(set_id)
        if not self.can_fit(set_id):
            raise ValueError("race_tire_rules cannot fit a set with no remaining race laps")
        self.completed_laps_by_set.setdefault(set_id, 0)
        self.current_set_id = set_id
        # A free refit may interrupt a running lap. No completed use is inferred.
        self._lap_in_progress = False
        return set_id

    def begin_running(self):
        """Mark the fitted set actually used, without crediting a completed lap."""
        if self.current_set_id is None:
            raise ValueError("race_tire_rules requires a fitted set before running")
        remaining = self.remaining_laps()
        if remaining is not None and remaining <= 0:
            raise ValueError("race_tire_rules cannot run a set with no remaining race laps")
        self.used_set_ids.add(self.current_set_id)
        self._lap_in_progress = True

    def complete_lap(self):
        """Consume exactly one completed own race lap after actual running began."""
        if self.current_set_id is None or not self._lap_in_progress:
            raise ValueError("race_tire_rules cannot complete a lap before running")
        remaining = self.remaining_laps()
        if remaining is not None and remaining <= 0:
            raise ValueError("race_tire_rules completed race laps would exceed the set cap")
        self.completed_laps_by_set[self.current_set_id] += 1
        self._lap_in_progress = False

    def needs_new_set(self, lap, total_laps):
        """Reserve one upcoming running lap for each still-required distinct set."""
        if type(lap) is not int or type(total_laps) is not int or lap < 1 or total_laps < 1:
            raise ValueError("race_tire_rules lap and total_laps must be positive integers")
        if lap > total_laps or self.current_set_id not in self.used_set_ids:
            return False
        missing = max(0, self.rules.get("minimum_sets", 1) - len(self.used_set_ids))
        return missing > 0 and missing >= total_laps - lap + 1
