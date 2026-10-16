"""Execution helpers for optional race-wide tyre-use constraints."""

from copy import deepcopy
from dataclasses import replace
from math import ceil

from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.tire_rules import RACE_TIRE_RULES_POLICY, RaceTireUsage


class RaceTireRuleMixin:
    @staticmethod
    def _race_tire_dry_forecast_options(state, *, replacement=False):
        usage = state.race_tire_usage
        if usage is None:
            return {}
        cap = usage.rules.get("max_set_laps")
        prospective = len(usage.used_set_ids) + int(
            replacement or usage.current_set_id not in usage.used_set_ids)
        return {"maximum_set_laps": cap,
                "current_remaining_set_laps": cap if replacement else usage.remaining_laps(),
                "minimum_new_sets": max(0, usage.rules.get("minimum_sets", 1)-prospective)}

    @classmethod
    def _race_tire_required_stops(cls, state, remaining_laps, *, replacement=False):
        options = cls._race_tire_dry_forecast_options(state, replacement=replacement)
        if not options:
            return 0
        cap = options["maximum_set_laps"]
        available = options["current_remaining_set_laps"]
        cap_stops = (0 if cap is None else
                     ceil(max(0, remaining_laps-available)/cap))
        return max(options["minimum_new_sets"], cap_stops)

    @classmethod
    def _attach_race_tire_usage(cls, results, states):
        by_id = {state.driver.id: state for state in states}
        for result in results:
            state = by_id[result.driver_id]
            fields = cls._race_tire_usage_fields(state)
            if fields:
                result.race_tire_usage = fields["race_tire_usage"]
        return results

    def _initialize_race_tire_usage(self, state):
        if not self.race_tire_rules:
            return
        state.race_tire_usage = RaceTireUsage(self.race_tire_rules)
        identity = None if state.tire_inventory is None else state.tire_inventory.current_set_id
        state.race_tire_usage.fit(identity)
        if state.tire_inventory is not None:
            state.tire_inventory.race_tire_usage = state.race_tire_usage

    @staticmethod
    def _race_tire_rule_reason(state, track, lap):
        usage = state.race_tire_usage
        if usage is None:
            return None
        fresh_required = usage.needs_new_set(lap, track.total_laps)
        if state.tire_inventory is not None:
            state.tire_inventory.require_unused_race_set = fresh_required
        if usage.remaining_laps() == 0:
            return "race_tire_set_limit"
        if fresh_required:
            return "race_tire_minimum_sets"
        return None

    @staticmethod
    def _race_tire_retention_legal(state, remaining_laps):
        usage = state.race_tire_usage
        if usage is None:
            return True
        remaining = usage.remaining_laps()
        prospective = len(usage.used_set_ids | {usage.current_set_id})
        return ((remaining is None or remaining >= remaining_laps)
                and prospective >= usage.rules.get("minimum_sets", 1))

    @staticmethod
    def _race_tire_usage_fields(state):
        usage = state.race_tire_usage
        if usage is None:
            return {}
        completed = dict(usage.completed_laps_by_set)
        cap = usage.rules.get("max_set_laps")
        cap_ok = cap is None or all(value <= cap for value in completed.values())
        minimum_ok = len(usage.used_set_ids) >= usage.rules.get("minimum_sets", 1)
        return {"race_tire_usage": {
            "policy": RACE_TIRE_RULES_POLICY, "rules": deepcopy(usage.rules),
            "accounting_scope": "actual_running_to_race_decision",
            "current_set_id": usage.current_set_id,
            "used_set_ids": sorted(usage.used_set_ids),
            "completed_race_laps_by_set": completed,
            "max_set_laps_satisfied": cap_ok, "minimum_sets_satisfied": minimum_ok,
            "compliant": cap_ok and minimum_ok,
            "violations": ([] if cap_ok else ["max_set_laps"])
            + ([] if minimum_ok else ["minimum_sets"]),
        }}

    @staticmethod
    def _race_tire_planning_inventory(state):
        """A detached constraint view of real stock; never persist effective budgets."""
        inventory, usage = state.tire_inventory, state.race_tire_usage
        if usage is None or inventory is None:
            return inventory
        records = []
        for identity, item in inventory.sets.items():
            age = state.tire_laps if identity == inventory.current_set_id else item.age
            declared = (None if item.remaining_laps is None else
                        item.remaining_laps-(age-item.age))
            cap = usage.remaining_laps(identity)
            budgets = [value for value in (declared, cap) if value is not None]
            records.append(replace(item, age=age, remaining_laps=min(budgets) if budgets else None))
        view = TireInventory(records)
        view.current_set_id = inventory.current_set_id
        view.unavailable_ids = set(inventory.unavailable_ids)
        if inventory.require_unused_race_set:
            view.unavailable_ids.update(usage.used_set_ids-{inventory.current_set_id})
        return view
