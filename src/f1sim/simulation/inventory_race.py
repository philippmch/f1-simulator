"""Native race policy and fit accounting for explicit, reusable tyre pools."""

from copy import deepcopy

import numpy as np

from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import projected_surfaces
from f1sim.simulation.tire_inventory import TireInventory


def _timed_stop_budget_envelope(maximum, dry_limit, damp_limit, weather_clock):
    """Keep delayed branches inside every weather-state stop allowance.

    A paid stop can advance an external weather clock past a transition that
    is absent from the no-stop forecast used by the caller. The planner still
    applies ``dry_limit`` and ``damp_limit`` at each branch; this is only the
    global search envelope, including the existing four-stop rain allowance.
    """
    if weather_clock is None:
        return maximum
    return max(maximum, dry_limit, damp_limit, 4)


class InventoryStrategyMixin:
    """Finite-set execution shares the race's clocks, physics and stop limits."""

    def _inventory_opening_set(self, driver, car, track, weather, strategy, records,
                               compound=None, age=0):
        from f1sim.simulation.opening_strategy import inventory_opening_policy_costs

        inventory = TireInventory.from_sets(records)
        if compound is not None:
            selected = next(item for item in inventory.sets.values()
                            if item.compound == compound and item.age == age)
        else:
            scores = inventory_opening_policy_costs(
                driver, car, track, weather, strategy, self.strategy_tuning,
                self.strategy_profiles, records,
                **({"tire_warmup": self.tire_warmup} if self.tire_warmup else {}),
            )
            selected = inventory.sets[min(scores, key=lambda item: item[1])[0]]
        return inventory, selected

    @staticmethod
    def _initialize_inventory(state, inventory, selected, *, lap=1):
        fitted = inventory.fit(selected.id)
        state.tire_inventory = inventory
        state.current_tire = TIRE_COMPOUNDS[fitted.compound].model_copy(deep=True)
        state.tire_laps = state.prior_tire_laps = fitted.age
        state.driver.current_tire_laps = fitted.age
        state.tire_compound_history = [fitted.compound.value]
        state.tire_set_history = [dict(lap=lap, kind="start", set_id=fitted.id,
                                      compound=fitted.compound.value, age_at_fit=fitted.age,
                                      age_at_end=fitted.age, laps_used=0)]

    @staticmethod
    def _finish_inventory_stint(state):
        if state.tire_set_history:
            stint = state.tire_set_history[-1]
            stint["age_at_end"] = state.tire_laps
            stint["laps_used"] = state.tire_laps - stint["age_at_fit"]

    def _fit_inventory_tire(self, state, set_id, lap, kind):
        inventory = state.tire_inventory
        if set_id == inventory.current_set_id:
            inventory.fit(set_id, current_age=state.tire_laps)
            return  # Retaining a physical set is not a new stint or fresh tyres.
        fitted = inventory.fit(set_id, current_age=state.tire_laps)
        self._finish_inventory_stint(state)
        self._fit_tire(state, fitted.compound)
        state.tire_laps = state.prior_tire_laps = fitted.age
        state.driver.current_tire_laps = fitted.age
        state.tire_set_history.append(dict(
            lap=lap, kind=kind, set_id=fitted.id, compound=fitted.compound.value,
            age_at_fit=fitted.age, age_at_end=fitted.age, laps_used=0,
        ))

    @staticmethod
    def _inventory_result_fields(state):
        if state.tire_inventory is None:
            return {}
        history = deepcopy(state.tire_set_history)
        if history:
            history[-1]["age_at_end"] = state.tire_laps
            history[-1]["laps_used"] = state.tire_laps - history[-1]["age_at_fit"]
        return dict(tire_set_history=history,
                    tire_inventory=state.tire_inventory.snapshot(state.tire_laps))

    @staticmethod
    def _damage_inventory_tire(state):
        if state.tire_inventory is not None:
            state.tire_inventory.mark_current_unavailable(state.tire_laps)

    @staticmethod
    def _retire_without_inventory_tire(state):
        from f1sim.simulation.race import DriverStatus

        state.status = DriverStatus.DNF
        state.dnf_reason = "No suitable replacement tyre set available"
        state.driver.dnf = True
        state.driver.dnf_reason = state.dnf_reason
        state.inventory_pit_proposal = None
        state.pit_decision_context = None

    def _plan_inventory(self, state, track, weather, lap, *, physical_total_laps=None,
                        weather_intervals=None, additional_current_stop_cost=0,
                        current_traffic_gaps=None, force_stop=False, free_fit=False,
                        weather_clock: StrategyWeatherClock | None = None):
        from f1sim.simulation.inventory_strategy import plan_inventory_strategy

        surfaces = projected_surfaces(weather, track.total_laps - lap + 1, weather_intervals)
        dry_limit = self._dry_stop_budget(state, track)
        damp_limit = self._ordinary_stop_budget(state, track)
        maximum = damp_limit
        if any(surface.track_wetness < .08 and surface.rain_intensity < .15
               for surface in surfaces):
            maximum = max(maximum, dry_limit)
        if (state.current_tire.compound.value in {"intermediate", "wet"}
                or any(surface.track_wetness > .3 or surface.fresh_rain_compound() is not None
                       for surface in surfaces)):
            maximum = max(maximum, 4)
        maximum = _timed_stop_budget_envelope(
            maximum, dry_limit, damp_limit, weather_clock,
        )
        options = {}
        if weather_clock is not None:
            options["weather_clock"] = weather_clock
        if self.tire_warmup:
            options["tire_warmup"] = self.tire_warmup
            options["current_fit_pending"] = state.fit_lap_pending
        return plan_inventory_strategy(
            state.driver, state.car, track, weather, state.tire_inventory, lap,
            tire_age=state.tire_laps, remaining_stops=max(0, maximum - state.pit_stops),
            remaining_dry_stops=max(0, dry_limit - state.pit_stops),
            remaining_damp_stops=max(0, damp_limit - state.pit_stops),
            used_compounds=self._actually_used_compounds(state),
            pit_lane_factor=self._pit_lane_factor(),
            additional_current_stop_cost=additional_current_stop_cost,
            current_lap_time_modifier=self.event_manager.get_lap_time_modifier(),
            active_aero_enabled=self.event_manager.is_active_aero_allowed(),
            physical_total_laps=physical_total_laps, weather_intervals=weather_intervals,
            current_traffic_gaps=current_traffic_gaps, force_stop=force_stop, free_fit=free_fit,
            require_compound_rule=(physical_total_laps or track.total_laps) > 1,
            **options,
        )

    def _inventory_immediate_set(self, state, track, weather, lap, *, free_fit=False,
                                 physical_total_laps=None,
                                 weather_clock: StrategyWeatherClock | None = None):
        """Survive the next lap if no complete forecast is feasible.

        Later weather or finish changes can invalidate today's complete plan.
        A mandatory final correction still requires a genuinely unused compound.
        """
        inventory = state.tire_inventory
        candidates = list(inventory.replacements())
        if free_fit and inventory.current_set_id not in inventory.unavailable_ids:
            candidates.insert(0, inventory.sets[inventory.current_set_id])
        candidates = [item for item in candidates
                      if weather.tire_mismatch(item.compound) != "critical"]
        if lap >= max(2, track.total_laps) and not self._stay_satisfies_tire_rule(state):
            used = self._actually_used_compounds(state)
            candidates = [item for item in candidates
                          if any(c.value in {"intermediate", "wet"} for c in used | {item.compound})
                          or len(used | {item.compound}) >= 2]
        if not candidates:
            return None
        driver = state.driver.model_copy(deep=True)
        simulator = LapSimulator(np.random.default_rng(0))
        ranking_weather = weather
        if weather_clock is not None and not free_fit:
            ranking_weather, _ = self._projected_stint_weather(
                weather, weather_clock, None, 1,
            )

        def cost(item):
            driver.current_tire_laps = (state.tire_laps if item.id == inventory.current_set_id
                                       else item.age)
            value = simulator.calculate_lap_time(
                driver, state.car, track, TIRE_COMPOUNDS[item.compound], ranking_weather,
                lap, physical_total_laps or track.total_laps, sample_variation=False,
                active_aero_enabled=self.event_manager.is_active_aero_allowed(),
            )
            if self.tire_warmup and (
                item.id != inventory.current_set_id or state.fit_lap_pending
            ):
                value += self.tire_warmup.get(item.compound.value, 0.0)
            return value

        return min(candidates, key=cost).id

    def _should_pit_inventory(self, state, all_states, track, lap, weather,
                              additional_current_stop_cost=0, physical_total_laps=None,
                              traffic_snapshot=None, weather_intervals=None,
                              weather_clock: StrategyWeatherClock | None = None,
                              current_overtake_mode_active=False):
        from dataclasses import replace

        from f1sim.simulation.race import TeamStrategyArchetype

        state.inventory_pit_proposal = None
        state.pit_decision_context = None
        if self.event_manager.red_flag_active:
            return False
        gap = (self._get_gap_to_car_ahead(state, all_states) if traffic_snapshot is None
               else traffic_snapshot.gap_ahead)
        if self._should_switch_conservative_to_balanced(state, lap, track, gap):
            state.strategy_archetype = TeamStrategyArchetype.BALANCED
        inventory = state.tire_inventory
        compulsory = (inventory.current_set_id in inventory.unavailable_ids
                      or weather.tire_mismatch(state.current_tire.compound) == "critical"
                      or (lap >= max(2, track.total_laps)
                          and not self._stay_satisfies_tire_rule(state)))
        if lap <= 1 and not compulsory:
            return False
        gaps = None
        traffic_cost = 0
        if self.event_manager.is_active_aero_allowed():
            if traffic_snapshot is None:
                gaps = self._pit_rejoin_traffic_gaps(
                    state, all_states, track, additional_current_stop_cost,
                )
            else:
                gaps = traffic_snapshot.current_traffic_gaps
                if gaps is None:
                    traffic_cost = traffic_snapshot.rejoin_traffic_cost
                    traffic_cost *= self.lap_simulator.weather_pace_multiplier(
                        state.driver, state.car, weather,
                    )
        decision = self._plan_inventory(
            state, track, weather, lap, physical_total_laps=physical_total_laps,
            weather_intervals=weather_intervals,
            additional_current_stop_cost=additional_current_stop_cost + traffic_cost,
            current_traffic_gaps=gaps, force_stop=compulsory, weather_clock=weather_clock,
        )
        mode_gain = self._strategy_mode_gain(
            state, track, weather, lap, current_overtake_mode_active,
            gaps[0] if gaps is not None else None, physical_total_laps,
        )
        if mode_gain:
            decision = replace(decision, wait_cost=decision.wait_cost - mode_gain)
        bias = ({TeamStrategyArchetype.AGGRESSIVE: .1, TeamStrategyArchetype.BALANCED: 0,
                 TeamStrategyArchetype.CONSERVATIVE: -.1}[state.strategy_archetype]
                if weather.track_wetness < .08 and weather.rain_intensity < .15 else 0)
        if compulsory or decision.should_pit(bias):
            state.inventory_pit_proposal = (lap, decision.set_id)
            critical_weather = weather.tire_mismatch(state.current_tire.compound) == "critical"
            unavailable = inventory.current_set_id in inventory.unavailable_ids
            compound_requirement = (
                lap >= max(2, track.total_laps)
                and not self._stay_satisfies_tire_rule(state)
            )
            if critical_weather:
                reason = "critical_weather"
            elif unavailable:
                reason = "forced_repair"
            elif compound_requirement:
                reason = "compound_requirement"
            else:
                reason = "inventory_forecast"
            self._capture_pit_decision_context(
                state, lap, reason, None if compulsory else decision,
            )
            return True
        return False

    def _prepare_inventory_pit(self, state, track, weather, lap, *, physical_total_laps=None,
                               weather_intervals=None, current_traffic_gaps=None,
                               additional_current_stop_cost=0,
                               weather_clock: StrategyWeatherClock | None = None):
        """Select an actual available set before reserving or sampling service."""
        inventory = state.tire_inventory
        proposal = state.inventory_pit_proposal
        available = {item.id: item for item in inventory.replacements()
                     if weather.tire_mismatch(item.compound) != "critical"}
        selected = proposal[1] if proposal is not None and proposal[0] == lap else None
        if selected not in available:
            decision = self._plan_inventory(
                state, track, weather, lap, force_stop=True,
                physical_total_laps=physical_total_laps, weather_intervals=weather_intervals,
                current_traffic_gaps=current_traffic_gaps,
                additional_current_stop_cost=additional_current_stop_cost,
                weather_clock=weather_clock,
            )
            selected = decision.set_id or self._inventory_immediate_set(
                state, track, weather, lap, physical_total_laps=physical_total_laps,
                weather_clock=weather_clock,
            )
        if selected is None:
            self._retire_without_inventory_tire(state)
            return False
        state.inventory_pit_proposal = (lap, selected)
        return True

    def _refit_inventory_free(self, state, track, weather, current_lap, *,
                              physical_total_laps=None, weather_intervals=None,
                              weather_clock: StrategyWeatherClock | None = None):
        state.pit_decision_context = None
        lap = current_lap + 1
        decision = self._plan_inventory(
            state, track, weather, lap, free_fit=True,
            physical_total_laps=physical_total_laps, weather_intervals=weather_intervals,
            weather_clock=weather_clock,
        )
        selected = decision.set_id or self._inventory_immediate_set(
            state, track, weather, lap, free_fit=True, physical_total_laps=physical_total_laps,
        )
        if selected is None:
            self._retire_without_inventory_tire(state)
            return False
        self._fit_inventory_tire(state, selected, lap, "red_flag")
        state.force_pit_next_lap = False
        state.inventory_pit_proposal = None
        state.dry_pit_proposal = None
        state.weather_pit_proposal = None
        return True
