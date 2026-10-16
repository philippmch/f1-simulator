"""Optional race rules constrain actual physical-set use in both race engines."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverStatus, RaceSimulator


@pytest.fixture(params=["standard", "chronological"])
def execute(request, monkeypatch):
    def run(rules=None, *, records=None, age=0, laps=8, plans=None, controls=None, failure=None):
        sim = RaceSimulator(np.random.default_rng(73), race_tire_rules=rules,
                            control_schedule=controls)
        driver = Driver(id="A", name="A", team_id="A")
        car = Car(team_id="A", team_name="A")
        track = Track(id="T", name="T", country="T", total_laps=laps, base_lap_time=90)
        if controls is None and failure is None:
            monkeypatch.setattr(sim.event_manager, "process_lap", lambda *a, **kw: [])
        monkeypatch.setattr(sim.event_manager, "_check_red_flag_conditions", lambda *a, **kw: None)
        def mechanical(driver, car, track, lap, weather):
            if lap == failure:
                driver.dnf = True
                driver.dnf_reason = "mechanical"
                return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
            return None
        monkeypatch.setattr(sim.event_manager, "_check_mechanical_failure", mechanical)
        monkeypatch.setattr(sim.event_manager, "_check_random_incident", lambda *a, **kw: None)
        monkeypatch.setattr(sim, "_ordinary_stop_budget", lambda *a: 0)
        monkeypatch.setattr(sim, "_dry_stop_budget", lambda *a: 0)
        monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
        method = sim.simulate_race if request.param == "standard" else ChronologicalRace(sim).run
        options = {} if records is None else {"tire_inventory": {"A": records}}
        if plans is not None:
            options["pit_plans"] = {"A": plans}
        result = method([driver], {"A": car}, track,
                        Weather(track_wetness=.5, rain_intensity=.5, change_probability=0),
                        ["A"], starting_tires={"A": "intermediate"},
                        starting_tire_ages={"A": age}, **options)[0]
        return result, sim
    return run


def test_disabled_none_and_empty_rules_preserve_numeric_replay(execute):
    first, _ = execute()
    second, _ = execute({})
    assert asdict(first) == asdict(second)
    assert first.race_tire_usage is None


def test_unlimited_fresh_pool_cap_ignores_starting_wear_and_overrides_empty_plan(execute):
    result, _ = execute({"max_set_laps": 3}, age=17, plans=[])
    assert result.status == DriverStatus.FINISHED
    assert result.pit_laps == [4, 7]
    usage = result.race_tire_usage
    assert usage["completed_race_laps_by_set"] == {
        "unlimited-fresh-1": 3, "unlimited-fresh-2": 3, "unlimited-fresh-3": 2}
    assert len(usage["used_set_ids"]) == 3 and usage["compliant"]
    assert result.tire_inventory is None  # Unlimited stock is not a fabricated finite pool.
    assert all(p["decision_reason"] == "race_tire_set_limit" for p in result.pit_stop_details)


def test_manual_remaining_allowance_and_race_cap_stay_independent(execute):
    records = [dict(id="I1", compound="intermediate", age=17, remaining_laps=2),
               dict(id="I2", compound="intermediate", remaining_laps=8),
               dict(id="I3", compound="intermediate", remaining_laps=8)]
    before = deepcopy(records)
    result, _ = execute({"max_set_laps": 3}, records=records, age=17, plans=[])
    assert result.status == DriverStatus.FINISHED
    assert result.pit_laps == [3, 6]
    usage = result.race_tire_usage
    assert sorted(usage["completed_race_laps_by_set"].values()) == [2, 3, 3]
    assert usage["compliant"] and sum(usage["completed_race_laps_by_set"].values()) == 8
    final = {item["id"]: item for item in result.tire_inventory}
    for identity in ("I2", "I3"):
        assert final[identity]["remaining_laps"] == 8-usage["completed_race_laps_by_set"][identity]
    assert final["I1"]["remaining_laps"] == 0
    assert records == before


def test_minimum_three_sets_applies_in_wet_race_despite_empty_plan(execute):
    result, _ = execute({"minimum_sets": 3}, plans=[])
    assert result.status == DriverStatus.FINISHED
    assert result.pit_laps == [7, 8]
    assert result.race_tire_usage["compliant"]
    assert len(result.race_tire_usage["used_set_ids"]) == 3
    assert len(set(result.strategy)) == 1  # Physical sets, not distinct compounds.
    assert all(p["decision_reason"] == "race_tire_minimum_sets"
               for p in result.pit_stop_details)


def test_inventory_minimum_forces_unused_physical_sets(execute):
    records = [dict(id=f"I{i}", compound="intermediate") for i in range(1, 4)]
    result, _ = execute({"minimum_sets": 3}, records=records, plans=[])
    assert result.status == DriverStatus.FINISHED
    assert result.race_tire_usage["used_set_ids"] == ["I1", "I2", "I3"]
    assert result.pit_stops == 2 and result.race_tire_usage["compliant"]


def test_insufficient_real_stock_retires_before_exceeding_cap(execute):
    records = [dict(id="I1", compound="intermediate"),
               dict(id="I2", compound="intermediate")]
    result, _ = execute({"max_set_laps": 2}, records=records, plans=[])
    assert result.status == DriverStatus.DNF
    assert result.laps_completed == 4
    assert result.race_tire_usage["max_set_laps_satisfied"]
    assert sum(result.race_tire_usage["completed_race_laps_by_set"].values()) == 4


def test_impossible_minimum_is_reported_without_fabricated_used_sets(execute):
    result, _ = execute({"minimum_sets": 3}, laps=2, plans=[])
    assert len(result.race_tire_usage["used_set_ids"]) <= 2
    assert not result.race_tire_usage["compliant"]
    assert result.race_tire_usage["violations"] == ["minimum_sets"]


def test_result_usage_is_detached_from_other_results_and_input_rules(execute):
    rules = {"max_set_laps": 3}
    first, _ = execute(rules, plans=[])
    second, _ = execute(rules, plans=[])
    first.race_tire_usage["rules"]["max_set_laps"] = 99
    first.race_tire_usage["completed_race_laps_by_set"].clear()
    assert second.race_tire_usage["rules"] == rules == {"max_set_laps": 3}
    assert sum(second.race_tire_usage["completed_race_laps_by_set"].values()) == 8


def test_repeated_requested_compound_does_not_reset_physical_set_cap(execute):
    records = [dict(id=f"I{i}", compound="intermediate") for i in range(1, 4)]
    plans = [dict(lap=lap, compound="intermediate") for lap in (2, 4, 6)]
    result, _ = execute({"max_set_laps": 3}, records=records, plans=plans)
    assert result.status == DriverStatus.FINISHED
    history = [stint["set_id"] for stint in result.tire_set_history]
    assert len(history) > len(set(history))  # At least one actual physical set is reused.
    assert sorted(result.race_tire_usage["completed_race_laps_by_set"].values()) == [2, 3, 3]
    assert result.race_tire_usage["compliant"]


def test_reusing_a_set_does_not_satisfy_the_three_set_minimum(execute):
    records = [dict(id=f"I{i}", compound="intermediate") for i in range(1, 4)]
    records[-1]["age"] = 10  # Fresh-to-this-race I3 is less attractive until compulsory.
    plans = [dict(lap=lap, compound="intermediate") for lap in (2, 4)]
    result, _ = execute({"minimum_sets": 3}, records=records, plans=plans)
    assert result.status == DriverStatus.FINISHED
    assert [stint["set_id"] for stint in result.tire_set_history] == ["I1", "I2", "I1", "I3"]
    assert result.pit_laps == [2, 4, 8]
    assert result.race_tire_usage["used_set_ids"] == ["I1", "I2", "I3"]


def test_red_flag_unused_fittings_and_retained_set_do_not_fake_credit():
    from test_inventory_race import state_with_pool

    from f1sim.simulation.race_tire_policy import RaceTireRuleMixin

    records = [dict(id=f"I{i}", compound="intermediate") for i in range(1, 4)]
    sim, state, _ = state_with_pool(records, "I1", 0)
    sim.race_tire_rules = {"max_set_laps": 2, "minimum_sets": 3}
    sim._initialize_race_tire_usage(state)
    usage = state.race_tire_usage
    usage.begin_running()
    usage.complete_lap()
    state.tire_laps = 1
    sim._fit_inventory_tire(state, "I1", 2, "red_flag")
    assert usage.remaining_laps() == 1 and usage.used_set_ids == {"I1"}
    sim._fit_inventory_tire(state, "I2", 2, "red_flag")
    sim._fit_inventory_tire(state, "I3", 2, "red_flag")
    assert usage.used_set_ids == {"I1"}  # Neither unused fitting was run yet.
    usage.begin_running()
    usage.complete_lap()
    state.tire_laps = 1
    value = RaceTireRuleMixin._race_tire_usage_fields(state)["race_tire_usage"]
    assert value["used_set_ids"] == ["I1", "I3"]
    assert value["completed_race_laps_by_set"] == {"I1": 1, "I2": 0, "I3": 1}
    assert not value["minimum_sets_satisfied"]


def test_effective_forecast_cap_does_not_mutate_declared_inventory_budget():
    from test_inventory_race import state_with_pool

    records = [dict(id="I1", compound="intermediate", age=20, remaining_laps=10),
               dict(id="I2", compound="intermediate", remaining_laps=1)]
    sim, state, _ = state_with_pool(records, "I1", 20)
    sim.race_tire_rules = {"max_set_laps": 3}
    sim._initialize_race_tire_usage(state)
    state.race_tire_usage.begin_running()
    state.race_tire_usage.complete_lap()
    state.tire_laps = 21
    view = sim._race_tire_planning_inventory(state)
    assert view.sets["I1"].remaining_laps == 2
    assert view.sets["I2"].remaining_laps == 1
    assert state.tire_inventory.sets["I1"].remaining_laps == 10
    assert state.tire_inventory.snapshot(21)[0]["remaining_laps"] == 9


@pytest.mark.parametrize("control", ["safety_car", "vsc", "red_flag"])
def test_constraints_hold_through_observed_neutralization_and_free_restart(execute, control):
    schedule = [{"lap": 2, "control": control}]
    if control != "red_flag":
        schedule[0]["duration_laps"] = 4
    else:
        schedule[0]["action"] = "resume"
    result, _ = execute({"max_set_laps": 3, "minimum_sets": 3}, plans=[], controls=schedule)
    usage = result.race_tire_usage
    assert result.status == DriverStatus.FINISHED and result.laps_completed == 8
    assert usage["compliant"]
    assert max(usage["completed_race_laps_by_set"].values()) <= 3
    assert sum(usage["completed_race_laps_by_set"].values()) == 8
    assert len(usage["used_set_ids"]) >= 3


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_abandonment_reports_actual_use_at_decision_separate_from_countback(monkeypatch, engine):
    import test_red_flag_abandonment as abandonment

    original = RaceSimulator
    monkeypatch.setattr(abandonment, "RaceSimulator", lambda *a, **kw:
                        original(*a, **kw, race_tire_rules={"minimum_sets": 3}))
    results, _, _, _, _ = abandonment.run_case(monkeypatch, engine, after=2)
    for result in results:
        usage = result.race_tire_usage
        assert result.laps_completed == 1
        assert not usage["minimum_sets_satisfied"]
        assert usage["violations"] == ["minimum_sets"]
        assert sum(usage["completed_race_laps_by_set"].values()) >= result.laps_completed


def test_retirement_does_not_consume_an_uncompleted_cap_lap(execute):
    result, _ = execute({"max_set_laps": 2}, failure=2, plans=[], controls=[])
    assert result.status == DriverStatus.DNF and result.laps_completed == 1
    assert result.race_tire_usage["completed_race_laps_by_set"] == {"unlimited-fresh-1": 1}


def test_free_restart_cannot_retain_a_cap_exhausted_physical_set():
    from test_inventory_race import state_with_pool

    records = [dict(id="I1", compound="intermediate"), dict(id="I2", compound="intermediate")]
    sim, state, track = state_with_pool(records, "I1", 0)
    sim.race_tire_rules = {"max_set_laps": 2}
    sim._initialize_race_tire_usage(state)
    for _ in range(2):
        state.race_tire_usage.begin_running()
        state.race_tire_usage.complete_lap()
    state.tire_laps = 2
    assert sim._refit_inventory_free(state, track, Weather(track_wetness=.5, rain_intensity=.5), 2)
    assert state.tire_inventory.current_set_id == state.race_tire_usage.current_set_id == "I2"
    assert state.race_tire_usage.completed_laps_by_set == {"I1": 2, "I2": 0}
    assert state.race_tire_usage.used_set_ids == {"I1"}
    assert state.pit_stops == 0


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_automatic_three_set_dry_strategy_prices_interior_stops(monkeypatch, engine):
    sim = RaceSimulator(np.random.default_rng(11), race_tire_rules={"minimum_sets": 3})
    for name in ("process_lap", "_check_random_incident", "_check_mechanical_failure"):
        monkeypatch.setattr(sim.event_manager, name,
                            (lambda *a, **kw: []) if name == "process_lap" else
                            (lambda *a, **kw: None))
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    method = sim.simulate_race if engine == "standard" else ChronologicalRace(sim).run
    track = Track(id="monaco-like", name="Low stress race", country="T", total_laps=78,
                  base_lap_time=75, tire_stress=.2, pit_lane_delta=25)
    result = method([Driver(id="A", name="A", team_id="A", consistency=1)],
                    {"A": Car(team_id="A", team_name="A", reliability=1)},
                    track, Weather(change_probability=0), ["A"],
                    starting_tires={"A": "hard"})[0]
    assert result.status == DriverStatus.FINISHED
    assert result.race_tire_usage["compliant"]
    assert result.pit_stops == 2
    assert result.pit_laps[-2] < track.total_laps-2
    assert result.pit_laps[-1] < track.total_laps-1


def test_forward_minimum_uses_actual_running_credit_even_without_completed_wear(monkeypatch):
    from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
    from f1sim.simulation import race
    from f1sim.simulation.pit_strategy import DryPitDecision
    from f1sim.simulation.race import DriverRaceState

    sim = RaceSimulator(np.random.default_rng(1), race_tire_rules={"minimum_sets": 3})
    driver, car = Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A")
    state = DriverRaceState(driver, car, 1, current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM])
    sim._initialize_race_tire_usage(state)
    state.race_tire_usage.begin_running()  # Started running but no completed wear lap yet.
    seen = []
    monkeypatch.setattr(race, "plan_dry_stop", lambda *a, **kw:
                        seen.append(kw) or DryPitDecision(float("inf"), 0., None))
    sim._should_pit(state, [state], Track(id="T", name="T", country="T", total_laps=8,
                                        base_lap_time=90), 2, False, Weather())
    assert seen[0]["current_set_used"] is True
    assert seen[0]["minimum_new_sets"] == 2
