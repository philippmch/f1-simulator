"""Every strategy phase must inherit frozen event constraints without weakening."""

from copy import deepcopy

import pytest

from f1sim.analysis import rival_strategy_selection as rival
from f1sim.analysis import strategy_comparison as comparison
from f1sim.analysis import strategy_selection as selection
from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.tire_rules import RACE_TIRE_RULES_POLICY


def runner(rules=None):
    drivers = [Driver(id=code, name=code, team_id=code) for code in ("A", "B")]
    return MonteCarloRunner(
        drivers, {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers},
        Track(id="test", name="Test", country="T", total_laps=6, base_lap_time=90),
        Weather(change_probability=0), seed=31,
        **({"race_tire_rules": rules} if rules else {}),
    )


PLANS = {"reference": None, "no_elective_stop": []}
SCENARIOS = {
    "normal": {"weight": 1, "pit_plans": {"B": None}},
    "hot": {"weight": 1, "pit_plans": {"B": []}, "weather": {"track_temperature": 45}},
}


@pytest.mark.parametrize("with_rivals", [False, True])
def test_all_prepared_phases_keep_strictest_copied_rules_without_running(with_rivals, monkeypatch):
    base = runner({"max_set_laps": 3, "minimum_sets": 2})
    requested = {"max_set_laps": 4, "minimum_sets": 3}
    expected = {"max_set_laps": 3, "minimum_sets": 3}
    monkeypatch.setattr(MonteCarloRunner, "run", lambda *a, **k: pytest.fail("ran during prepare"))
    if with_rivals:
        prepared = rival.prepare_rival_pit_plan_selection(
            base, 1, PLANS, "reference", SCENARIOS, driver_id="A",
            training_simulations=2, validation_simulations=2, race_tire_rules=requested,
        )
        phases = [variants for phase in ("training_runners", "validation_candidates")
                  for variants in prepared[phase].values()]
    else:
        prepared = selection.prepare_pit_plan_selection(
            base, 1, PLANS, "reference", driver_id="A",
            training_simulations=2, validation_simulations=2, race_tire_rules=requested,
        )
        phases = [prepared[phase] for phase in ("training_runners", "validation_candidates")]
    requested["minimum_sets"] = 99
    assert prepared["race_tire_rules"] == expected
    assert base.race_tire_rules == {"max_set_laps": 3, "minimum_sets": 2}
    variants = [variant for phase in phases for variant in phase.values()]
    assert all(variant.race_tire_rules == expected for variant in variants)
    variants[0].race_tire_rules["minimum_sets"] = 4
    assert prepared["race_tire_rules"] == expected
    assert all(variant.race_tire_rules == expected for variant in variants[1:])


@pytest.mark.parametrize("with_rivals", [False, True])
def test_selected_plan_provenance_reports_the_frozen_constraints(with_rivals, monkeypatch):
    base = runner({"max_set_laps": 3})
    module = rival if with_rivals else selection
    if with_rivals:
        prepared = rival.prepare_rival_pit_plan_selection(
            base, 1, PLANS, "reference", {"normal": SCENARIOS["normal"]},
            driver_id="A", training_simulations=1, validation_simulations=1,
        )
    else:
        prepared = selection.prepare_pit_plan_selection(
            base, 1, PLANS, "reference", driver_id="A",
            training_simulations=1, validation_simulations=1,
        )
    def phase_results(variants, count, **kwargs):
        assert all(v.race_tire_rules == {"max_set_laps": 3} for v in variants.values())
        return {label: object() for label in variants}

    def phase_points(results, labels, reference_label, **kwargs):
        return {label: [0] * kwargs["expected_count"] for label in labels}, {}

    monkeypatch.setattr(module, "_run_variants", phase_results)
    monkeypatch.setattr(module, "_phase_points", phase_points)
    if with_rivals:
        monkeypatch.setattr(rival, "_check_scenario_cohort", lambda *a, **k: None)
        result = rival.evaluate_prepared_rival_pit_plan_selection(prepared)
    else:
        result = selection.evaluate_prepared_pit_plan_selection(prepared)
    assert result["selection"]["race_tire_rules"] == {"max_set_laps": 3}
    assert result["selection"]["race_tire_rules_policy"] == RACE_TIRE_RULES_POLICY
    prepared["race_tire_rules"]["max_set_laps"] = 9
    assert result["selection"]["race_tire_rules"] == {"max_set_laps": 3}


@pytest.mark.parametrize("prepare", [selection.prepare_pit_plan_selection,
                                     rival.prepare_rival_pit_plan_selection])
def test_invalid_additional_rules_fail_before_reading_the_runner(prepare):
    args = (None, 1, PLANS, "reference")
    if prepare is rival.prepare_rival_pit_plan_selection:
        args += (SCENARIOS,)
    with pytest.raises(ValueError, match="positive integer"):
        prepare(*args, driver_id="A", race_tire_rules={"max_set_laps": False})


def test_clone_inherits_rules_and_cannot_weaken_them():
    base = runner({"max_set_laps": 3, "minimum_sets": 3})
    cloned = comparison._runner_variant(base, race_tire_rules={"max_set_laps": 25})
    assert cloned.race_tire_rules == base.race_tire_rules
    cloned.race_tire_rules["max_set_laps"] = 2
    assert base.race_tire_rules["max_set_laps"] == 3


@pytest.mark.parametrize("kind", ["engines", "starting_tires"])
def test_saved_comparison_direct_constructors_keep_rules(kind, monkeypatch):
    base = runner({"max_set_laps": 3, "minimum_sets": 3})
    before = deepcopy(base.race_tire_rules)
    monkeypatch.setattr(comparison, "_load_saved_runner", lambda *a: (base, 1))
    monkeypatch.setattr(MonteCarloRunner, "run", lambda self, *a, **k: self.race_tire_rules.copy())
    if kind == "engines":
        results = comparison.compare_saved_race_engines("unused.json", num_simulations=1)
    else:
        results = comparison.compare_saved_starting_tires(
            "unused.json", "A", compounds=("automatic", "medium"), num_simulations=1,
        )
    assert results and all(rules == before for rules in results.values())
    assert base.race_tire_rules == before
