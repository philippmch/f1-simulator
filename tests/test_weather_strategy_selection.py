"""Joint race-weather/rival assumptions retain a shared qualifying experiment."""

import json
from copy import deepcopy
from fractions import Fraction
from math import sqrt
from statistics import mean, stdev

import pytest
from test_rival_strategy_selection import (
    _classification_runner,
    _controlled_run,
    _set_classification,
)

from f1sim.analysis import rival_strategy_selection as selector
from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import native_physics
from f1sim.output import Exporter
from f1sim.simulation.qualifying_weather import effective_qualifying_weather


def _stop(lap):
    return [{"lap": lap, "compound": "hard"}]


def _assumptions():
    return {
        "source": {"weight": 1, "pit_plans": {}},
        "dry": {"weight": 2, "pit_plans": {"R": []},
                "weather": {"condition": "dry", "rain_intensity": 0, "track_wetness": 0},
                "weather_schedule": []},
        "rain": {"weight": 1, "pit_plans": {"R": _stop(2)},
                 "weather": {"condition": "light_rain", "rain_intensity": .4,
                             "track_wetness": .5},
                 "weather_schedule": [{"lap": 4, "rain_intensity": .85,
                                       "condition": "heavy_rain"}]},
    }


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("objective", ["points", "win", "podium"])
@pytest.mark.parametrize("selection_method", ["weighted_mean", "minimax_regret"])
def test_native_weather_selection_freezes_inputs_qualifying_and_choice(
    tmp_path, engine, objective, selection_method,
):
    drivers = [Driver(id=name, name=name, team_id=team, consistency=1.)
               for name, team in (("A", "T"), ("B", "T"), ("R", "U"), ("C", "V"))]
    cars = {team: Car(team_id=team, team_name=team, reliability=1., pit_stop_avg=4.5,
                      pit_stop_std=.1) for team in ("T", "U", "V")}
    inventory = {driver.id: [{"id": f"{driver.id}-{compound}", "compound": compound, "age": 0}
                             for compound in ("medium", "hard", "intermediate", "wet")]
                 for driver in drivers}
    weather = Weather(track_temperature=42, air_temperature=29, humidity=.7, change_probability=0)
    schedule = [{"lap": 3, "rain_intensity": .2}]
    qualifying = {"Q1": {"condition": "cloudy", "track_wetness": .1}}
    runner = MonteCarloRunner(
        drivers, cars, Track(id="t", name="T", country="T", total_laps=6, base_lap_time=90.),
        weather, seed=81000, race_engine=engine,
        starting_tires={driver.id: "medium" for driver in drivers}, tire_inventory=inventory,
        weather_schedule=schedule, qualifying_weather=qualifying,
    )
    source = Exporter(tmp_path).export_statistics_json(runner.run(1, parallel=False))
    before = source.read_bytes()
    plans = {"together": {"A": _stop(2), "B": _stop(2)},
             "staggered": {"A": _stop(2), "B": _stop(3)}}
    assumptions = _assumptions()
    frozen_request = deepcopy(assumptions)
    assert native_physics()
    outcome = selector.evaluate_saved_rival_pit_plan_selection(
        source, plans, "together", assumptions, constructor_id="T", objective=objective,
        training_simulations=2, validation_simulations=2,
        selection_method=selection_method,
    )
    assert native_physics() and source.read_bytes() == before and assumptions == frozen_request
    selection = outcome["selection"]
    assert selection["method"] == (
        "minimax_regret_weather_and_rival_training_then_disjoint_seed_validation"
        if selection_method == "minimax_regret" else
        "weighted_weather_and_rival_training_then_disjoint_seed_validation"
    )
    expected_qualifying = effective_qualifying_weather(weather, qualifying)
    assert selection["frozen_qualifying_weather"] == expected_qualifying
    scenarios = {row["name"]: row for row in selection["rival_scenarios"]}
    assert scenarios["source"]["weather_schedule"] == schedule
    assert scenarios["dry"]["weather_schedule"] == []
    assert scenarios["rain"]["weather_schedule"] == assumptions["rain"]["weather_schedule"]
    for scenario in scenarios.values():
        assert scenario["weather"]["track_temperature"] == 42
        assert scenario["weather"]["air_temperature"] == 29
        assert scenario["weather"]["humidity"] == .7

    def score(race):
        team = [row for row in race if row.driver_id in ("A", "B")]
        if objective == "points":
            return sum(row.points_awarded for row in team)
        return int(any(row.classified is True and row.position <= (1 if objective == "win" else 3)
                       for row in team))

    weighted_scores = {label: [Fraction(0)] * 2 for label in plans}
    scenario_means = {}
    shared_qualifying = next(iter(outcome["training_results"].values()))[
        "together"
    ].qualifying_results
    for name, variants in outcome["training_results"].items():
        weight = Fraction(scenarios[name]["normalized_weight"])
        scenario_means[name] = {}
        for label, results in variants.items():
            scenario_means[name][label] = sum(
                (Fraction(score(race)) for race in results.race_results), Fraction(),
            ) / 2
            assert results.qualifying_results == shared_qualifying
            assert results.input_snapshot["weather"] == scenarios[name]["weather"]
            assert results.input_snapshot.get("weather_schedule", []) == (
                scenarios[name]["weather_schedule"]
            )
            assert results.input_snapshot["qualifying_weather"] == expected_qualifying
            assert results.input_snapshot["tire_inventory"] == inventory
            for index, race in enumerate(results.race_results):
                assert {row.driver_id for row in race} == {driver.id for driver in drivers}
                weighted_scores[label][index] += weight * score(race)
                for row in race:
                    assert sum(fit["laps_used"] for fit in row.tire_set_history) == (
                        row.laps_completed
                    )
                    assert len(row.pit_stop_details) == row.pit_stops
    means = {label: mean(values) for label, values in weighted_scores.items()}
    if selection_method == "minimax_regret":
        maximum_regrets = {
            label: max(max(values.values()) - values[label] for values in scenario_means.values())
            for label in plans
        }
        winner = next(label for label in plans
                      if maximum_regrets[label] == min(maximum_regrets.values()))
        assert {row["label"]: row["maximum_regret"]
                for row in selection["training_regret_table"]} == {
            label: float(value) for label, value in maximum_regrets.items()
        }
    else:
        winner = next(label for label in plans if means[label] == max(means.values()))
    assert selection["selected_label"] == winner
    for row in selection["training_score_table"]:
        assert row["mean_score"] == float(means[row["label"]])
    differences = [Fraction(0)] * 2
    valid_labels = ["together"] if winner == "together" else ["together", winner]
    for name, variants in outcome["validation_results"].items():
        assert list(variants) == valid_labels
        assert all(results.seed == 81003 for results in variants.values())
        for results in variants.values():
            assert results.input_snapshot["weather"] == scenarios[name]["weather"]
            assert results.input_snapshot["qualifying_weather"] == expected_qualifying
            assert results.qualifying_results == outcome["validation_results"]["source"][
                "together"
            ].qualifying_results
        for index in range(2):
            differences[index] += Fraction(scenarios[name]["normalized_weight"]) * (
                score(variants[winner].race_results[index])
                - score(variants["together"].race_results[index])
            )
    metrics = selection["validation_target_metrics"]
    assert metrics["mean_score_difference"] == float(mean(differences))
    assert metrics["score_difference_standard_error"] == (
        None if winner == "together" else pytest.approx(stdev(differences) / sqrt(2))
    )


def test_weighted_weather_winner_changes_with_weights_and_stays_frozen(monkeypatch):
    runner = _classification_runner()

    def outcomes(variant, result):
        planned = bool((variant.pit_plans or {}).get("A"))
        wet = variant.weather.track_wetness > .3
        for race in result.race_results:
            points = (25 if planned == wet else 0) if variant.base_seed == 72 else (
                0 if planned else 25
            )
            _set_classification(race, {"A": points})

    _controlled_run(monkeypatch, outcomes)
    plans = {"reference": None, "planned": _stop(2)}
    for wet_weight, winner in ((1, "reference"), (3, "planned")):
        prepared = selector.prepare_rival_pit_plan_selection(
            runner, 1, plans, "reference",
            {"dry": {"weight": 2, "pit_plans": {}},
             "wet": {"weight": wet_weight, "pit_plans": {},
                     "weather": {"track_wetness": .5, "rain_intensity": .4}}},
            driver_id="A", objective="win", training_simulations=2, validation_simulations=2,
        )
        selected = selector.evaluate_prepared_rival_pit_plan_selection(prepared)["selection"]
        assert selected["selected_label"] == winner
        assert selected["validation_target_metrics"]["mean_score_difference"] == (
            0 if winner == "reference" else -1
        )


@pytest.mark.parametrize("override", [
    {"weather": {"unknown": 1}}, {"weather": {"rain_intensity": True}},
    {"weather": {"rain_intensity": "0.5"}}, {"weather": {"rain_intensity": float("nan")}},
    {"weather": "wet"}, {"weather": {"condition": "hail"}},
    {"weather_schedule": "rain"}, {"weather_schedule": [{"lap": 1, "rain_intensity": .5}]},
])
def test_invalid_weather_assumptions_rejected_before_loading_or_trials(override):
    with pytest.raises(ValueError, match="Invalid race weather"):
        selector.evaluate_saved_rival_pit_plan_selection(
            "missing.json", {"reference": None, "planned": _stop(2)}, "reference",
            {"invalid": {"weight": 1, "pit_plans": {}, **override}},
            driver_id="A", training_simulations=1, validation_simulations=1,
        )


def test_late_invalid_weather_schedule_spends_no_trials(monkeypatch):
    runner = _classification_runner()
    monkeypatch.setattr(MonteCarloRunner, "run", lambda *args, **kwargs: pytest.fail("trial work"))
    with pytest.raises(ValueError, match="through the scheduled distance"):
        selector.prepare_rival_pit_plan_selection(
            runner, 1, {"reference": None, "planned": _stop(2)}, "reference",
            {"good": {"weight": 1, "pit_plans": {}, "weather": {"track_wetness": .5}},
             "bad": {"weight": 1, "pit_plans": {},
                     "weather_schedule": [{"lap": 100, "rain_intensity": .2}]}},
            driver_id="A", training_simulations=1, validation_simulations=1,
        )


@pytest.mark.parametrize("field,message", [
    ("weather", "different frozen race weather"),
    ("qualifying_weather", "different frozen qualifying weather"),
    ("track", "incompatible non-plan inputs"),
])
def test_frozen_weather_inputs_are_checked_against_actual_phase_evidence(
    monkeypatch, field, message,
):
    def outcomes(runner, result):
        if runner.weather.track_wetness > .3:
            if field == "track":
                result.input_snapshot["track"]["base_lap_time"] += 1
            elif field == "qualifying_weather":
                result.input_snapshot[field]["Q1"]["track_wetness"] = .25
            else:
                result.input_snapshot[field]["track_wetness"] = .25

    _controlled_run(monkeypatch, outcomes)
    prepared = selector.prepare_rival_pit_plan_selection(
        _classification_runner(), 1, {"reference": None, "same": None}, "reference",
        {"dry": {"weight": 1, "pit_plans": {}},
         "wet": {"weight": 1, "pit_plans": {}, "weather": {"track_wetness": .5}}},
        driver_id="A", training_simulations=1, validation_simulations=1,
    )
    with pytest.raises(ValueError, match=message):
        selector.evaluate_prepared_rival_pit_plan_selection(prepared)


def test_preparation_freezes_partial_weather_and_schedule_objects():
    assumptions = {"wet": {"weight": 1, "pit_plans": {},
                           "weather": {"track_wetness": .5},
                           "weather_schedule": [{"lap": 3, "rain_intensity": .2}]}}
    prepared = selector.prepare_rival_pit_plan_selection(
        _classification_runner(), 1, {"reference": None, "same": None}, "reference",
        assumptions, driver_id="A", training_simulations=1, validation_simulations=1,
    )
    frozen = json.dumps(prepared["weather_contexts"], sort_keys=True)
    assumptions["wet"]["weather"]["track_wetness"] = .9
    assumptions["wet"]["weather_schedule"][0]["rain_intensity"] = .9
    assert json.dumps(prepared["weather_contexts"], sort_keys=True) == frozen


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_weather_cases_produce_identical_serial_and_process_outcomes(engine):
    runner = _classification_runner()
    runner.race_engine = engine
    assumptions = {
        "dry": {"weight": 1, "pit_plans": {}, "weather_schedule": []},
        "rain": {"weight": 2, "pit_plans": {}, "weather": {"track_wetness": .5},
                 "weather_schedule": [{"lap": 2, "rain_intensity": .8}]},
    }
    def evaluate(parallel):
        prepared = selector.prepare_rival_pit_plan_selection(
            runner, 1, {"reference": None, "planned": _stop(2)}, "reference", assumptions,
            driver_id="A", objective="win", training_simulations=2, validation_simulations=2,
        )
        return selector.evaluate_prepared_rival_pit_plan_selection(
            prepared, parallel=parallel, max_workers=2,
        )

    serial, parallel = evaluate(False), evaluate(True)
    assert serial["selection"] == parallel["selection"]
    for phase in ("training_results", "validation_results"):
        for case, variants in serial[phase].items():
            for label, result in variants.items():
                replay = parallel[phase][case][label]
                assert replay.race_results == result.race_results
                assert replay.qualifying_results == result.qualifying_results
                assert replay.input_snapshot == result.input_snapshot
