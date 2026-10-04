"""Native weighted team selection keeps finite stock, box queues and held-out seeds."""

from fractions import Fraction
from math import sqrt
from statistics import mean, stdev

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.rival_strategy_selection import evaluate_saved_rival_pit_plan_selection
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import native_physics
from f1sim.output import Exporter


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("objective", ["points", "win", "podium"])
def test_native_constructor_selection_preserves_actual_queue_costs_and_frozen_validation(
    tmp_path, engine, objective,
):
    def stop(lap):
        return [{"lap": lap, "compound": "hard"}]

    def metric(team):
        if objective == "points":
            return sum(row.points_awarded for row in team)
        return int(any(row.classified is True and row.position <= (1 if objective == "win" else 3)
                       for row in team))

    drivers = [Driver(id=key, name=key, team_id=team, consistency=1.)
               for key, team in (("A", "T"), ("B", "T"), ("R", "U"))]
    cars = {team: Car(team_id=team, team_name=team, reliability=1.,
                      base_pace=pace, pit_stop_avg=4.5, pit_stop_std=.1)
            for team, pace in (("T", .9), ("U", .8))}
    inventory = {driver.id: [{"id": driver.id + "-M", "compound": "medium", "age": 0},
                             {"id": driver.id + "-H", "compound": "hard", "age": 0}]
                 for driver in drivers}
    opening = {driver.id: "medium" for driver in drivers}
    candidates = {"together": {"A": stop(3), "B": stop(3)},
                  "staggered": {"A": stop(3), "B": stop(4)}}
    source = MonteCarloRunner(
        drivers, cars, Track(id="t", name="T", country="T", total_laps=8, base_lap_time=90.),
        Weather(change_probability=0.), seed=81000, race_engine=engine,
        starting_tires=opening, tire_inventory=inventory,
        pit_plans={**candidates["together"], "R": stop(4)}, tire_warmup={"hard": .5},
    ).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(source)
    before = path.read_bytes()
    rivals = {"early": {"weight": 1, "pit_plans": {"R": stop(2)}},
              "late": {"weight": 2, "pit_plans": {"R": stop(5)}}}
    assert native_physics()
    result = evaluate_saved_rival_pit_plan_selection(
        path, candidates, "together", rivals, constructor_id="T", objective=objective,
        training_simulations=2, validation_simulations=2, parallel=False)
    assert native_physics() and path.read_bytes() == before
    selection = result["selection"]
    assert selection["target_member_ids"] == ["A", "B"]
    assert selection["seed_ranges"] == {
        "source": {"first_seed": 81000, "last_seed": 81000, "trials": 1},
        "training": {"first_seed": 81001, "last_seed": 81002, "trials": 2},
        "validation": {"first_seed": 81003, "last_seed": 81004, "trials": 2},
    }
    weights = {row["name"]: Fraction(row["normalized_weight"])
               for row in selection["rival_scenarios"]}
    scores = {label: Fraction(0) for label in candidates}
    points = {label: Fraction(0) for label in candidates}
    queued = {label: 0. for label in candidates}
    for scenario, variants in result["training_results"].items():
        assert list(variants) == list(candidates)
        qualifying = variants["together"].qualifying_results
        for label, cohort in variants.items():
            assert cohort.seed == 81001 and cohort.num_simulations == 2
            assert cohort.qualifying_results == qualifying
            assert cohort.input_snapshot["tire_inventory"] == inventory
            assert cohort.input_snapshot["starting_tires"] == opening
            assert cohort.input_snapshot["pit_plans"] == {
                **candidates[label], **rivals[scenario]["pit_plans"],
            }
            for race in cohort.race_results:
                assert {row.driver_id for row in race} == {"A", "B", "R"}
                team = [row for row in race if row.driver_id in ("A", "B")]
                scores[label] += weights[scenario] * metric(team) / 2
                points[label] += weights[scenario] * sum(row.points_awarded for row in team) / 2
                for row in team:
                    assert len(row.pit_stop_details) == row.pit_stops
                    queued[label] += sum(stop["queue_time"] for stop in row.pit_stop_details)
                    assert (sum(fit["laps_used"] for fit in row.tire_set_history)
                            == row.laps_completed)
                    assert {fit["set_id"] for fit in row.tire_set_history} <= {
                        row.driver_id + "-M", row.driver_id + "-H",
                    }
                    requested = candidates[label][row.driver_id][0]
                    if row.laps_completed >= requested["lap"]:
                        assert row.pit_plan_history[0]["status"] == "executed"
                        assert row.pit_plan_history[0]["actual_set_id"] == row.driver_id + "-H"
                        assert requested["lap"] in row.pit_laps
    assert queued["together"] > 0. and queued["staggered"] == 0.
    best = max(scores.values())
    expected = next(label for label in candidates if scores[label] == best)
    assert selection["selected_label"] == expected
    for row in selection["training_score_table"]:
        assert row["mean_points"] == float(points[row["label"]])
        assert row["mean_score"] == float(scores[row["label"]])
    validation_labels = (["together"] if expected == "together" else ["together", expected])
    differences = [Fraction(0), Fraction(0)]
    for scenario, variants in result["validation_results"].items():
        assert list(variants) == validation_labels
        assert all(cohort.seed == 81003 and cohort.num_simulations == 2
                   for cohort in variants.values())
        for index in range(2):
            values = {
                label: metric([row for row in cohort.race_results[index]
                               if row.driver_id in ("A", "B")])
                for label, cohort in variants.items()
            }
            differences[index] += weights[scenario] * (values[expected] - values["together"])
    reported = selection["validation_target_metrics"]
    assert reported["mean_score_difference"] == float(mean(differences))
    if expected == "together":
        assert reported["score_difference_standard_error"] is None
    else:
        assert reported["score_difference_standard_error"] == pytest.approx(
            stdev(differences) / sqrt(2), rel=1.e-12)
