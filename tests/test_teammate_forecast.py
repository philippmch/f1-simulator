"""Causal history, forecast allocation, proper scoring and native replay."""

import itertools
import json
from copy import deepcopy
from dataclasses import asdict
from math import prod, sqrt

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner, wilson_interval
from f1sim.analysis.race_probability_scores import score_winner_counts, summarize_winner_counts
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.analysis.teammate_forecast import (
    build_teammate_allocation,
    score_teammate_forecast,
    teammate_winner_forecast,
    validate_teammate_allocation,
)
from f1sim.data.current import CurrentSeasonDataLoader
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter


def history():
    return [
        {"round": number, "driver_id": driver, "team_id": "a", "points": points}
        for number in (1, 2)
        for driver, points in (("AA", 25.0), ("BB", 0.0))
    ]


def allocation(rows=None):
    return build_teammate_allocation(
        {"AA": "a", "BB": "a", "CC": "c"}, history() if rows is None else rows, cutoff_round=2
    )


def test_constructor_mass_raw_counts_and_intervals_are_preserved():
    native = summarize_winner_counts({"AA": 1, "BB": 5, "CC": 3}, 1)
    before = deepcopy(native)
    frozen = allocation()
    result = teammate_winner_forecast(native, frozen)
    assert result["drivers"]["AA"]["probability"] == pytest.approx(0.45)
    assert result["drivers"]["BB"]["probability"] == pytest.approx(0.15)
    assert result["drivers"]["CC"]["probability"] == 0.3
    assert result["no_classified_winner_probability"] == 0.1
    assert result["drivers"]["AA"]["native_win_count"] == 1
    assert result["drivers"]["AA"]["native_probability"] == 0.1
    interval = wilson_interval(6, 10)
    assert result["drivers"]["AA"]["mc_sampling_interval_95"] == pytest.approx(
        {key: 0.75 * value / 100 for key, value in interval.items()}
    )
    assert native == before
    result["allocation"]["teams"]["a"]["weights"]["AA"] = 0.0
    assert frozen["teams"]["a"]["weights"]["AA"] == 0.75


def test_sparse_missing_conflicting_and_transferred_history_falls_back():
    rows = history()
    assert allocation(rows[:2])["teams"]["a"]["status"] == "native_fallback"
    assert allocation(rows + rows)["teams"]["a"]["entries"]["AA"] == 2
    bad = deepcopy(rows)
    bad[0]["points"] = None
    assert allocation(bad)["teams"]["a"]["reason"] == "invalid_or_conflicting_points"
    conflict = deepcopy(rows[0])
    conflict["points"] = 18.0
    assert allocation(rows + [conflict])["teams"]["a"]["status"] == "native_fallback"
    for row in bad:
        if row["driver_id"] == "AA":
            row["team_id"] = "old_team"
    assert allocation(bad)["teams"]["a"]["status"] == "native_fallback"


@pytest.mark.parametrize("points", [True, False, "25", -1.0, float("nan"), float("inf")])
def test_unusable_points_do_not_become_zero_evidence(points):
    rows = history()
    rows[0]["points"] = points
    assert allocation(rows)["teams"]["a"]["status"] == "native_fallback"


@pytest.mark.parametrize("number", [True, 0, -1, 3, "1"])
def test_later_or_unidentified_rounds_are_rejected(number):
    rows = history()
    rows[0]["round"] = number
    with pytest.raises(ValueError, match="earlier"):
        allocation(rows)


def test_resealed_derived_weights_and_bool_coverage_are_not_trusted():
    frozen = allocation()
    frozen["teams"]["a"]["weights"]["AA"] = 0.5
    with pytest.raises(ValueError, match="history"):
        validate_teammate_allocation(frozen)
    frozen = allocation(history()[:2])
    frozen["teams"]["a"]["entries"]["AA"] = True
    with pytest.raises(ValueError, match="history"):
        validate_teammate_allocation(frozen)
    with pytest.raises(ValueError, match="constructors"):
        validate_teammate_allocation(allocation(), {"AA": "b", "BB": "a", "CC": "c"})


@pytest.mark.parametrize("winner", ["AA", "BB", "CC", None])
def test_identity_fallback_score_and_bias_equal_native_scores(winner):
    native = summarize_winner_counts({"AA": 4, "BB": 2, "CC": 1}, 1)
    score = score_teammate_forecast(native, allocation([]), winner)
    expected = score_winner_counts({"AA": 4, "BB": 2, "CC": 1}, 1, winner)
    assert score["brier_score"] == expected["brier_score"]
    for field in ("estimated_empirical_score_bias", "adjusted_brier_score", "mc_standard_error"):
        assert score["mc_adjustment"][field] == pytest.approx(expected["mc_adjustment"][field])


def test_linear_score_correction_and_sampling_error_against_exact_enumeration():
    # Independent enumeration of complete four-draw samples from a known law.
    true = {"AA": 0.5, "BB": 0.25, "CC": 0.25}
    native = summarize_winner_counts({"AA": 2, "BB": 1, "CC": 1}, 0)
    score = score_teammate_forecast(native, allocation(), "AA")
    losses = []
    for sample in itertools.product(true, repeat=4):
        probability = prod(true[key] for key in sample)
        observed = summarize_winner_counts({key: sample.count(key) for key in true}, 0)
        losses.append(
            (
                probability,
                score_teammate_forecast(observed, allocation(), "AA")["mc_adjustment"][
                    "adjusted_brier_score"
                ],
            )
        )
    expected = sum(p * loss for p, loss in losses)
    true_brier = (0.75 * 0.75 - 1) ** 2 + (0.75 * 0.25) ** 2 + 0.25**2
    assert expected == pytest.approx(true_brier)
    variance = sum(p * (loss - expected) ** 2 for p, loss in losses)
    assert score["mc_adjustment"]["mc_standard_error"] == pytest.approx(sqrt(variance))


def test_single_trial_has_no_finite_ensemble_correction():
    score = score_teammate_forecast(
        summarize_winner_counts({"AA": 1, "BB": 0, "CC": 0}, 0), allocation(), "AA"
    )
    assert score["mc_adjustment"]["status"] == "unavailable"
    assert score["mc_adjustment"]["estimated_empirical_score_bias"] is None


def test_live_point_collection_filters_target_rows_and_preserves_old_constructor_identity():
    loader = CurrentSeasonDataLoader()
    drivers = [
        Driver(id=key, name=key, team_id=team)
        for key, team in (("AA", "a"), ("BB", "a"), ("CC", "c"))
    ]
    rows = []
    for observation in history():
        rows.append(
            {
                "round": observation["round"],
                "points": str(observation["points"]),
                "Driver": {"code": observation["driver_id"]},
                "Constructor": {"constructorId": "a", "name": "A"},
            }
        )
    rows += [
        {
            "round": 3,
            "points": "1000",
            "Driver": {"code": "BB"},
            "Constructor": {"constructorId": "a", "name": "A"},
        },
        {
            "round": 1,
            "points": "1000",
            "Driver": {"code": "CC"},
            "Constructor": {"constructorId": "old_team", "name": "Old Team"},
        },
    ]
    loader._event_for_race = lambda *a: {"round": 3}
    loader._season_data = lambda *a: (rows, [])
    loader.get_event_schedule = lambda *a: [{"round": number} for number in (1, 2, 3)]
    frozen = loader.get_winner_allocation(loader.current_year, 3, drivers)
    assert frozen["teams"]["a"]["weights"] == {"AA": 0.75, "BB": 0.25}
    assert frozen["teams"]["c"]["status"] == "native_fallback"
    assert all(row["round"] < 3 for row in frozen["prior_race_points"])


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_real_workers_old_inputs_exports_and_second_trial_replay(tmp_path, engine):
    drivers = [
        Driver(id=key, name=key, team_id=team)
        for key, team in (("AA", "a"), ("BB", "a"), ("CC", "c"))
    ]
    cars = {team: Car(team_id=team, team_name=team) for team in ("a", "c")}
    track = Track(id="test", name="Test", country="Test", total_laps=3, base_lap_time=80.0)
    kwargs = dict(seed=42, race_engine=engine)
    plain = MonteCarloRunner(drivers, cars, track, Weather(), **kwargs).run(2, parallel=False)
    calibrated = MonteCarloRunner(
        drivers, cars, track, Weather(), winner_allocation=allocation(), **kwargs
    ).run(2, parallel=True, max_workers=2)
    assert calibrated.race_results == plain.race_results
    assert calibrated.qualifying_results == plain.qualifying_results
    assert calibrated.weather_histories == plain.weather_histories
    assert asdict(calibrated.event_stats) == asdict(plain.event_stats)
    assert calibrated.input_snapshot == plain.input_snapshot
    assert plain.get_winner_forecast() is None
    assert calibrated.get_winner_forecast()["policy"] == "teammate_race_points_v1"
    native_path = Exporter(tmp_path).export_statistics_json(plain, "native.json")
    path = Exporter(tmp_path).export_statistics_json(calibrated, "calibrated.json")
    saved = json.loads(path.read_bytes())
    assert saved["winner_forecast"] == calibrated.get_winner_forecast()
    assert saved["driver_statistics"]["AA"]["wins"] == plain.driver_stats["AA"].wins
    replayed = replay_saved_simulation(path, 2)
    old_replay = replay_saved_simulation(native_path, 2)
    assert replayed.race_results[0] == plain.race_results[1] == old_replay.race_results[0]
    assert replayed.winner_allocation == allocation()
    assert old_replay.winner_allocation is None
    report = Exporter(tmp_path).export_report_html(calibrated).read_text(encoding="utf-8")
    assert 'id="winner-forecast"' in report and "Simulated wins" in report
