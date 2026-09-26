"""Held-out race probabilities keep target outcomes out of forecast inputs."""

import copy
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from f1sim.analysis import race_probability_evaluation as evaluation
from f1sim.analysis.holdout_folds import assemble_holdout_fold
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader

YEAR = datetime.now(timezone.utc).year


def load_cli_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "evaluate_race_probabilities.py"
    spec = importlib.util.spec_from_file_location("evaluate_race_probabilities_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(round_number, code, team, q1, position):
    return {
        "round": round_number,
        "Driver": {
            "driverId": code.lower(), "code": code,
            "givenName": code, "familyName": "Racer",
        },
        "Constructor": {"constructorId": team, "name": team},
        "Q1": q1,
        "position": str(position),
        "status": "Finished",
        "FastestLap": {
            "AverageSpeed": {"speed": str(230 - position)},
            "Time": {"time": "1:00.000"},
        },
    }


def standings(round_number):
    return {"MRData": {"total": "2", "offset": "0", "StandingsTable": {
        "season": str(YEAR), "round": str(round_number),
        "StandingsLists": [{"season": str(YEAR), "round": str(round_number),
        "ConstructorStandings": [
            {"Constructor": {"constructorId": "a", "name": "a"}, "points": "40"},
            {"Constructor": {"constructorId": "b", "name": "b"}, "points": "20"},
        ]}]}}}


@pytest.fixture
def evaluation_data(monkeypatch):
    loader = CurrentSeasonDataLoader(current_year=YEAR, http_getter=lambda *a, **k: {})
    events = [
        {"round": n, "race": f"Race {n}", "circuit_id": f"test{n}", "completed": True}
        for n in (1, 2, 3)
    ]
    qualifying = [
        row(n, code, team, f"1:{time:02}.000", pos)
        for n in (1, 2, 3)
        for pos, (code, team, time) in enumerate([
            ("A1", "a", 10), ("A2", "a", 11), ("B1", "b", 12), ("B2", "b", 13),
        ], 1)
    ]
    results = copy.deepcopy(qualifying)

    def fetch(url):
        round_number = int(url.split(f"/{YEAR}/")[1].split("/")[0])
        return standings(round_number)

    monkeypatch.setattr(loader, "_fetch_json", fetch)
    monkeypatch.setattr(loader, "_season_data", lambda _: copy.deepcopy((results, qualifying)))
    monkeypatch.setattr(loader, "get_event_schedule", lambda _: copy.deepcopy(events))
    monkeypatch.setattr(
        loader, "_standings", lambda *a, **k: pytest.fail("live driver standings"),
    )
    monkeypatch.setattr(
        loader, "_round_data", lambda *a, **k: pytest.fail("live round data"),
    )
    monkeypatch.setattr(
        loader, "get_official_roster", lambda *a, **k: pytest.fail("live roster"),
    )
    return loader, results, qualifying, events


class _FakeRunner:
    calls = []

    def __init__(self, *, drivers, cars, track, weather, seed, race_engine, rng_policy):
        self.driver_ids = [driver.id for driver in drivers]
        self.ranked_ids = [
            driver.id for driver in sorted(
                drivers, key=lambda driver: (-driver.skill_rating, driver.id),
            )
        ]
        self.seed = seed
        self.race_engine = race_engine
        self.rng_policy = rng_policy
        self._call = {
            "driver_ids": self.driver_ids,
            "ranked_ids": self.ranked_ids,
            "car_ids": list(cars),
            "drivers": [driver.model_dump() for driver in drivers],
            "seed": seed,
            "race_engine": race_engine,
            "rng_policy": rng_policy,
            "track_laps": track.total_laps,
            "weather": weather.model_dump(),
        }
        self.__class__.calls.append(self._call)

    def run(self, *, num_simulations, parallel):
        assert parallel is False
        races = []
        winner_ids = []
        for trial in range(num_simulations):
            winner_id = self.ranked_ids[trial % len(self.ranked_ids)]
            winner_ids.append(winner_id)
            order = [winner_id] + [
                driver_id for driver_id in self.driver_ids if driver_id != winner_id
            ]
            positions = {driver_id: index + 1 for index, driver_id in enumerate(order)}
            races.append([
                SimpleNamespace(
                    driver_id=driver_id,
                    position=positions[driver_id],
                    classified=True,
                    status="Finished",
                )
                for driver_id in self.driver_ids
            ])
        self._call["winner_ids"] = winner_ids
        return SimpleNamespace(
            race_results=races,
            input_snapshot={"drivers": copy.deepcopy(self._call["drivers"])},
        )


@pytest.fixture
def fake_runner(monkeypatch):
    _FakeRunner.calls = []
    monkeypatch.setattr(evaluation, "MonteCarloRunner", _FakeRunner)
    return _FakeRunner.calls


def run_target(loader, **kwargs):
    return evaluation.evaluate_race_probabilities(
        loader, YEAR, target_race=2, **kwargs,
    )


def test_target_and_future_result_mutations_change_labels_not_forecasts(
    evaluation_data, fake_runner,
):
    loader, results, qualifying, _events = evaluation_data
    before = run_target(loader, trials=4)
    for rows in (results, qualifying):
        for record in rows:
            if record["round"] >= 2:
                record["Q1"] = "2:00.000"
                record["position"] = "22"
                record["FastestLap"] = {"AverageSpeed": {"speed": "999"},
                                        "Time": {"time": "0:01.000"}}
    # Keep target entrant identities and result coverage fixed while changing
    # only observed target labels; future performance is also heavily changed.
    after = run_target(loader, trials=4)

    first, second = before["folds"][0], after["folds"][0]
    assert first["forecast"] == second["forecast"]
    assert first["training"] == second["training"]
    assert first["coverage"] == second["coverage"]
    assert first["simulation_inputs"] == second["simulation_inputs"]
    assert second["observed_outcome"] == {
        "status": "excluded", "reason": "missing_position_one_result",
    }
    assert fake_runner[0]["seed"] == fake_runner[1]["seed"]


def test_prior_eligible_evidence_changes_simulated_model_inputs(
    evaluation_data, monkeypatch, fake_runner,
):
    loader, results, qualifying, _events = evaluation_data
    original_assemble = assemble_holdout_fold
    captured = []

    def capture_assembly(*args, **kwargs):
        assembled, observations = original_assemble(*args, **kwargs)
        captured.append({
            "drivers": [driver.model_dump() for driver in assembled.drivers],
            "cars": {key: value.model_dump() for key, value in assembled.cars.items()},
        })
        return assembled, observations

    monkeypatch.setattr(evaluation, "assemble_holdout_fold", capture_assembly)
    first = run_target(loader, trials=5)
    for record in qualifying:
        if record["round"] == 1 and record["Driver"]["code"] == "A1":
            record["Q1"] = "0:45.000"
    second = run_target(loader, trials=5)

    assert captured[0] != captured[1]
    assert first["folds"][0]["simulation_inputs"] != second["folds"][0]["simulation_inputs"]


def test_missing_target_q1_still_keeps_entrant_in_forecast(evaluation_data, fake_runner):
    loader, _results, qualifying, _events = evaluation_data
    for record in qualifying:
        if record["round"] == 2 and record["Driver"]["code"] == "A1":
            record["Q1"] = None

    report = run_target(loader, trials=4)
    fold = report["folds"][0]
    assert fold["status"] == "scored"
    assert fold["entrant_ids"] == ["A1", "A2", "B1", "B2"]
    assert set(fold["forecast"]["drivers"]) == set(fold["entrant_ids"])
    assert fake_runner[0]["driver_ids"] == fold["entrant_ids"]


def test_target_feed_permutation_does_not_change_seed_assignment_or_forecast(
    evaluation_data, fake_runner, monkeypatch,
):
    loader, results, qualifying, _events = evaluation_data
    before = run_target(loader, trials=4)
    qualifying[:] = list(reversed(qualifying))
    results[:] = list(reversed(results))
    original_assemble = assemble_holdout_fold

    def permuted_assembly(*args, **kwargs):
        assembled, observations = original_assemble(*args, **kwargs)
        assembled.drivers.reverse()
        assembled.cars = dict(reversed(list(assembled.cars.items())))
        return assembled, observations

    monkeypatch.setattr(evaluation, "assemble_holdout_fold", permuted_assembly)
    after = run_target(loader, trials=4)

    assert before["folds"][0]["forecast"] == after["folds"][0]["forecast"]
    assert fake_runner[0]["driver_ids"] == fake_runner[1]["driver_ids"]
    assert fake_runner[0]["car_ids"] == fake_runner[1]["car_ids"]


def test_event_seed_and_trial_prefix_are_selection_independent(evaluation_data, fake_runner):
    loader, _results, _qualifying, _events = evaluation_data
    selected = run_target(loader, trials=2, seed=123)
    all_events = evaluation.evaluate_race_probabilities(
        loader, YEAR, all_targets=True, trials=2, seed=123,
    )
    longer = run_target(loader, trials=4, seed=123)

    selected_fold = selected["folds"][0]
    all_fold = next(fold for fold in all_events["folds"] if fold["round"] == 2)
    longer_fold = longer["folds"][0]
    assert selected_fold["simulation"]["event_seed"] == all_fold["simulation"]["event_seed"]
    assert selected_fold["simulation"]["event_seed"] == longer_fold["simulation"]["event_seed"]
    assert selected_fold["forecast"]["drivers"] == all_fold["forecast"]["drivers"]

    ids = selected_fold["entrant_ids"]
    event_call = next(
        call for call in fake_runner
        if call["seed"] == selected_fold["simulation"]["event_seed"]
    )
    expected_first_two = event_call["winner_ids"][:2]
    expected_counts = {driver_id: expected_first_two.count(driver_id) for driver_id in ids}
    assert {driver_id: item["wins"] for driver_id, item
            in selected_fold["forecast"]["drivers"].items()} == expected_counts
    assert all(
        longer_fold["forecast"]["drivers"][driver_id]["wins"] >= wins
        for driver_id, wins in expected_counts.items()
    )


def test_insufficient_coverage_is_reported_without_partial_scoring(evaluation_data, fake_runner):
    loader, results, _qualifying, _events = evaluation_data
    target_rows = [record for record in results if record["round"] == 2]
    results[:] = [record for record in results if record["round"] != 2]
    results.extend(target_rows[:-1])

    report = run_target(loader, trials=2)
    fold = report["folds"][0]
    assert fold["status"] == "excluded"
    assert fold["reason"] == "insufficient_target_coverage"
    assert fold["coverage"] == {
        "qualifying_entrants": 4,
        "result_entrants": 3,
        "matched_result_entrants": 3,
        "expected_result_entrants": 4,
    }
    assert fold["forecast"] is None
    assert report["aggregate"]["mean_brier_score"] is None
    assert fake_runner == []


def test_identical_duplicate_position_one_feed_row_is_counted_once(evaluation_data, fake_runner):
    loader, results, _qualifying, _events = evaluation_data
    winner = next(
        record for record in results
        if record["round"] == 2 and record["position"] == "1"
    )
    results.append(copy.deepcopy(winner))

    report = run_target(loader, trials=2)
    assert report["folds"][0]["status"] == "scored"
    assert report["folds"][0]["observed_outcome"]["winner_id"] == "A1"


def test_conflicting_duplicate_target_result_identity_aborts(evaluation_data, fake_runner):
    loader, results, _qualifying, _events = evaluation_data
    winner = next(
        record for record in results
        if record["round"] == 2 and record["position"] == "1"
    )
    conflict = copy.deepcopy(winner)
    conflict["status"] = "Retired"
    results.append(conflict)

    with pytest.raises(CurrentSeasonDataError, match="Conflicting evaluation driver records"):
        run_target(loader, trials=2)
    assert fake_runner == []


def test_explicit_target_without_results_is_not_treated_as_completed(
    evaluation_data, fake_runner,
):
    loader, results, _qualifying, _events = evaluation_data
    results[:] = [record for record in results if record["round"] != 2]
    with pytest.raises(ValueError, match="completed current-season target"):
        run_target(loader, trials=2)
    assert fake_runner == []


@pytest.mark.parametrize(
        ("positions", "extra_result", "classified", "reason"),
    [
        ([2, 3, 4, 5], None, True, "missing_position_one_result"),
        ([1, 1, 3, 4], None, True, "ambiguous_position_one_result"),
        ([2, 3, 4, 5], "outside", True, "winner_outside_entrant_roster"),
        ([1, 2, 3, 4], None, False, "position_one_result_not_classified"),
    ],
)
def test_invalid_observed_winner_is_excluded_with_reason(
    evaluation_data, fake_runner, positions, extra_result, classified, reason,
):
    loader, results, _qualifying, _events = evaluation_data
    for record in results:
        if record["round"] == 2:
            index = ["A1", "A2", "B1", "B2"].index(record["Driver"]["code"])
            record["position"] = str(positions[index])
            record["status"] = "Finished" if classified else "Retired"
    if extra_result == "outside":
        outside = row(2, "X1", "outside", "1:14.000", 1)
        results.append(outside)

    report = run_target(loader, trials=2)
    fold = report["folds"][0]
    assert fold["status"] == "excluded"
    assert fold["reason"] == reason
    assert fold["score"] is None
    assert report["aggregate"]["scored_events"] == 0
    assert report["aggregate"]["mean_brier_score"] is None


def test_all_target_trial_budget_is_checked_before_any_simulation(evaluation_data, fake_runner):
    loader, _results, _qualifying, _events = evaluation_data
    with pytest.raises(ValueError, match="total trials"):
        evaluation.evaluate_race_probabilities(
            loader, YEAR, all_targets=True, trials=4_000,
        )
    assert fake_runner == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"trials": True}, "trials"),
        ({"trials": 0}, "trials"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"form_races": 25}, "form_races"),
        ({"race_engine": "other"}, "race_engine"),
        ({"scenario": "cloudy"}, "scenario"),
    ],
)
def test_api_validates_controls_before_fetch(evaluation_data, kwargs, message):
    loader, *_ = evaluation_data
    with pytest.raises(ValueError, match=message):
        run_target(loader, **kwargs)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_one_lap_real_runner_smoke_simulates_qualifying_for_each_engine(
    evaluation_data, monkeypatch, engine,
):
    loader, _results, _qualifying, _events = evaluation_data
    original_track_stats = loader._track_stats_from_event
    monkeypatch.setattr(
        loader,
        "_track_stats_from_event",
        lambda year, event: original_track_stats(year, event).model_copy(
            update={"total_laps": 1},
        ),
    )

    report = run_target(loader, trials=1, race_engine=engine)
    fold = report["folds"][0]
    assert fold["status"] == "scored"
    assert fold["forecast"]["trials"] == 1
    assert fold["simulation"]["race_engine"] == engine
    assert fold["simulation"]["qualifying"] == "simulated_per_trial"
    assert sum(item["probability"] for item in fold["forecast"]["drivers"].values()) + (
        fold["forecast"]["no_classified_winner"]["probability"]
    ) == pytest.approx(1.0)


def test_cli_requires_explicit_target_and_emits_no_partial_json_on_failure(monkeypatch, capsys):
    cli = load_cli_module()
    with pytest.raises(SystemExit) as missing_target:
        cli.main([])
    assert missing_target.value.code == 2
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(cli, "CurrentSeasonDataLoader", lambda **kwargs: object())

    def fail_evaluation(*args, **kwargs):
        raise CurrentSeasonDataError("provider feed unavailable")

    monkeypatch.setattr(cli, "evaluate_race_probabilities", fail_evaluation)
    with pytest.raises(SystemExit) as failed:
        cli.main(["--race", "2"])
    captured = capsys.readouterr()
    assert failed.value.code == 1
    assert captured.out == ""
    assert "provider feed unavailable" in captured.err


def test_cli_flags_and_json_output_are_machine_readable(monkeypatch, capsys):
    cli = load_cli_module()
    monkeypatch.setattr(cli, "CurrentSeasonDataLoader", lambda **kwargs: object())
    monkeypatch.setattr(
        cli, "evaluate_race_probabilities", lambda *args, **kwargs: {"ok": True},
    )

    with pytest.raises(SystemExit) as both_targets:
        cli.main(["--race", "2", "--all"])
    assert both_targets.value.code == 2
    assert capsys.readouterr().out == ""

    cli.main(["--race", "2", "--trials", "1", "--seed", "7"])
    assert json.loads(capsys.readouterr().out) == {"ok": True}
