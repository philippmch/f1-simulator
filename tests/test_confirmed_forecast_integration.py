"""Portable publication and display checks with complete synthetic native paths."""

import csv
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest
from test_post_qualifying_forecast import PostQualifyingLoader
from test_recorded_forecast import result as observed_result

from f1sim.analysis.confirmed_forecast import LEARNED_FORECAST
from f1sim.analysis.learned_afterqual import (
    FrozenAfterQualifyingModel,
    canonical_digest,
    load_frozen_afterqual_model,
)
from f1sim.analysis.montecarlo import DriverStatistics, RaceEventStatistics, SimulationResults
from f1sim.analysis.provenance import simulation_runtime
from f1sim.analysis.race_probability_scores import score_winner_probabilities
from f1sim.analysis.recorded_forecast import (
    _digest,
    load_recorded_forecast,
    record_race_forecast,
    save_recorded_forecast,
    score_recorded_forecast,
    validate_recorded_forecast,
)
from f1sim.models import Weather
from f1sim.output.export import Exporter
from f1sim.simulation.qualifying import QualifyingResult
from f1sim.simulation.race import DriverStatus, RaceResult


class LearnedRecordingLoader(PostQualifyingLoader):
    def __init__(self):
        super().__init__()
        self.instant = datetime(2027, 1, 20, 12, tzinfo=timezone.utc)
        qualifying, race = self.instant - timedelta(hours=3), self.instant + timedelta(days=1)
        self.event.update(round=17, date=race.date().isoformat(),
                          time=race.time().isoformat() + "Z")
        self.event["sessions"]["Qualifying"] = {
            "date": qualifying.date().isoformat(), "time": qualifying.time().isoformat() + "Z",
        }
        for row in self.qualifying_rows:
            row["round"] = 17

    def _assert_current_year(self, year):
        assert year == 2027

    def get_starting_grid(self, year, race, drivers):
        grid = super().get_starting_grid(year, race, drivers)
        self._race_grid["round"] = 17
        return grid


@pytest.fixture
def mocked_recording(monkeypatch):
    loader = LearnedRecordingLoader()
    captured = []

    def features(_loader, year, target, drivers, grid, pits, *, as_of=None):
        from f1sim.data.confirmed_grid_forecast import build_feature_snapshot

        assert grid == ["BB", "AA"] and pits == []
        predictors = {"year": year, "round": target, "roster": [
            {"driver": code.lower(), "code": code, "team": "a",
             "qualifying_position": grid.index(code) + 1, "grid_position": grid.index(code) + 1,
             "official_table_order_index": grid.index(code) + 1, "pit_lane_start": False,
             "q1": None, "q2": None, "q3": None} for code in ("AA", "BB")
        ]}
        snapshot = build_feature_snapshot(predictors, [])
        event = loader.event
        qualifying = event["sessions"]["Qualifying"]
        provenance = loader.get_provenance()
        provenance["fetched_at"] = as_of.isoformat()
        snapshot.update({"as_of": as_of.isoformat(), "context": {
            "as_of": as_of.isoformat(), "event": event,
            "qualifying_starts_at": f'{qualifying["date"]}T{qualifying["time"]}',
            "race_starts_at": f'{event["date"]}T{event["time"]}',
            "provider_fetched_at": as_of.isoformat(), "source_provenance": provenance,
            "modeled_seats": {driver.id: driver.team_id for driver in drivers},
            "prior_race_starts_at": {}, "target_predictors": predictors,
            "completed_history": [], "target_predictors_sha256": canonical_digest(predictors),
            "completed_history_sha256": canonical_digest([]),
            "target_race_performance_used": False,
        }})
        _reseal(snapshot)
        return snapshot

    def native_run(runner, num_simulations, **kwargs):
        assert num_simulations == 400
        races, qualifying = [], []
        for trial in range(num_simulations):
            winner = "AA" if trial < 300 else "BB"
            races.append([
                RaceResult(code, code, "A", 1 if code == winner else 2,
                           270., 0., 1, 90., DriverStatus.FINISHED, classified=True)
                for code in ("AA", "BB")
            ])
            qualifying.append([
                QualifyingResult(code, code, position, 90. + position, None, None, None, None)
                for position, code in enumerate(("BB", "AA"), 1)
            ])
        stats = {code: DriverStatistics(
            code, code, "A", wins=300 if code == "AA" else 100, podiums=400,
            points_finishes=400, positions=[race[i].position for race in races],
        ) for i, code in enumerate(("AA", "BB"))}
        inputs = {
            "schema_version": 14, "drivers": [driver.model_dump() for driver in runner.drivers],
            "cars": {code: car.model_dump() for code, car in runner.cars.items()},
            "track": runner.track.model_dump(), "weather": runner.weather.model_dump(),
            "starting_tires": {}, "starting_grid": list(runner.starting_grid),
            "rng_policy": runner.rng_policy, "runtime": simulation_runtime(),
        }
        native = SimulationResults(
            400, runner.track.name, stats, races, qualifying,
            event_stats=RaceEventStatistics(num_simulations=400), seed=int(runner.base_seed),
            parallel=False, race_engine=runner.race_engine, input_snapshot=inputs,
        )
        captured.append(native)
        return native

    monkeypatch.setattr("f1sim.data.confirmed_grid_forecast.build_confirmed_grid_features",
                        features)
    monkeypatch.setattr("f1sim.analysis.montecarlo.MonteCarloRunner.run", native_run)

    def record(**kwargs):
        return record_race_forecast(
            loader, 2027, 17, trials=400, stage="post_qualifying",
            forecast_policy=LEARNED_FORECAST, now=kwargs.pop("now", lambda: loader.instant),
            **kwargs,
        )

    return loader, captured, record


def _reseal(value):
    value["content_sha256"] = canonical_digest(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_schema_five_roundtrip_and_offline_scoring_keep_raw_counts(
    mocked_recording, monkeypatch, tmp_path,
):
    loader, captured, record = mocked_recording
    saved = record()
    assert saved["schema_version"] == 5
    assert saved["winner_forecast"]["drivers"]["AA"]["wins"] == 300
    assert saved["winner_forecast"]["drivers"]["AA"]["probability"] == .75
    assert saved["forecast_estimate"]["win_probabilities"]["BB"] > .75
    validate_recorded_forecast(saved)
    path = save_recorded_forecast(tmp_path / "confirmed.json", saved)
    assert load_recorded_forecast(path) == saved
    before = path.read_bytes()
    monkeypatch.setattr("f1sim.analysis.montecarlo.MonteCarloRunner.run",
                        lambda *args, **kwargs: pytest.fail("offline scoring ran a race"))
    monkeypatch.setattr(loader, "get_starting_grid",
                        lambda *args: pytest.fail("offline scoring fetched a new grid"))
    observed = [observed_result(code, place) for place, code in enumerate(("BB", "AA"), 1)]
    for row in observed:
        row["round"] = 17
    scored = score_recorded_forecast(saved, loader, observed, loader.qualifying_rows)
    expected = score_winner_probabilities(
        saved["forecast_estimate"]["win_probabilities"], 0., "BB",
    )
    assert scored["winner_score"] == expected
    assert scored["winner_policy"] == LEARNED_FORECAST
    assert scored["native_winner_score"] != scored["winner_score"]
    assert scored["pole_score"] is None
    assert path.read_bytes() == before
    assert captured[0].driver_stats["AA"].wins == 300


@pytest.mark.parametrize("mutation", ["outer_seal", "forecast_marginal", "joint_winner",
                                    "raw_count", "training_overlap", "artifact_deadline",
                                    "feature_cutoff", "grid", "eligibility"])
def test_schema_five_rejects_inconsistent_or_resealed_evidence(mocked_recording, mutation):
    loader, _, record = mocked_recording
    saved = record()
    estimate = saved["forecast_estimate"]
    if mutation == "outer_seal":
        saved["forecast_policy"] = "native"
        with pytest.raises(ValueError, match="seal"):
            validate_recorded_forecast(saved)
        return
    if mutation == "forecast_marginal":
        estimate["nonfinish_probabilities"]["AA"] = .1
    elif mutation == "joint_winner":
        estimate["joint_outcomes"][0][0]["classified"] = False
    elif mutation == "raw_count":
        estimate["native_winner_counts"]["AA"] -= 1
    elif mutation == "training_overlap":
        estimate["feature_snapshot"]["year"] = 2026
        estimate["feature_snapshot"]["round"] = 15
    elif mutation == "artifact_deadline":
        estimate["artifact"]["created_at"] = (loader.instant + timedelta(hours=1)).isoformat()
        _reseal(estimate["artifact"])
    elif mutation == "feature_cutoff":
        estimate["feature_snapshot"]["as_of"] = (loader.instant - timedelta(hours=1)).isoformat()
    elif mutation == "grid":
        saved["simulation_inputs"]["starting_grid"].reverse()
    elif mutation == "eligibility":
        estimate["eligibility"]["policy"] = "native"
    _reseal(estimate)
    _reseal(saved)
    with pytest.raises(ValueError):
        validate_recorded_forecast(saved)


@pytest.mark.parametrize("weather", [Weather(track_temperature=36), Weather(air_temperature=26),
                                   Weather(humidity=.6), Weather(rain_intensity=.1),
                                   Weather(track_wetness=.1)])
def test_recorder_refuses_unsupported_effective_weather_before_native_trials(
    mocked_recording, weather,
):
    _, captured, record = mocked_recording
    with pytest.raises(ValueError, match="fixed dry"):
        record(weather=weather)
    assert captured == []


def test_recorder_fixed_rainfall_contract_pins_effective_weather_change(mocked_recording):
    _, _, record = mocked_recording
    saved = record(weather=Weather(change_probability=.2))
    assert saved["simulation_inputs"]["weather"]["change_probability"] == 0


def test_feature_source_refusal_precedes_native_trials(mocked_recording, monkeypatch):
    _, captured, record = mocked_recording

    def unavailable(*args, **kwargs):
        raise ValueError("unavailable published qualifying source")

    monkeypatch.setattr("f1sim.data.confirmed_grid_forecast.build_confirmed_grid_features",
                        unavailable)
    with pytest.raises(ValueError, match="qualifying source"):
        record()
    assert captured == []


def test_recorder_cannot_publish_a_run_finishing_after_the_race_deadline(mocked_recording):
    loader, captured, record = mocked_recording
    deadline = loader.instant + timedelta(days=1)
    with pytest.raises(ValueError, match="race started"):
        record(now=lambda: deadline if captured else loader.instant)


@pytest.mark.parametrize("mutation", ["car_pace", "driver_skill", "race_distance"])
def test_schema_five_binds_the_complete_native_models_after_outer_resealing(
    mocked_recording, mutation,
):
    _, _, record = mocked_recording
    saved = record()
    if mutation == "car_pace":
        saved["simulation_inputs"]["cars"]["a"]["base_pace"] = .5
    elif mutation == "driver_skill":
        saved["simulation_inputs"]["drivers"][0]["skill_rating"] = .5
    else:
        saved["simulation_inputs"]["track"]["total_laps"] = 99
    _reseal(saved)
    with pytest.raises(ValueError, match="prepared stock models"):
        validate_recorded_forecast(saved)


def test_application_refuses_a_model_changed_after_the_pretrial_binding(
    mocked_recording, monkeypatch,
):
    from f1sim.analysis.montecarlo import MonteCarloRunner

    _, _, record = mocked_recording
    original = MonteCarloRunner.run

    def changed_model(runner, count, **kwargs):
        runner.cars["a"].base_pace = .5
        return original(runner, count, **kwargs)

    monkeypatch.setattr(MonteCarloRunner, "run", changed_model)
    with pytest.raises(ValueError, match="prepared stock simulation inputs"):
        record()


def test_a_model_created_after_the_recording_cutoff_is_refused_before_trials(
    mocked_recording, monkeypatch,
):
    loader, captured, record = mocked_recording
    artifact = load_frozen_afterqual_model().to_dict()
    artifact["created_at"] = (loader.instant + timedelta(seconds=1)).isoformat()
    _reseal(artifact)
    future_artifact = FrozenAfterQualifyingModel.from_mapping(artifact)
    monkeypatch.setattr("f1sim.analysis.learned_afterqual.load_frozen_afterqual_model",
                        lambda: future_artifact)
    with pytest.raises(ValueError, match="recording cutoff"):
        record()
    assert captured == []


def test_application_cannot_reuse_a_bound_estimate_for_another_race_distance(mocked_recording):
    from f1sim.analysis.confirmed_forecast import apply_confirmed_forecast

    _, captured, record = mocked_recording
    saved = record()
    estimate = saved["forecast_estimate"]
    prepared = {"features": estimate["feature_snapshot"], "artifact": estimate["artifact"],
                "eligibility": estimate["eligibility"]}
    changed = deepcopy(captured[0])
    changed.input_snapshot["track"]["total_laps"] = 99
    with pytest.raises(ValueError, match="prepared stock simulation inputs"):
        apply_confirmed_forecast(changed, prepared)


def test_export_and_dashboard_views_keep_native_counts_and_label_weighted_estimates(
    mocked_recording, tmp_path,
):
    from f1sim.web.server import _summarize_scenario_results

    _, captured, record = mocked_recording
    saved = record()
    native = captured[0]
    original_counts = {code: row.wins for code, row in native.driver_stats.items()}
    exporter = Exporter(output_dir=tmp_path)
    statistics = json.loads(exporter.export_statistics_json(native).read_text(encoding="utf-8"))
    assert statistics["forecast_estimate"] == saved["forecast_estimate"]
    assert statistics["win_probabilities"]["AA"] == 75.
    assert statistics["driver_statistics"]["AA"]["wins"] == 300
    with exporter.export_forecast_csv(native).open(newline="", encoding="utf-8") as stream:
        rows = {row["driver_id"]: row for row in csv.DictReader(stream)}
    assert rows["AA"]["native_wins"] == "300"
    assert float(rows["BB"]["win_probability"]) == (
        saved["forecast_estimate"]["win_probabilities"]["BB"]
    )
    html = exporter.export_report_html(native).read_text(encoding="utf-8")
    assert "Experimental confirmed-grid forecast" in html
    assert "raw simulation" in html
    assert "Nonfinish" in html
    scenario = _summarize_scenario_results({"Dry": native})["scenarios"]["Dry"]
    assert dict(scenario["win_probabilities"])["AA"] == 75.
    assert scenario["forecast_estimate"] == saved["forecast_estimate"]
    assert {code: row.wins for code, row in native.driver_stats.items()} == original_counts
    assert _digest({key: value for key, value in saved.items() if key != "content_sha256"}) == (
        saved["content_sha256"]
    )


@pytest.mark.parametrize("override", [
    {"race_grid_mode": "simulated"}, {"weather_mode": "evolving"},
    {"scenarios": "light_rain"}, {"simulations": 32},
    {"postqual_pace_policy": "season_points"}, {"control_schedule": []},
])
def test_http_route_rejects_contradictory_learned_options_before_provider_io(
    monkeypatch, override,
):
    from fastapi.testclient import TestClient

    from f1sim.web import server

    monkeypatch.setattr(server, "_get_loader",
                        lambda: pytest.fail("unsupported public request fetched live inputs"))
    payload = {"year": datetime.now(timezone.utc).year, "race": "17", "simulations": 400,
               "scenarios": "dry", "race_grid_mode": "official",
               "weather_mode": "fixed_rainfall", "race_engine": "chronological",
               "rng_policy": "isolated_weather_v1", "forecast_policy": LEARNED_FORECAST,
               **override}
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"]
