"""Pre-event timing, sealed probabilities and scoring without simulation."""

import importlib.util
import json
import signal
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from f1sim.analysis.recorded_forecast import (
    _digest,
    load_recorded_forecast,
    record_race_forecast,
    save_recorded_forecast,
    score_recorded_forecast,
    validate_recorded_forecast,
)
from f1sim.cancellation import SimulationCancelled
from f1sim.data.current import CurrentSeasonDataLoader, DriverStats
from f1sim.models import Car, Driver, Track


class ForecastLoader(CurrentSeasonDataLoader):
    def __init__(self):
        super().__init__()
        self.instant = datetime.now(timezone.utc)
        qualifying = self.instant + timedelta(days=3)
        self.event = {
            "round": 2, "race": "Future Race", "circuit_id": "test",
            "date": (qualifying + timedelta(days=1)).date().isoformat(),
            "sessions": {"Qualifying": {"date": qualifying.date().isoformat(),
                                         "time": qualifying.time().isoformat() + "Z"}},
        }
        self.rows = []
        self.qualifying_rows = []

    def _event_for_race(self, year, race):
        return self.event

    def _season_data(self, year):
        return self.rows, self.qualifying_rows

    def get_winner_allocation(self, year, race, drivers):
        # Legacy-record coverage remains separate from the new allocation tests.
        return None

    def get_weighted_driver_stats(self, year, race):
        return {identity: DriverStats(driver_id=identity, driver_name=identity,
                                      team_id="a", team_name="A", constructor_points=10)
                for identity in ("AA", "BB")}

    def create_drivers_from_stats(self, stats):
        return [Driver(id=identity, name=identity, team_id="a") for identity in stats]

    def create_cars_from_stats(self, stats):
        return {"a": Car(team_id="a", team_name="A")}

    def _track_stats_from_event(self, year, event):
        return None

    def create_track_from_stats(self, stats):
        return Track(id="test", name="Test", country="Test", total_laps=3, base_lap_time=90.)


def result(identity, position, *, qualifying=False):
    row = {"round": 2, "position": position, "Driver": {"code": identity,
           "driverId": identity.lower(), "givenName": identity, "familyName": identity},
           "Constructor": {"constructorId": "a", "name": "A"}}
    if not qualifying:
        row.update(status="Finished", laps="3")
    return row


class AllocationForecastLoader(ForecastLoader):
    def __init__(self):
        super().__init__()
        self.event["round"] = 3
        for number in (1, 2):
            for identity, points in (("AA", 25), ("BB", 0)):
                row = result(identity, 1 if identity == "AA" else 2)
                row.update(round=number, points=str(points))
                self.rows.append(row)

    def get_event_schedule(self, year):
        return [{"round": 1}, {"round": 2}, self.event]

    def get_winner_allocation(self, year, race, drivers):
        return CurrentSeasonDataLoader.get_winner_allocation(self, year, race, drivers)


def test_schema_two_freezes_points_preserves_raw_counts_and_scores_without_refitting(
    tmp_path, monkeypatch,
):
    loader = AllocationForecastLoader()
    record = record_race_forecast(loader, loader.instant.year, 3, trials=2,
                                now=lambda: loader.instant)
    assert record["schema_version"] == 2
    assert record["winner_estimate"]["drivers"]["AA"]["probability"] == .75
    assert record["winner_estimate"]["allocation"]["cutoff_round"] == 2
    assert record["performance_rounds"] == [1, 2]
    path = save_recorded_forecast(tmp_path / "calibrated.json", record)
    before = path.read_bytes()
    assert load_recorded_forecast(path) == record
    monkeypatch.setattr(loader, "get_winner_allocation",
                        lambda *a: pytest.fail("Scoring cannot refit from live points"))
    monkeypatch.setattr("f1sim.analysis.montecarlo.MonteCarloRunner.run",
                        lambda *a, **k: pytest.fail("Scoring cannot rerun a simulation"))
    observed = [result("AA", 1), result("BB", 2)]
    for row in observed:
        row["round"] = 3
    score = score_recorded_forecast(record, loader, observed, [])
    assert score["winner_score"]["brier_score"] == .125
    assert score["winner_policy"] == "teammate_race_points_v1"
    assert "native_winner_score" in score
    assert path.read_bytes() == before


@pytest.mark.parametrize("edit", ["probability", "weights", "target_points", "constructor"])
def test_resealed_schema_two_record_rejects_modified_or_leaking_forecasts(edit):
    loader = AllocationForecastLoader()
    record = record_race_forecast(loader, loader.instant.year, 3, trials=1,
                                now=lambda: loader.instant)
    estimate = record["winner_estimate"]
    if edit == "probability":
        estimate["drivers"]["AA"]["probability"] = .6
    elif edit == "weights":
        estimate["allocation"]["teams"]["a"]["weights"]["AA"] = .6
    elif edit == "target_points":
        estimate["allocation"]["prior_race_points"][0]["round"] = 3
    else:
        record["simulation_inputs"]["drivers"][0]["team_id"] = "another_constructor"
    body = {key: value for key, value in record.items() if key != "content_sha256"}
    record["content_sha256"] = _digest(body)
    with pytest.raises(ValueError):
        validate_recorded_forecast(record)


def test_real_forecast_round_trip_score_and_existing_file_preservation(tmp_path, monkeypatch):
    loader = ForecastLoader()
    year = loader.instant.year
    record = record_race_forecast(loader, year, 2, trials=2, now=lambda: loader.instant)
    path = save_recorded_forecast(tmp_path / "forecast.json", record)
    assert load_recorded_forecast(path) == record
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_recorded_forecast(path, record)
    assert path.read_bytes() == before
    # A score is obtained from the frozen probabilities, with no new model runs.
    monkeypatch.setattr("f1sim.analysis.montecarlo.MonteCarloRunner.run",
                        lambda *a, **k: pytest.fail("Scoring must not run simulations"))
    scored = score_recorded_forecast(record, loader,
        [result("AA", 1), result("BB", 2)],
        [result("BB", 1, qualifying=True), result("AA", 2, qualifying=True)])
    assert scored["status"] == "scored"
    assert scored["observed_winner"] == "AA"
    assert scored["observed_pole"] == "BB"
    assert scored["qualifying_scored_drivers"] == 2
    assert path.read_bytes() == before
    edited = deepcopy(record)
    edited["winner_forecast"]["drivers"]["AA"]["probability"] = .123
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="seal"):
        load_recorded_forecast(path)
    body = {key: value for key, value in edited.items() if key != "content_sha256"}
    edited["content_sha256"] = _digest(body)
    with pytest.raises(ValueError, match="counts"):
        save_recorded_forecast(tmp_path / "bad.json", edited)


@pytest.mark.parametrize("phase", ["results", "qualifying"])
def test_target_performance_prevents_a_pre_event_record(phase):
    loader = ForecastLoader()
    if phase == "results":
        loader.rows = [result("AA", 1)]
    else:
        loader.qualifying_rows = [result("AA", 1, qualifying=True)]
    with pytest.raises(ValueError, match="performance"):
        record_race_forecast(loader, loader.instant.year, 2, trials=1)


def test_past_or_missing_qualifying_deadlines_are_rejected():
    loader = ForecastLoader()
    with pytest.raises(ValueError, match="before qualifying"):
        record_race_forecast(loader, loader.instant.year, 2, trials=1,
                             now=lambda: loader.instant + timedelta(days=5))
    loader.event["sessions"] = {}
    with pytest.raises(ValueError, match="dated qualifying"):
        record_race_forecast(loader, loader.instant.year, 2, trials=1)


def test_crossing_the_qualifying_deadline_during_computation_is_rejected():
    loader = ForecastLoader()
    times = iter((loader.instant, loader.instant + timedelta(days=5)))
    with pytest.raises(ValueError, match="finished after"):
        record_race_forecast(loader, loader.instant.year, 2, trials=1, now=lambda: next(times))


def test_naive_clock_is_rejected():
    loader = ForecastLoader()
    with pytest.raises(ValueError, match="timezone-aware"):
        record_race_forecast(loader, loader.instant.year, 2, trials=1,
                             now=lambda: loader.instant.replace(tzinfo=None))


@pytest.mark.parametrize("field,value", [
    ("metadata", None), ("started_at", None), ("year", True), ("performance_rounds", None),
])
def test_malformed_sealed_record_reports_validation_error(field, value):
    loader = ForecastLoader()
    record = record_race_forecast(loader, loader.instant.year, 2, trials=1,
                                now=lambda: loader.instant)
    body = {key: item for key, item in record.items() if key != "content_sha256"}
    body[field] = value
    with pytest.raises(ValueError):
        validate_recorded_forecast({**body, "content_sha256": _digest(body)})


def test_rescheduled_event_is_not_scored_against_old_forecast():
    loader = ForecastLoader()
    record = record_race_forecast(loader, loader.instant.year, 2, trials=1,
                                now=lambda: loader.instant)
    loader.event["circuit_id"] = "replacement_venue"
    with pytest.raises(ValueError, match="circuit changed"):
        score_recorded_forecast(record, loader, [result("AA", 1), result("BB", 2)], [])


def test_cancellation_during_trials_never_returns_a_forecast():
    loader = ForecastLoader()
    done = [False]

    def progress(completed, total):
        if completed:
            done[0] = True

    with pytest.raises(SimulationCancelled):
        record_race_forecast(loader, loader.instant.year, 2, trials=3,
            now=lambda: loader.instant, progress_callback=progress,
            cancel_requested=lambda: done[0])


def load_record_cli():
    path = Path(__file__).resolve().parents[1] / "examples/record_race_forecast.py"
    spec = importlib.util.spec_from_file_location("record_race_forecast_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_cli_output_is_rejected_before_live_collection(tmp_path, monkeypatch):
    path = tmp_path / "forecast.json"
    path.write_text("existing forecast", encoding="utf-8")
    cli = load_record_cli()
    monkeypatch.setattr(cli, "CurrentSeasonDataLoader",
                        lambda **k: pytest.fail("existing records must fail before collection"))
    monkeypatch.setattr("sys.argv", ["record", "--race", "2", "--scenario", "dry",
                                    "--output", str(path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert path.read_text() == "existing forecast"


def test_cli_cancellation_restores_signal_handler_and_publishes_nothing(
    tmp_path, monkeypatch, capsys,
):
    cli = load_record_cli()
    monkeypatch.setattr(cli, "CurrentSeasonDataLoader", lambda **k: object())

    def cancel(*args, **kwargs):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert kwargs["cancel_requested"]()
        raise SimulationCancelled("cancelled")

    monkeypatch.setattr(cli, "record_race_forecast", cancel)
    path = tmp_path / "forecast.json"
    monkeypatch.setattr("sys.argv", ["record", "--race", "2", "--scenario", "dry",
                                    "--output", str(path)])
    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 130
    assert signal.getsignal(signal.SIGINT) is previous
    assert not path.exists()
    assert capsys.readouterr().out == ""
