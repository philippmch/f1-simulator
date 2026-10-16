from copy import deepcopy
from types import SimpleNamespace

import pytest

from f1sim.analysis.confirmed_forecast import (
    LEARNED_FORECAST,
    apply_confirmed_forecast,
    native_forecast_input_digest,
    validate_confirmed_forecast_options,
)
from f1sim.models import Weather
from f1sim.web import server


def _options(**changes):
    return {
        "simulations": 400, "race_grid_mode": "official",
        "postqual_pace_policy": "qualifying_led", "race_engine": "chronological",
        "rng_policy": "isolated_weather_v1", "weather_mode": "fixed_rainfall",
        "scenarios": ["dry"], **changes,
    }


def test_native_requests_keep_the_existing_scenario_scope():
    validate_confirmed_forecast_options("native", **_options(
        simulations=10, race_grid_mode="simulated", scenarios=["heavy_rain"],
        control_schedule=[], starting_tires={"VER": "soft"},
    ))
    validate_confirmed_forecast_options(LEARNED_FORECAST, **_options())


@pytest.mark.parametrize("change", [
    {"simulations": 399}, {"race_grid_mode": "auto"},
    {"postqual_pace_policy": "season_points"}, {"race_engine": "standard"},
    {"rng_policy": "isolated_race_v1"}, {"weather_mode": "evolving"},
    {"scenarios": ["dry", "light_rain"]}, {"control_schedule": []},
    {"starting_tires": {"VER": "soft"}}, {"pit_plans": {"VER": []}},
    {"race_tire_rules": {"minimum_sets": 3}}, {"qualifying_weather": {"Q1": {}}},
    {"compare_automatic": True}, {"pit_plan_selection": {"driver_id": "VER"}},
])
def test_unsupported_experiments_are_rejected_before_simulation(change):
    with pytest.raises(ValueError, match="learned forecast"):
        validate_confirmed_forecast_options(LEARNED_FORECAST, **_options(**change))


def test_dashboard_rejects_unsupported_forecasts_before_loading_sources(monkeypatch):
    def forbidden():
        pytest.fail("Unsupported forecast must not fetch data")

    monkeypatch.setattr(server, "_get_loader", forbidden)
    request = server.DashboardRunRequest(
        forecast_policy=LEARNED_FORECAST, simulations=400, race_grid_mode="official",
        scenarios="dry", weather_mode="fixed_rainfall", control_schedule=[],
    )
    with pytest.raises(ValueError, match="automatic race control"):
        server.run_dashboard_simulation(request)


def _result():
    return SimpleNamespace(
        num_simulations=400, race_engine="chronological", seed=42, forecast_estimate=None,
        input_snapshot={"rng_policy": "isolated_weather_v1", "starting_grid": ["A", "B"],
                        "weather": Weather(change_probability=0).model_dump()},
    )


def _prepared():
    return {"features": {"ids": ["a", "b"], "codes": ["A", "B"],
                         "grid_ids": ["a", "b"], "tree_x": [[1, 1, 0], [0, 0, 0]]},
            "artifact": {},
            "eligibility": {"eligible": True, "native_input_sha256":
                            native_forecast_input_digest(_result().input_snapshot,
                                race_engine="chronological", seed=42, trials=400)}}


@pytest.mark.parametrize("change", [
    {"starting_grid": ["B", "A"]}, {"rng_policy": "isolated_race_v1"},
    {"pit_lane_starters": ["B"]},
    {"control_schedule": []}, {"pit_plans": {"A": []}},
    {"weather": Weather(change_probability=.1).model_dump()},
])
def test_application_rechecks_the_actual_simulated_context(monkeypatch, change):
    def forbidden(*args, **kwargs):
        pytest.fail("An unsupported simulated context must not be labeled a learned forecast")

    monkeypatch.setattr("f1sim.analysis.learned_forecast.build_learned_forecast", forbidden)
    result = _result()
    result.input_snapshot.update(change)
    with pytest.raises(ValueError):
        apply_confirmed_forecast(result, _prepared())
    assert result.forecast_estimate is None


def test_application_attaches_only_the_separate_estimate(monkeypatch):
    estimate = {"win_probabilities": {"A": .6, "B": .4}}
    monkeypatch.setattr("f1sim.analysis.learned_forecast.build_learned_forecast",
                        lambda *args, **kwargs: deepcopy(estimate))
    result = _result()
    original = deepcopy(result.input_snapshot)
    assert apply_confirmed_forecast(result, _prepared()) == estimate
    assert result.input_snapshot == original
