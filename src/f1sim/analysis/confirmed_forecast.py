"""Admission and application of the experimental confirmed-grid forecast.

The forecast is a separate view of complete simulated race paths.  It is only
supported for the stock, fixed-dry, after-qualifying workflow evaluated by the
historical study.  Simulation and strategy statistics remain native.
"""

from copy import deepcopy
from datetime import datetime, timezone

NATIVE_FORECAST = "native"
LEARNED_FORECAST = "learned_afterqual_v1"
MIN_LEARNED_TRIALS = 400
_STOCK_WEATHER = {
    "condition": "dry", "track_temperature": 35., "air_temperature": 25.,
    "humidity": .5, "rain_intensity": 0., "track_wetness": 0., "change_probability": 0.,
}


def validate_forecast_policy(value):
    if value not in (NATIVE_FORECAST, LEARNED_FORECAST):
        raise ValueError("forecast_policy must be native or learned_afterqual_v1")
    return value


def validate_confirmed_forecast_weather(weather):
    values = weather.model_dump() if hasattr(weather, "model_dump") else weather
    if values != _STOCK_WEATHER:
        raise ValueError("The learned forecast requires stock fixed dry weather")


def validate_confirmed_forecast_options(
    policy, *, simulations, race_grid_mode, postqual_pace_policy,
    race_engine, rng_policy, weather_mode, scenarios,
    starting_grid=None, pit_lane_starters=None, starting_tires=None,
    starting_tire_ages=None, tire_inventory=None, pit_plans=None,
    tire_warmup=None, qualifying_weather=None, weather_schedule=None,
    control_schedule=None, race_tire_rules=None, compare_automatic=False,
    pit_plan_selection=None,
):
    """Reject unsupported combinations before provider I/O or simulation."""
    if validate_forecast_policy(policy) == NATIVE_FORECAST:
        return
    required = {
        "race_grid_mode": (race_grid_mode, "official"),
        "postqual_pace_policy": (postqual_pace_policy, "qualifying_led"),
        "race_engine": (race_engine, "chronological"),
        "rng_policy": (rng_policy, "isolated_weather_v1"),
        "weather_mode": (weather_mode, "fixed_rainfall"),
    }
    for name, (actual, expected) in required.items():
        if actual != expected:
            raise ValueError(f"The learned forecast requires {name}={expected}")
    if list(scenarios) != ["dry"]:
        raise ValueError("The learned forecast supports one fixed dry scenario")
    if type(simulations) is not int or simulations < MIN_LEARNED_TRIALS:
        raise ValueError(f"The learned forecast requires at least {MIN_LEARNED_TRIALS} trials")
    custom = {
        "starting_grid": starting_grid, "pit_lane_starters": pit_lane_starters,
        "starting_tires": starting_tires, "starting_tire_ages": starting_tire_ages,
        "tire_inventory": tire_inventory, "pit_plans": pit_plans,
        "tire_warmup": tire_warmup, "qualifying_weather": qualifying_weather,
        "weather_schedule": weather_schedule, "race_tire_rules": race_tire_rules,
        "compare_automatic": compare_automatic, "pit_plan_selection": pit_plan_selection,
    }
    for name, value in custom.items():
        if value:
            raise ValueError(f"The learned forecast does not support custom {name}")
    # An empty prescribed control schedule disables random SC/VSC and is itself
    # an intervention.  It must not be confused with the default None.
    if control_schedule is not None:
        raise ValueError("The learned forecast requires automatic race control")


def prepare_confirmed_forecast(loader, year, race, drivers, starting_grid,
                               pit_lane_starters, *, as_of=None):
    """Freeze strictly available qualifying/history and inference before the run."""
    from f1sim.analysis.learned_afterqual import (
        AfterQualifyingFeatures,
        canonical_digest,
        load_frozen_afterqual_model,
        predict_after_qualifying,
    )
    from f1sim.data.confirmed_grid_forecast import build_confirmed_grid_features

    if as_of is not None and (not isinstance(as_of, datetime) or as_of.utcoffset() is None):
        raise ValueError("The forecast cutoff must be timezone-aware")
    source_features = build_confirmed_grid_features(
        loader, year, race, drivers, starting_grid, pit_lane_starters, as_of=as_of,
    )
    features = AfterQualifyingFeatures.from_mapping(source_features).to_dict()
    instant = datetime.fromisoformat(features["as_of"].replace("Z", "+00:00"))
    if instant.utcoffset() is None:
        raise ValueError("The forecast cutoff must be timezone-aware")
    instant = instant.astimezone(timezone.utc)
    artifact = load_frozen_afterqual_model()
    created = datetime.fromisoformat(artifact.to_dict()["created_at"].replace("Z", "+00:00"))
    if created.utcoffset() is None or created > instant:
        raise ValueError("The frozen forecast model must exist by the recording cutoff")
    prediction = predict_after_qualifying(features, artifact)
    return {
        "features": deepcopy(features), "artifact": artifact.to_dict(),
        "eligibility": {
            "eligible": True, "policy": LEARNED_FORECAST,
            "scope": "stock_confirmed_grid_fixed_dry",
            "as_of": instant.isoformat(),
            "feature_sha256": canonical_digest(features),
            "feature_source_sha256": source_features.get("content_sha256"),
            "artifact_sha256": prediction["artifact_sha256"],
            "training_boundary": prediction["training_boundary"],
            "minimum_trials": MIN_LEARNED_TRIALS,
        },
    }


def native_forecast_input_digest(inputs, *, race_engine, seed, trials):
    """Bind a forecast to the complete effective native models and assumptions."""
    from f1sim.analysis.learned_afterqual import canonical_digest

    values = {
        "drivers": inputs.get("drivers", []), "cars": inputs.get("cars", {}),
        "track": inputs.get("track", {}), "weather": inputs.get("weather", {}),
        "starting_grid": inputs.get("starting_grid"),
        "pit_lane_starters": inputs.get("pit_lane_starters", []),
        "rng_policy": inputs.get("rng_policy"),
        "control_schedule": inputs.get("control_schedule"),
        **{name: inputs.get(name) or {} for name in (
            "starting_tires", "starting_tire_ages", "tire_inventory", "pit_plans",
            "tire_warmup", "qualifying_weather", "race_tire_rules",
        )},
        "weather_schedule": inputs.get("weather_schedule") or [],
    }
    return canonical_digest({"inputs": values, "race_engine": race_engine,
                             "seed": int(seed), "trials": trials})


def bind_confirmed_forecast_inputs(prepared, runner, trials):
    """Pin the stock application runner before any trials can mutate its state."""
    inputs = {
        "drivers": [driver.model_dump() for driver in runner.drivers],
        "cars": {key: car.model_dump() for key, car in runner.cars.items()},
        "track": runner.track.model_dump(), "weather": runner.weather.model_dump(),
        **{name: deepcopy(getattr(runner, name)) for name in (
            "starting_grid", "pit_lane_starters", "rng_policy", "control_schedule",
            "starting_tires", "starting_tire_ages", "tire_inventory", "pit_plans",
            "tire_warmup", "qualifying_weather", "race_tire_rules", "weather_schedule",
        )},
    }
    prepared["eligibility"]["native_input_sha256"] = native_forecast_input_digest(
        inputs, race_engine=runner.race_engine, seed=runner.base_seed, trials=trials,
    )


def apply_confirmed_forecast(result, prepared):
    """Attach a coherent forecast view without changing any native result."""
    from f1sim.analysis.learned_forecast import build_learned_forecast

    if result.num_simulations < MIN_LEARNED_TRIALS:
        raise ValueError(f"The learned forecast requires at least {MIN_LEARNED_TRIALS} trials")
    inputs = result.input_snapshot or {}
    expected_digest = prepared["eligibility"].get("native_input_sha256")
    if expected_digest != native_forecast_input_digest(
        inputs, race_engine=result.race_engine, seed=result.seed, trials=result.num_simulations,
    ):
        raise ValueError("The learned forecast must match its prepared stock simulation inputs")
    if result.race_engine != "chronological" or inputs.get("rng_policy") != "isolated_weather_v1":
        raise ValueError("The learned forecast requires the evaluated race engine and RNG policy")
    features = prepared["features"]
    mapping = dict(zip(features["ids"], features["codes"], strict=True))
    if inputs.get("starting_grid") != [mapping[identity] for identity in features["grid_ids"]]:
        raise ValueError("Forecast features and the simulated official grid disagree")
    pit_codes = {code for code, row in zip(features["codes"], features["tree_x"], strict=True)
                 if row[2] == 1}
    expected_pits = [code for code in inputs["starting_grid"] if code in pit_codes]
    if inputs.get("pit_lane_starters", []) != expected_pits:
        raise ValueError("Forecast features and the simulated pit queue disagree")
    weather = inputs.get("weather", {})
    validate_confirmed_forecast_weather(weather)
    for name in ("starting_tires", "starting_tire_ages", "tire_inventory", "pit_plans",
                 "tire_warmup", "qualifying_weather", "weather_schedule"):
        if inputs.get(name):
            raise ValueError(f"The learned forecast does not support custom {name}")
    if inputs.get("control_schedule") is not None:
        raise ValueError("The learned forecast requires automatic race control")
    result.forecast_estimate = build_learned_forecast(
        result, prepared["features"], prepared["artifact"],
        eligibility=prepared["eligibility"],
    )
    return result.forecast_estimate
