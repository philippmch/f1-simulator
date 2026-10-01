"""Opening policy forecasts use the same post-fit pace as executed races."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation import chronological_race, opening_strategy, race
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype

PROFILE = {"soft": .5, "medium": 1.0, "hard": .25}


def _models():
    return (
        Driver(id="d", name="Driver", team_id="t"),
        Car(team_id="t", team_name="Team"),
        Track(id="t", name="Track", country="Test", total_laps=90,
              base_lap_time=110, pit_lane_delta=22),
        Weather(change_probability=0),
    )


def _capture_forecasts(monkeypatch, module):
    original = module.forecast_final_lap
    calls = []

    def capture(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "forecast_final_lap", capture)
    return calls


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_opening_post_fit_forecast_matches_native_physical_pace(monkeypatch, engine):
    driver, car, track, weather = _models()
    simulator = RaceSimulator(np.random.default_rng(7), tire_warmup=PROFILE)
    before = deepcopy((driver, car, track, weather, PROFILE))
    rng_before = deepcopy(simulator.rng.bit_generator.state)
    helper_calls = _capture_forecasts(monkeypatch, opening_strategy)
    consume = RaceSimulator._consume_tire_warmup
    fit_costs = []

    def capture_fit(self, state):
        cost = consume(self, state)
        if cost:
            fit_costs.append((state.laps_completed, state.current_tire.compound.value, cost))
        return cost

    monkeypatch.setattr(RaceSimulator, "_consume_tire_warmup", capture_fit)

    def project(profile):
        return opening_strategy._policy_path_outcome(
            driver, car, track, weather, TeamStrategyArchetype.BALANCED,
            simulator.strategy_tuning, simulator.strategy_profiles,
            TireCompound.SOFT, 0, tire_warmup=profile,
        )

    expected_laps, expected_time = project(PROFILE)
    fitted_calls = helper_calls.copy()
    assert fit_costs == [(21, "hard", PROFILE["hard"])]
    assert (driver, car, track, weather, PROFILE) == before
    assert simulator.rng.bit_generator.state == rng_before

    monkeypatch.setattr(simulator, "_infer_team_strategy",
                        lambda *args: TeamStrategyArchetype.BALANCED)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *args, **kwargs: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident",
                        lambda *args, **kwargs: None)
    physics = simulator.lap_simulator.calculate_lap_time
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda *args, **kwargs: physics(
                            *args, **{**kwargs, "sample_variation": False},
                        ))
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        expected_stationary_time)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())
    module = race if engine == "standard" else chronological_race
    native_calls = _capture_forecasts(monkeypatch, module)
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result, = execute([driver], {"t": car}, track, weather, ["d"],
                      starting_tires={"d": TireCompound.SOFT})

    assert result.pit_laps == [22]
    assert result.strategy == ["soft", "hard"]
    assert result.laps_completed == expected_laps == 65
    assert result.total_time == pytest.approx(expected_time, abs=1e-8)
    # The grid softs are ready. Only the subsequent hard fit pays its fee,
    # once, even though that set completes many more laps.
    assert fit_costs == [(21, "hard", PROFILE["hard"])] * 2
    helper_after_fit, = [call for call in fitted_calls if call[1] == 22]
    native_after_fit = next(call for call in native_calls if call[1] == 22)
    assert helper_after_fit[:5] == pytest.approx(native_after_fit[:5], abs=1e-8)
    # The ready grid tyres carry no fee into the first observed crossing.
    first_crossing, = [call for call in fitted_calls if call[1] == 1]
    assert first_crossing[2] == first_crossing[3]
