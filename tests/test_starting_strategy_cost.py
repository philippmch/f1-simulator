"""Opening slicks retain seeded variety only among fastest projected schedules."""

import copy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.opening_strategy import dry_opening_policy_costs
from f1sim.simulation.pit_strategy import SLICKS, expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


def fixture():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="test", name="Test", country="Test", total_laps=30,
                  base_lap_time=90, pit_lane_delta=22)
    return driver, car, track


@pytest.mark.parametrize("style", list(TeamStrategyArchetype))
def test_seeded_opening_variety_only_in_equal_cost_minima(style):
    driver, car, track = fixture()
    simulator = RaceSimulator()
    scores = dict(dry_opening_policy_costs(driver, car, track, Weather(), style,
                                          simulator.strategy_tuning, simulator.strategy_profiles))
    best = min(scores.values())
    optimal = {compound for compound, score in scores.items()
               if score.negative_mean_laps == best.negative_mean_laps
               and score.mean_time <= best.mean_time + 1e-9}
    assert scores[TireCompound.HARD].mean_time > best.mean_time + 3
    selected = {RaceSimulator(np.random.default_rng(seed))._choose_starting_compound(
        style, track, Weather(), driver, car,
    ) for seed in range(40)}
    assert selected == optimal
    if style == TeamStrategyArchetype.BALANCED:
        assert selected == {TireCompound.SOFT, TireCompound.MEDIUM}


def test_projection_forwards_individual_physics_and_settings_without_mutation(monkeypatch):
    import f1sim.simulation.race as race_module
    from f1sim.simulation.opening_strategy import dry_opening_policy_costs

    driver, car, track = fixture()
    driver.tire_management = 0.4
    car.tire_degradation_factor = 1.5
    simulator = RaceSimulator(np.random.default_rng(42))
    observed = []

    def project(*args):
        observed.append(args)
        return dry_opening_policy_costs(*args)

    monkeypatch.setattr(race_module, "dry_opening_policy_costs", project)
    before = copy.deepcopy((driver, car))
    simulator._choose_starting_compound(TeamStrategyArchetype.CONSERVATIVE,
                                       track, Weather(), driver, car)
    assert len(observed) == 1
    args = observed[0]
    assert args[0] is driver and args[1] is car and args[2] is track
    assert args[4] == TeamStrategyArchetype.CONSERVATIVE
    assert args[5] is simulator.strategy_tuning
    assert args[6] is simulator.strategy_profiles
    assert (driver, car) == before
    # Exactly the original single choice draw, with no projection draws.
    expected_rng = np.random.default_rng(42)
    expected_rng.choice(3, p=[0.5, 0.5, 0])
    assert simulator.rng.random() == expected_rng.random()


def test_no_finite_schedule_retains_original_seeded_prior():
    driver, car, track = fixture()
    track.total_laps = 1
    for seed in range(8):
        with_context = RaceSimulator(np.random.default_rng(seed))._choose_starting_compound(
            TeamStrategyArchetype.BALANCED, track, Weather(), driver, car,
        )
        legacy = RaceSimulator(np.random.default_rng(seed))._choose_starting_compound(
            TeamStrategyArchetype.BALANCED, track, Weather(),
        )
        assert with_context == legacy


def test_weather_selection_skips_dry_projection(monkeypatch):
    import f1sim.simulation.race as race_module

    driver, car, track = fixture()
    monkeypatch.setattr(race_module, "plan_dry_stop",
                        lambda *args, **kwargs: pytest.fail("Wet start projected dry stints"))
    simulator = RaceSimulator(np.random.default_rng(42))
    before = copy.deepcopy(simulator.rng.bit_generator.state)
    assert simulator._choose_starting_compound(TeamStrategyArchetype.BALANCED, track,
                                               Weather(track_wetness=0.9), driver, car) == (
        TireCompound.WET
    )
    assert simulator.rng.bit_generator.state == before


@pytest.mark.parametrize("style", list(TeamStrategyArchetype))
def test_full_race_opening_matches_best_explicit_start_with_same_later_policy(style):
    class MeanPace:
        def normal(self, mean, std):
            return mean

    def run(compound=None):
        driver, car, track = fixture()
        simulator = RaceSimulator(np.random.default_rng(4))
        simulator._infer_team_strategy = lambda *args: style
        simulator.lap_simulator.rng = MeanPace()
        simulator.lap_simulator.calculate_pit_stop_time = lambda car: expected_stationary_time(car)
        simulator.event_manager.process_lap = lambda **kwargs: []
        return simulator.simulate_race(
            [driver], {"A": car}, track, Weather(change_probability=0), ["A"],
            starting_tires={"A": compound} if compound is not None else None,
        )[0]

    chosen = run()
    alternatives = [run(compound) for compound in SLICKS]
    assert chosen.total_time == pytest.approx(min(r.total_time for r in alternatives), abs=1e-8)
    assert alternatives[2].total_time > chosen.total_time + 3
    assert [r.strategy[0] for r in alternatives] == [c.value for c in SLICKS]


@pytest.mark.parametrize("management,degradation,optimal", [
    (1.0, 0.5, {TireCompound.SOFT, TireCompound.MEDIUM}),
    (0.0, 1.5, {TireCompound.MEDIUM, TireCompound.HARD}),
])
def test_driver_and_car_wear_change_optimal_opening_sets(management, degradation, optimal):
    driver, car, track = fixture()
    track.total_laps = 50
    track.tire_stress = 0.5
    driver.tire_management = management
    car.tire_degradation_factor = degradation
    choices = {RaceSimulator(np.random.default_rng(seed))._choose_starting_compound(
        TeamStrategyArchetype.BALANCED, track, Weather(), driver, car,
    ) for seed in range(40)}
    assert choices == optimal
