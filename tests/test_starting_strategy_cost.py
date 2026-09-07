"""Opening slicks retain seeded variety only among fastest projected schedules."""

import copy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.pit_strategy import SLICKS, expected_stationary_time, plan_dry_stop
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
    costs = {compound: plan_dry_stop(driver, car, track, TIRE_COMPOUNDS[compound],
                                   0, track.total_laps, 3, {compound}).wait_cost
             for compound in SLICKS}
    assert costs[TireCompound.SOFT] == pytest.approx(costs[TireCompound.MEDIUM])
    assert costs[TireCompound.HARD] > min(costs.values()) + 3
    selected = {RaceSimulator(np.random.default_rng(seed))._choose_starting_compound(
        style, track, Weather(), driver, car,
    ) for seed in range(40)}
    assert selected == {TireCompound.SOFT, TireCompound.MEDIUM}


def test_projection_forwards_individual_physics_and_budget_without_mutation(monkeypatch):
    import f1sim.simulation.race as race_module

    driver, car, track = fixture()
    driver.tire_management = 0.4
    car.tire_degradation_factor = 1.5
    simulator = RaceSimulator(np.random.default_rng(42))
    observed = []
    budgets = []

    def budget(state, track):
        budgets.append((state.strategy_archetype, state.pit_stops, state.current_tire.compound))
        return 1

    def project(*args, **kwargs):
        observed.append(args)
        return plan_dry_stop(*args, **kwargs)

    monkeypatch.setattr(simulator, "_dry_stop_budget", budget)
    monkeypatch.setattr(race_module, "plan_dry_stop", project)
    before = copy.deepcopy((driver, car))
    simulator._choose_starting_compound(TeamStrategyArchetype.CONSERVATIVE,
                                       track, Weather(), driver, car)
    assert len(observed) == len(budgets) == 3
    for args in observed:
        assert args[0] is driver and args[1] is car
        assert args[4:7] == (0, 30, 1)
        assert args[7] == {args[3].compound}
    assert all(style == TeamStrategyArchetype.CONSERVATIVE and stops == 0
               for style, stops, _ in budgets)
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


def test_full_race_opening_matches_best_explicit_start_with_same_later_policy():
    class MeanPace:
        def normal(self, mean, std):
            return mean

    def run(compound=None):
        driver, car, track = fixture()
        simulator = RaceSimulator(np.random.default_rng(4))
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
