"""Deterministic regressions for race-simulation edge cases."""

import multiprocessing as mp
from typing import Any, cast

import numpy as np
import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import ActiveAeroZone, Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.models.weather import WeatherCondition
from f1sim.simulation.events import EventManager, EventType, RaceEvent
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.race import (
    DriverRaceState,
    DriverStatus,
    RaceResult,
    RaceSimulator,
    TeamStrategyArchetype,
)


def _track(
    *,
    total_laps: int = 50,
    tire_stress: float = 0.5,
    overtake_difficulty: float = 0.5,
    pit_lane_delta: float = 20.0,
    active_aero_zones: list[ActiveAeroZone] | None = None,
) -> Track:
    return Track(
        id="test",
        name="Test Circuit",
        country="Nowhere",
        total_laps=total_laps,
        base_lap_time=90.0,
        pit_lane_delta=pit_lane_delta,
        active_aero_zones=(
            active_aero_zones
            if active_aero_zones is not None
            else [ActiveAeroZone(zone_id=1, sector=2, time_gain=0.4)]
        ),
        overtake_difficulty=overtake_difficulty,
        tire_stress=tire_stress,
        safety_car_probability=0.0,
    )


def _driver(driver_id: str, *, team_id: str | None = None) -> Driver:
    return Driver(
        id=driver_id,
        name=driver_id,
        team_id=team_id or driver_id.lower(),
        skill_rating=0.8,
        consistency=1.0,
        tire_management=0.8,
    )


def _state(driver_id: str = "DRV") -> DriverRaceState:
    driver = _driver(driver_id, team_id=driver_id.lower())
    return DriverRaceState(
        driver=driver,
        car=Car(team_id=driver.team_id, team_name=driver_id),
        position=1,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
    )


def _run_seeded_full_field_race(output: Any) -> None:
    """Subprocess target for the simultaneous-pit position regression."""

    drivers = [_driver(f"D{index:02d}", team_id=f"t{index:02d}") for index in range(22)]
    cars = {
        driver.team_id: Car(
            team_id=driver.team_id,
            team_name=driver.team_id,
            reliability=1.0,
        )
        for driver in drivers
    }
    results = RaceSimulator(rng=np.random.default_rng(42)).simulate_race(
        drivers,
        cars,
        _track(total_laps=57),
        Weather(),
        starting_grid=[driver.id for driver in drivers],
    )
    output.put([result.position for result in results])


def test_seeded_full_field_race_finishes_with_unique_positions() -> None:
    """A simultaneous pit batch must never enter an equal-position spin."""

    context = mp.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_run_seeded_full_field_race, args=(output,))
    process.start()
    process.join(10.0)
    timed_out = process.is_alive()
    if timed_out:
        process.terminate()
        process.join(2.0)

    try:
        assert not timed_out, "seed-42 full-field race exceeded the 10-second safety bound"
        assert process.exitcode == 0
        positions = output.get(timeout=2.0)
        assert len(positions) == 22
        assert sorted(positions) == list(range(1, 23))
    finally:
        output.close()
        output.join_thread()


def test_fresh_slick_compounds_have_realistic_pace_order_and_crossover() -> None:
    driver = _driver("DRV")
    track = _track()
    car = Car(team_id=driver.team_id, team_name="Team")

    def lap_time(compound: TireCompound, *, age: int = 0, weather: Weather | None = None) -> float:
        driver.current_tire_laps = age
        return LapSimulator(np.random.default_rng(17)).calculate_lap_time(
            driver,
            car,
            track,
            TIRE_COMPOUNDS[compound],
            weather or Weather(),
            lap_number=1,
            total_laps=50,
        )

    fresh = {
        compound: lap_time(compound)
        for compound in (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)
    }
    assert fresh[TireCompound.SOFT] < fresh[TireCompound.MEDIUM] < fresh[TireCompound.HARD]

    # Soft grip advantage is consumed by a long stint rather than remaining a
    # permanent pace bonus.
    assert lap_time(TireCompound.SOFT, age=25) > lap_time(TireCompound.MEDIUM, age=25)

    wet = Weather(
        condition=WeatherCondition.HEAVY_RAIN,
        track_wetness=0.85,
        rain_intensity=0.9,
    )
    assert lap_time(TireCompound.WET, weather=wet) < lap_time(
        TireCompound.INTERMEDIATE,
        weather=wet,
    )


def test_starting_compound_uses_strategy_rng_and_wet_crossover() -> None:
    track = _track(total_laps=57, tire_stress=0.5)
    dry = Weather()
    simulator = RaceSimulator(rng=np.random.default_rng(23))
    choices = [
        simulator._choose_starting_compound(TeamStrategyArchetype.BALANCED, track, dry)
        for _ in range(100)
    ]

    assert set(choices) >= {
        TireCompound.SOFT,
        TireCompound.MEDIUM,
        TireCompound.HARD,
    }

    # The same seed produces the same strategy draws, independent of any grid
    # position because the helper has no position input.
    replay = RaceSimulator(rng=np.random.default_rng(23))
    assert choices == [
        replay._choose_starting_compound(TeamStrategyArchetype.BALANCED, track, dry)
        for _ in range(100)
    ]

    assert simulator._choose_starting_compound(
        TeamStrategyArchetype.BALANCED,
        track,
        Weather(condition=WeatherCondition.LIGHT_RAIN, rain_intensity=0.6),
    ) == TireCompound.INTERMEDIATE
    assert simulator._choose_starting_compound(
        TeamStrategyArchetype.BALANCED,
        track,
        Weather(condition=WeatherCondition.HEAVY_RAIN, track_wetness=0.8, rain_intensity=0.9),
    ) == TireCompound.WET


def test_explicit_starting_tire_override_is_preserved() -> None:
    driver = _driver("A", team_id="a")
    simulator = RaceSimulator(rng=np.random.default_rng(24))
    observed: list[TireCompound] = []

    def fixed_lap(**kwargs: object) -> float:
        observed.append(kwargs["tire"].compound)  # type: ignore[attr-defined]
        return 90.0

    simulator.lap_simulator.calculate_lap_time = fixed_lap  # type: ignore[method-assign]
    simulator.event_manager.process_lap = lambda **kwargs: []  # type: ignore[method-assign]
    simulator.simulate_race(
        [driver],
        {"a": Car(team_id="a", team_name="Team")},
        _track(total_laps=8),
        Weather(),
        starting_grid=["A"],
        starting_tires={"A": TireCompound.HARD},
    )

    assert observed[0] == TireCompound.HARD


def test_initial_weather_is_used_unchanged_on_lap_one() -> None:
    driver = _driver("A", team_id="a")
    initial = Weather(
        condition=WeatherCondition.LIGHT_RAIN,
        track_temperature=28.0,
        air_temperature=20.0,
        humidity=0.8,
        rain_intensity=0.5,
        track_wetness=0.4,
        change_probability=1.0,
    )
    simulator = RaceSimulator(rng=np.random.default_rng(25))
    observed: list[Weather] = []

    def fixed_lap(**kwargs: object) -> float:
        observed.append(kwargs["weather"])  # type: ignore[arg-type]
        return 90.0

    simulator.lap_simulator.calculate_lap_time = fixed_lap  # type: ignore[method-assign]
    simulator.event_manager.process_lap = lambda **kwargs: []  # type: ignore[method-assign]
    simulator.simulate_race(
        [driver],
        {"a": Car(team_id="a", team_name="Team")},
        _track(total_laps=3),
        initial,
        starting_grid=["A"],
    )

    assert observed[0].model_dump() == initial.model_dump()
    assert observed[1].track_wetness > observed[0].track_wetness


def test_material_incident_penalty_reclassifies_elapsed_time_order() -> None:
    drivers = [_driver("A", team_id="a"), _driver("B", team_id="b")]
    simulator = RaceSimulator(rng=np.random.default_rng(26))
    simulator.lap_simulator.calculate_lap_time = (  # type: ignore[method-assign]
        lambda **kwargs: 90.0
    )

    penalty = RaceEvent(
        event_type=EventType.SPIN,
        lap=1,
        drivers_involved=["A"],
        time_loss_seconds=30.0,
    )

    def process_lap(**kwargs: object) -> list[RaceEvent]:
        return [penalty] if kwargs["lap"] == 1 else []

    simulator.event_manager.process_lap = process_lap  # type: ignore[method-assign]
    results = simulator.simulate_race(
        drivers,
        {
            "a": Car(team_id="a", team_name="A"),
            "b": Car(team_id="b", team_name="B"),
        },
        _track(total_laps=3, overtake_difficulty=1.0),
        Weather(),
        starting_grid=["A", "B"],
    )
    by_driver = {result.driver_id: result for result in results}

    assert by_driver["B"].position == 1
    assert by_driver["A"].position == 2
    assert by_driver["A"].total_time > by_driver["B"].total_time


def test_pit_lane_loss_is_reduced_under_sc_and_vsc_but_service_remains() -> None:
    track = _track(pit_lane_delta=20.0)
    weather = Weather()

    def pit_time(safety_car: bool = False, vsc: bool = False) -> float:
        simulator = RaceSimulator(rng=np.random.default_rng(27))
        simulator.lap_simulator.calculate_pit_stop_time = lambda car: 3.2  # type: ignore[method-assign]
        simulator.event_manager.safety_car_active = safety_car
        simulator.event_manager.vsc_active = vsc
        return simulator._execute_pit_stop(_state(), track, weather, current_lap=25)

    green = pit_time()
    vsc = pit_time(vsc=True)
    safety_car = pit_time(safety_car=True)

    assert green > vsc > safety_car
    assert green == pytest.approx(20.0 + 3.2)
    assert vsc == pytest.approx(20.0 * 0.75 + 3.2)
    assert safety_car == pytest.approx(20.0 * 0.55 + 3.2)


def test_wet_tire_history_skips_late_two_dry_compound_stop() -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(28))
    track = _track(total_laps=50)
    state = _state()
    state.tire_compound_history = ["medium", "intermediate"]

    assert not simulator._should_pit(
        state,
        [state],
        track,
        lap=track.total_laps - 1,
        pit_window_open=False,
    )

    dry_state = _state("DRY")
    assert simulator._should_pit(
        dry_state,
        [dry_state],
        track,
        lap=track.total_laps - 1,
        pit_window_open=False,
        weather=Weather(),
    )


def test_dry_rule_forces_second_slick_for_no_stop_medium_start() -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(35))
    track = _track(total_laps=50)
    state = _state()

    assert simulator._should_pit(
        state,
        [state],
        track,
        lap=track.total_laps - 1,
        pit_window_open=False,
        weather=Weather(),
    )
    simulator.lap_simulator.calculate_pit_stop_time = lambda car: 2.5  # type: ignore[method-assign]
    simulator._execute_pit_stop(state, track, Weather(), current_lap=track.total_laps - 1)

    assert state.tire_compound_history[0] == "medium"
    assert state.tire_compound_history[1] in {"soft", "hard"}
    assert len(simulator._used_slick_compounds(state)) == 2


def test_dry_rule_ignores_same_compound_earlier_stop_and_forces_new_slick() -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(36))
    track = _track(total_laps=50)
    state = _state()
    state.pit_stops = 1
    state.tire_compound_history = ["medium", "medium"]

    # A repeated medium stint did not satisfy the distinct-compound rule.
    assert len(simulator._used_slick_compounds(state)) == 1
    assert simulator._should_pit(
        state,
        [state],
        track,
        lap=track.total_laps - 1,
        pit_window_open=False,
        weather=Weather(),
    )
    simulator.lap_simulator.calculate_pit_stop_time = lambda car: 2.5  # type: ignore[method-assign]
    simulator._execute_pit_stop(state, track, Weather(), current_lap=track.total_laps - 1)

    assert state.tire_compound_history[:2] == ["medium", "medium"]
    assert state.tire_compound_history[2] in {"soft", "hard"}
    assert len(simulator._used_slick_compounds(state)) == 2


def test_active_aero_is_not_proximity_gated() -> None:
    driver = _driver("A")
    car = Car(team_id="a", team_name="A", straight_line_speed=0.95)
    tire = TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True)
    track = _track()

    def lap_time(gap: float | None, active: bool = True, overtake: bool = False) -> float:
        return LapSimulator(np.random.default_rng(31)).calculate_lap_time(
            driver,
            car,
            track,
            tire,
            Weather(),
            lap_number=5,
            total_laps=50,
            gap_to_car_ahead=gap,
            active_aero_enabled=active,
            overtake_mode_active=overtake,
        )

    # Straight Mode is a common baseline effect; a car 10 seconds behind (or
    # the leader with no gap) receives the same configured-section gain.
    assert lap_time(None) == pytest.approx(lap_time(10.0))
    assert lap_time(None) < lap_time(None, active=False)


def test_overtake_mode_bonus_is_bounded_and_zero_on_zero_zone_track() -> None:
    driver = _driver("A")
    car = Car(team_id="a", team_name="A", straight_line_speed=1.0)
    tire = TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True)
    track = _track()

    def lap_time(track_value: Track, overtake: bool) -> float:
        return LapSimulator(np.random.default_rng(32)).calculate_lap_time(
            driver,
            car,
            track_value,
            tire,
            Weather(),
            lap_number=5,
            total_laps=50,
            gap_to_car_ahead=0.8,
            active_aero_enabled=True,
            overtake_mode_active=overtake,
        )

    normal = lap_time(track, False)
    deployed = lap_time(track, True)
    zero_zone = track.model_copy(update={"active_aero_zones": []})
    assert 0.0 < normal - deployed <= 0.35
    assert lap_time(zero_zone, True) == pytest.approx(lap_time(zero_zone, False))


def test_overtake_mode_energy_depletes_and_recharges_with_fixed_seed() -> None:
    track = _track()
    simulator = RaceSimulator(rng=np.random.default_rng(33))
    state = _state()

    assert simulator._deploy_overtake_mode_if_eligible(state, track, 1.0, True)
    assert state.overtake_mode_energy == pytest.approx(0.65)
    assert simulator._deploy_overtake_mode_if_eligible(state, track, 1.0, True)
    assert state.overtake_mode_energy == pytest.approx(0.30)
    assert not simulator._deploy_overtake_mode_if_eligible(state, track, 1.0, True)
    assert state.overtake_mode_deployments == 2

    # The configured detection gap is strict and the store never exceeds its
    # physical bound, including a neutralized recharge step.
    assert not simulator._deploy_overtake_mode_if_eligible(state, track, 1.01, True)
    simulator._recharge_overtake_mode_energy([state], neutralized=True)
    assert state.overtake_mode_energy == pytest.approx(0.42)
    state.overtake_mode_energy = 1.0
    simulator._recharge_overtake_mode_energy([state], neutralized=True)
    assert state.overtake_mode_energy == pytest.approx(1.0)


@pytest.mark.parametrize("reverse_iteration", [False, True])
def test_race_lap_gaps_use_completed_lap_times(
    monkeypatch: pytest.MonkeyPatch, reverse_iteration: bool,
) -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(33))
    drivers = [_driver("A"), _driver("B")]
    cars = {
        driver.team_id: Car(team_id=driver.team_id, team_name=driver.id)
        for driver in drivers
    }
    observations: dict[tuple[int, str], tuple[float | None, bool]] = {}

    def calculate_lap_time(**kwargs: Any) -> float:
        driver_id = kwargs["driver"].id
        observations[kwargs["lap_number"], driver_id] = (
            kwargs["gap_to_car_ahead"], kwargs["overtake_mode_active"],
        )
        return 90.0 if driver_id == "A" else 100.0

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", calculate_lap_time)
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
    monkeypatch.setattr(simulator, "_process_overtakes", lambda *args, **kwargs: 0)
    if reverse_iteration:
        update_positions = simulator._update_positions

        def reverse_states(states: list[DriverRaceState]) -> None:
            update_positions(states)
            states.reverse()

        monkeypatch.setattr(simulator, "_update_positions", reverse_states)

    results = simulator.simulate_race(
        drivers, cars, _track(total_laps=4), Weather(), ["A", "B"],
    )

    for lap in range(1, 5):
        assert observations[lap, "A"] == (None, False)
        assert observations[lap, "B"] == (10.0 * (lap - 1), False)
    assert results[1].gap_to_leader == pytest.approx(40.0)


def test_pit_decisions_and_lap_gaps_share_pre_stop_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(33))
    drivers = [_driver("A"), _driver("B"), _driver("C")]
    cars = {
        driver.team_id: Car(team_id=driver.team_id, team_name=driver.id)
        for driver in drivers
    }
    strategy_gaps: dict[tuple[int, str], tuple[float | None, float | None]] = {}
    lap_gaps: dict[tuple[int, str], float | None] = {}

    def should_pit(
        state: DriverRaceState, states: list[DriverRaceState],
        track: Track, lap: int, pit_window_open: bool, **kwargs: Any,
    ) -> bool:
        ahead = simulator._get_gap_to_car_ahead(state, states)
        behind = simulator._get_gap_to_car_behind(state, states)
        strategy_gaps[lap, state.driver.id] = (ahead, behind)
        return lap == 2 and (
            state.driver.id == "A" or (state.driver.id == "B" and ahead > 5.0)
        )

    def calculate_lap_time(**kwargs: Any) -> float:
        driver_id = kwargs["driver"].id
        lap_gaps[kwargs["lap_number"], driver_id] = kwargs["gap_to_car_ahead"]
        return {"A": 90.0, "B": 100.0, "C": 110.0}[driver_id]

    monkeypatch.setattr(simulator, "_should_pit", should_pit)
    monkeypatch.setattr(simulator, "_execute_pit_stop", lambda *args, **kwargs: 25.0)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", calculate_lap_time)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
    monkeypatch.setattr(simulator, "_process_overtakes", lambda *args, **kwargs: 0)

    results = simulator.simulate_race(
        drivers, cars, _track(total_laps=3), Weather(), ["A", "B", "C"],
    )

    assert strategy_gaps[2, "A"] == (None, 10.0)
    assert strategy_gaps[2, "B"] == (10.0, 10.0)
    assert strategy_gaps[2, "C"] == (10.0, None)
    assert lap_gaps[2, "B"] == 10.0
    assert lap_gaps[2, "C"] == 10.0
    assert {result.driver_id: result.pit_stops for result in results} == {
        "A": 1, "B": 1, "C": 0,
    }


def test_event_manager_separates_common_active_aero_from_mode_eligibility() -> None:
    manager = EventManager(rng=np.random.default_rng(34))
    assert manager.is_active_aero_allowed()
    assert not manager.is_overtake_mode_allowed(1)
    assert manager.is_overtake_mode_allowed(2)

    manager.safety_car_active = True
    assert not manager.is_active_aero_allowed()
    assert not manager.is_overtake_mode_allowed(2)
    manager.safety_car_active = False
    manager.red_flag_active = True
    assert not manager.is_active_aero_allowed()
    assert not manager.is_overtake_mode_allowed(2)


@pytest.mark.parametrize("successful_pass", [False, True])
def test_finishing_clocks_follow_blocked_and_successful_passes(
    monkeypatch: pytest.MonkeyPatch, successful_pass: bool,
) -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(36))
    drivers = [_driver("A"), _driver("B"), _driver("C")]
    cars = {
        driver.team_id: Car(team_id=driver.team_id, team_name=driver.id)
        for driver in drivers
    }
    # In the successful case B passes despite a slightly slower provisional
    # lap clock. In the blocked case B and C cannot bank their excess pace.
    pace = {"A": 100.0, "B": 100.5 if successful_pass else 99.0, "C": 98.0}
    attempts: list[str] = []
    lap_start_gaps: list[float | None] = []

    def calculate_lap_time(**kwargs: Any) -> float:
        lap_start_gaps.append(kwargs["gap_to_car_ahead"])
        return pace[kwargs["driver"].id]

    def attempt_overtake(**kwargs: Any) -> tuple[bool, bool]:
        attacker_id = kwargs["attacker"].id
        attempts.append(attacker_id)
        return successful_pass and attacker_id == "B", False

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", calculate_lap_time)
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
    monkeypatch.setattr(
        simulator.overtaking_model, "should_attempt_overtake", lambda *args: True,
    )
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", attempt_overtake)

    results = simulator.simulate_race(
        drivers, cars, _track(total_laps=3), Weather(), ["A", "B", "C"],
    )

    assert "B" in attempts
    assert [result.driver_id for result in results] == (
        ["B", "A", "C"] if successful_pass else ["A", "B", "C"]
    )
    constrained_pace = 100.5 if successful_pass else 100.0
    assert [result.total_time for result in results] == pytest.approx(
        [3 * constrained_pace] * 3,
    )
    # Waiting time is real lap time, so it cannot win a spurious fastest lap.
    assert [result.fastest_lap for result in results] == pytest.approx(
        [constrained_pace] * 3,
    )
    gaps = [result.gap_to_leader for result in results]
    assert gaps == sorted(gaps)
    assert all(gap >= 0.0 for gap in gaps)
    assert all(
        result.gap_to_leader == result.total_time - results[0].total_time
        for result in results
    )
    assert all(gap is None or gap == 0.0 for gap in lap_start_gaps)


@pytest.mark.parametrize("difficulty", [0.5, 1.0])
@pytest.mark.parametrize("faster_followers", [False, True])
@pytest.mark.parametrize("pit_losses", [{}, {"A": 25.0}, {"A": 40.0, "B": 25.0}])
def test_pit_batch_preserves_actual_loss_and_nonpitter_order(
    monkeypatch: pytest.MonkeyPatch, difficulty: float,
    faster_followers: bool, pit_losses: dict[str, float],
) -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(36))
    drivers = [_driver("A"), _driver("B"), _driver("C")]
    cars = {
        driver.team_id: Car(team_id=driver.team_id, team_name=driver.id)
        for driver in drivers
    }
    pace = {"A": 100.0, "B": 99.0, "C": 98.0} if faster_followers else dict.fromkeys(
        ["A", "B", "C"], 100.0,
    )

    def should_pit(
        state: DriverRaceState, states: list[DriverRaceState],
        track: Track, lap: int, pit_window_open: bool, **kwargs: Any,
    ) -> bool:
        return lap == 2 and state.driver.id in pit_losses

    monkeypatch.setattr(simulator, "_should_pit", should_pit)
    monkeypatch.setattr(
        simulator, "_execute_pit_stop",
        lambda state, *args, **kwargs: pit_losses[state.driver.id],
    )
    monkeypatch.setattr(
        simulator.lap_simulator, "calculate_lap_time",
        lambda **kwargs: pace[kwargs["driver"].id],
    )
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
    monkeypatch.setattr(simulator, "_process_overtakes", lambda *args, **kwargs: 0)

    results = simulator.simulate_race(
        drivers, cars, _track(total_laps=2, overtake_difficulty=difficulty),
        Weather(), ["A", "B", "C"],
    )

    if not pit_losses:
        expected_order = ["A", "B", "C"]
        expected_clocks = [200.0] * 3
    elif "B" not in pit_losses:
        expected_order = ["B", "C", "A"]
        # C cannot automatically pass B even when its provisional clock is
        # faster. Neither nonpitter inherits A's time spent in the pit lane.
        expected_clocks = [100.0 + pace["B"], 100.0 + pace["B"], 225.0]
    else:
        expected_order = ["C", "B", "A"]
        expected_clocks = [100.0 + pace["C"], 125.0 + pace["B"], 240.0]
    assert [result.driver_id for result in results] == expected_order
    assert [result.total_time for result in results] == pytest.approx(expected_clocks)
    assert [result.position for result in results] == [1, 2, 3]
    assert [result.gap_to_leader for result in results] == pytest.approx(
        [clock - expected_clocks[0] for clock in expected_clocks],
    )


def test_full_field_finisher_gaps_are_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    drivers = [_driver(f"D{index:02d}") for index in range(20)]
    cars = {
        driver.team_id: Car(
            team_id=driver.team_id, team_name=driver.id, reliability=1.0,
        )
        for driver in drivers
    }
    simulator = RaceSimulator(rng=np.random.default_rng(36))
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])

    results = simulator.simulate_race(
        drivers, cars, _track(total_laps=50), Weather(), [driver.id for driver in drivers],
    )

    assert len(results) == 20
    assert all(result.status == DriverStatus.FINISHED for result in results)
    clocks = [result.total_time for result in results]
    gaps = [result.gap_to_leader for result in results]
    assert clocks == sorted(clocks)
    assert gaps == sorted(gaps)
    assert all(gap >= 0.0 for gap in gaps)
    assert gaps == pytest.approx([clock - clocks[0] for clock in clocks])


def _race_result(driver_id: str, position: int, status: DriverStatus) -> RaceResult:
    return RaceResult(
        driver_id=driver_id,
        driver_name=driver_id,
        team=driver_id,
        position=position,
        total_time=float(position * 90),
        gap_to_leader=0.0,
        pit_stops=0,
        fastest_lap=90.0,
        status=status,
    )


def test_monte_carlo_excludes_dnfs_from_awards_and_tracks_p21_p22() -> None:
    drivers = [
        _driver("DNF", team_id="dnf"),
        _driver("P21", team_id="p21"),
        _driver("P22", team_id="p22"),
    ]
    cars = {
        driver.team_id: Car(team_id=driver.team_id, team_name=driver.id)
        for driver in drivers
    }
    runner = MonteCarloRunner(drivers, cars, _track(), Weather(), seed=29)
    stats = runner._aggregate_statistics(
        [[
            _race_result("DNF", 1, DriverStatus.DNF),
            _race_result("P21", 21, DriverStatus.FINISHED),
            _race_result("P22", 22, DriverStatus.FINISHED),
        ]],
        [[]],
    )

    dnf = stats["DNF"]
    assert dnf.wins == 0
    assert dnf.podiums == 0
    assert dnf.points_finishes == 0
    assert dnf.total_points == 0.0
    assert dnf.dnfs == 1

    assert stats["P21"].best_position == 21
    assert stats["P21"].worst_position == 21
    assert stats["P22"].best_position == 22
    assert stats["P22"].worst_position == 22


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "10"])
def test_monte_carlo_rejects_non_positive_integer_simulation_counts(invalid: object) -> None:
    with pytest.raises(ValueError, match="num_simulations must be greater than 0"):
        MonteCarloRunner.run(
            cast(MonteCarloRunner, object()),
            num_simulations=invalid,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "2"])
def test_monte_carlo_rejects_invalid_worker_counts(invalid: object) -> None:
    with pytest.raises(ValueError, match="max_workers must be greater than 0"):
        MonteCarloRunner.run(
            cast(MonteCarloRunner, object()),
            num_simulations=1,
            max_workers=invalid,  # type: ignore[arg-type]
        )


def test_monte_carlo_parallel_consumes_each_worker_result() -> None:
    drivers = [_driver("A", team_id="a"), _driver("B", team_id="b")]
    cars = {
        driver.team_id: Car(team_id=driver.team_id, team_name=driver.id, reliability=1.0)
        for driver in drivers
    }
    runner = MonteCarloRunner(drivers, cars, _track(total_laps=8), Weather(), seed=30)

    results = runner.run(num_simulations=2, parallel=True, max_workers=2)

    assert len(results.race_results) == 2
    assert len(results.qualifying_results) == 2
