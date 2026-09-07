"""Focused regressions for race realism behavior."""

import numpy as np
import pytest

from f1sim.models import (
    ActiveAeroZone,
    Car,
    Driver,
    Sector,
    Tire,
    TireCompound,
    Track,
    Weather,
)
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.models.weather import WeatherCondition
from f1sim.simulation.events import EventManager, EventType, RaceEvent
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.qualifying import QualifyingSimulator
from f1sim.simulation.race import DriverRaceState, RaceSimulator


def _track(
    *,
    sectors: list[Sector] | None = None,
    active_aero_zones: list[ActiveAeroZone] | None = None,
    tire_stress: float = 0.5,
    safety_car_probability: float = 0.3,
    total_laps: int = 50,
) -> Track:
    return Track(
        id="test",
        name="Test Circuit",
        country="Nowhere",
        total_laps=total_laps,
        base_lap_time=90.0,
        pit_lane_delta=20.0,
        sectors=sectors or [],
        active_aero_zones=(
            active_aero_zones
            if active_aero_zones is not None
            else [ActiveAeroZone(zone_id=1, sector=2, time_gain=0.4)]
        ),
        overtake_difficulty=0.5,
        tire_stress=tire_stress,
        safety_car_probability=safety_car_probability,
    )


def _driver(driver_id: str = "DRV", **updates: float | str) -> Driver:
    defaults: dict[str, float | str] = {
        "id": driver_id,
        "name": driver_id,
        "team_id": "team",
        "skill_rating": 0.8,
        "consistency": 1.0,
        "tire_management": 0.8,
    }
    defaults.update(updates)
    return Driver(**defaults)


def test_lap_time_uses_track_profile_and_car_attributes() -> None:
    corner_track = _track(
        sectors=[
            Sector(number=1, base_time=30, is_high_speed=False, overtake_opportunity=0.1),
            Sector(number=2, base_time=30, is_high_speed=False, overtake_opportunity=0.1),
            Sector(number=3, base_time=30, is_high_speed=False, overtake_opportunity=0.1),
        ]
    )
    straight_track = corner_track.model_copy(
        update={
            "sectors": [
                Sector(number=1, base_time=30, is_high_speed=True, overtake_opportunity=1.0),
                Sector(number=2, base_time=30, is_high_speed=True, overtake_opportunity=1.0),
                Sector(number=3, base_time=30, is_high_speed=True, overtake_opportunity=1.0),
            ]
        }
    )
    driver = _driver()
    tire = Tire(compound=TireCompound.MEDIUM)
    corner_car = Car(
        team_id="team",
        team_name="Team",
        base_pace=0.8,
        downforce_level=1.0,
        straight_line_speed=0.6,
    )
    straight_car = corner_car.model_copy(
        update={"downforce_level": 0.6, "straight_line_speed": 1.0}
    )

    corner_time = LapSimulator(np.random.default_rng(12)).calculate_lap_time(
        driver, corner_car, corner_track, tire, Weather(), 10, 50
    )
    corner_low_downforce_time = LapSimulator(np.random.default_rng(12)).calculate_lap_time(
        driver, straight_car, corner_track, tire, Weather(), 10, 50
    )
    straight_time = LapSimulator(np.random.default_rng(12)).calculate_lap_time(
        driver, straight_car, straight_track, tire, Weather(), 10, 50
    )
    straight_low_speed_time = LapSimulator(np.random.default_rng(12)).calculate_lap_time(
        driver, corner_car, straight_track, tire, Weather(), 10, 50
    )
    # The same specialised package is rewarded in its matching circuit
    # profile; this also proves sector opportunity/high-speed flags participate.
    assert corner_time < corner_low_downforce_time
    assert straight_time < straight_low_speed_time


def test_lap_time_uses_wet_package_and_tire_stress() -> None:
    track = _track(tire_stress=1.0)
    driver = _driver()
    driver.current_tire_laps = 15
    tire = Tire(compound=TireCompound.MEDIUM, degradation_rate=0.03)
    wet = Weather(
        condition=WeatherCondition.HEAVY_RAIN,
        track_wetness=0.8,
        rain_intensity=0.9,
    )
    dry = Weather()
    robust = Car(
        team_id="team",
        team_name="Team",
        wet_performance=1.0,
        tire_degradation_factor=0.7,
    )
    fragile = robust.model_copy(
        update={"wet_performance": 0.5, "tire_degradation_factor": 1.3}
    )

    robust_wet = LapSimulator(np.random.default_rng(8)).calculate_lap_time(
        driver, robust, track, tire, wet, 20, 50
    )
    fragile_wet = LapSimulator(np.random.default_rng(8)).calculate_lap_time(
        driver, fragile, track, tire, wet, 20, 50
    )
    robust_dry = LapSimulator(np.random.default_rng(8)).calculate_lap_time(
        driver, robust, track, tire, dry, 20, 50
    )
    fragile_dry = LapSimulator(np.random.default_rng(8)).calculate_lap_time(
        driver, fragile, track, tire, dry, 20, 50
    )
    assert fragile_wet > robust_wet
    assert fragile_dry > robust_dry
    # The wet-package gap is in addition to the dry tire-degradation gap.
    assert (fragile_wet - robust_wet) > (fragile_dry - robust_dry)


def test_overtake_mode_is_off_during_neutralizations_and_restart_delay() -> None:
    manager = EventManager(rng=np.random.default_rng(1))
    assert not manager.is_overtake_mode_allowed(1)
    assert manager.is_overtake_mode_allowed(2)

    manager.safety_car_active = True
    assert not manager.is_overtake_mode_allowed(3)
    manager.safety_car_active = False
    manager.vsc_active = True
    assert not manager.is_overtake_mode_allowed(3)
    manager.vsc_active = False
    manager.red_flag_active = True
    assert not manager.is_overtake_mode_allowed(3)
    manager.red_flag_active = False
    manager.sc_restart_lap = True
    assert not manager.is_overtake_mode_allowed(3)

    wet = Weather(condition=WeatherCondition.LIGHT_RAIN, track_wetness=0.25)
    assert not manager.is_overtake_mode_allowed(3, wet)


def test_safety_car_restart_and_red_flag_overtake_mode_timeline() -> None:
    track = _track(total_laps=8, safety_car_probability=0.0)
    manager = EventManager(rng=np.random.default_rng(30))
    manager.safety_car_active = True
    manager.safety_car_laps_remaining = 1

    # SC is still in force for the running lap that ends it; only the
    # following lap is the restart delay.
    assert not manager.is_overtake_mode_allowed(3)
    manager.process_lap(3, [], {}, track, Weather())
    assert not manager.is_overtake_mode_allowed(3)
    assert manager.is_restart_lap(4)
    assert not manager.is_overtake_mode_allowed(4)
    manager.process_lap(4, [], {}, track, Weather())
    assert manager.is_overtake_mode_allowed(5)

    manager.deploy_red_flag(5, "test")
    manager.process_lap(5, [], {}, track, Weather())
    manager.end_red_flag()
    assert manager.is_restart_lap(6)
    assert not manager.is_overtake_mode_allowed(6)
    manager.process_lap(6, [], {}, track, Weather())
    assert manager.is_overtake_mode_allowed(7)


def test_vsc_end_has_one_consistent_active_aero_and_overtake_timeline() -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(31))
    # Keep the controlled passing gap stable while checking race-control
    # eligibility; early pit decisions are covered by strategy tests.
    simulator._should_pit = lambda *args, **kwargs: False  # type: ignore[method-assign]
    driver_a = _driver("A", team_id="a")
    driver_b = _driver("B", team_id="b")
    cars = {
        "a": Car(team_id="a", team_name="A", reliability=1.0),
        "b": Car(team_id="b", team_name="B", reliability=1.0),
    }
    lap_active_aero: list[tuple[int, bool]] = []
    overtake_mode: list[tuple[int, bool]] = []
    current_lap = 0

    def fixed_lap_time(**kwargs: object) -> float:
        nonlocal current_lap
        current_lap = int(kwargs["lap_number"])
        lap_active_aero.append((current_lap, bool(kwargs["active_aero_enabled"])))
        return 90.0

    simulator.lap_simulator.calculate_lap_time = fixed_lap_time  # type: ignore[method-assign]
    simulator.overtaking_model.should_attempt_overtake = (  # type: ignore[method-assign]
        lambda *args, **kwargs: True
    )

    def capture_overtake(**kwargs: object) -> tuple[bool, bool]:
        overtake_mode.append((current_lap, bool(kwargs["overtake_mode_active"])))
        return False, False

    simulator.overtaking_model.attempt_overtake = capture_overtake  # type: ignore[method-assign]

    def fake_process_lap(**kwargs: object) -> list[RaceEvent]:
        lap = int(kwargs["lap"])
        if lap == 2:
            simulator.event_manager.vsc_active = True
            simulator.event_manager.vsc_laps_remaining = 1
        elif lap == 3:
            # VSC ends after lap 3 has been run.  The lap-3 snapshot must
            # remain disabled for both timing and overtaking.
            simulator.event_manager.vsc_active = False
        return []

    simulator.event_manager.process_lap = fake_process_lap  # type: ignore[method-assign]
    simulator.simulate_race(
        [driver_a, driver_b],
        cars,
        _track(total_laps=6, safety_car_probability=0.0),
        Weather(),
        starting_grid=["A", "B"],
    )

    assert [entry for entry in lap_active_aero if entry[0] == 3]
    assert all(not enabled for lap, enabled in lap_active_aero if lap == 3)
    assert all(enabled for lap, enabled in lap_active_aero if lap == 4)
    # Overtake Mode is not called while VSC is active and becomes deployable
    # only once green running resumes.
    assert all(not enabled for lap, enabled in overtake_mode if lap == 3)
    assert any(enabled for lap, enabled in overtake_mode if lap == 4)


def test_qualifying_classification_respects_session_progression() -> None:
    drivers = [_driver(f"D{i:02d}", team_id=f"team{i}") for i in range(20)]
    cars = {
        driver.team_id: Car(team_id=driver.team_id, team_name=driver.team_id)
        for driver in drivers
    }
    simulator = QualifyingSimulator(rng=np.random.default_rng(2))
    calls = 0

    def fake_session(
        session_drivers: list[Driver], *args: object, **kwargs: object
    ) -> dict[str, float]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {driver.id: float(index + 1) for index, driver in enumerate(session_drivers)}
        if calls == 2:
            # Q2 eliminations have excellent Q2 times, but must still start
            # behind every driver who reached Q3.
            return {
                driver.id: (float(index + 1) if index < 10 else float(20 + index))
                for index, driver in enumerate(session_drivers)
            }
        return {driver.id: float(30 + index) for index, driver in enumerate(session_drivers)}

    simulator._simulate_session = fake_session  # type: ignore[method-assign]
    results = simulator.simulate_qualifying(drivers, cars, _track(), Weather())

    assert [result.driver_id for result in results[:10]] == [f"D{i:02d}" for i in range(10)]
    assert [result.driver_id for result in results[10:15]] == [f"D{i:02d}" for i in range(10, 15)]
    assert [result.driver_id for result in results[15:]] == [f"D{i:02d}" for i in range(15, 20)]
    assert all(result.eliminated_in is None for result in results[:10])
    assert all(result.eliminated_in == "Q2" for result in results[10:15])
    assert all(result.eliminated_in == "Q1" for result in results[15:])


def test_22_car_qualifying_eliminates_six_in_q1_and_q2() -> None:
    drivers = [_driver(f"D{i:02d}", team_id=f"team{i}") for i in range(22)]
    cars = {
        driver.team_id: Car(team_id=driver.team_id, team_name=driver.team_id)
        for driver in drivers
    }
    simulator = QualifyingSimulator(rng=np.random.default_rng(22))
    results = simulator.simulate_qualifying(drivers, cars, _track(), Weather())

    assert len(results) == 22
    assert sum(result.eliminated_in == "Q1" for result in results) == 6
    assert sum(result.eliminated_in == "Q2" for result in results) == 6
    assert sum(result.q3_time is not None for result in results) == 10
    assert all(result.eliminated_in is None for result in results[:10])


def test_incident_risk_and_safety_hazard_are_calibrated() -> None:
    manager = EventManager(rng=np.random.default_rng(3))
    dry = Weather()
    wet = Weather(
        condition=WeatherCondition.LIGHT_RAIN,
        track_wetness=0.6,
        rain_intensity=0.5,
    )
    consistent = _driver("CONS", consistency=1.0, wet_skill_modifier=1.5)
    vulnerable = _driver("RISK", consistency=0.7, wet_skill_modifier=0.5)
    weights = manager._incident_driver_weights([consistent, vulnerable], wet)
    assert weights[1] > weights[0]

    no_risk = _track(safety_car_probability=0.0)
    low_risk = _track(safety_car_probability=0.1)
    high_risk = _track(safety_car_probability=0.8)
    assert manager._calibrated_safety_probs(no_risk, dry, incidents=0, lap=20) == (0.0, 0.0)
    low_sc, low_vsc = manager._calibrated_safety_probs(low_risk, dry, incidents=0, lap=20)
    high_sc, high_vsc = manager._calibrated_safety_probs(high_risk, dry, incidents=0, lap=20)
    assert 0.0 < low_sc < 0.03
    assert 0.0 < low_vsc < 0.03
    assert high_sc > low_sc
    assert high_vsc > low_vsc


def test_puncture_applies_time_loss_forces_pit_and_records_real_strategy() -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(4))
    driver_a = _driver("A", team_id="a")
    driver_b = _driver("B", team_id="b")
    cars = {
        "a": Car(team_id="a", team_name="A", reliability=1.0),
        "b": Car(team_id="b", team_name="B", reliability=1.0),
    }
    event = RaceEvent(
        event_type=EventType.PUNCTURE,
        lap=1,
        drivers_involved=["A"],
        time_loss_seconds=10.0,
        forces_pit_stop=True,
    )

    def fake_process_lap(**kwargs: object) -> list[RaceEvent]:
        return [event] if kwargs["lap"] == 1 else []

    simulator.event_manager.process_lap = fake_process_lap  # type: ignore[method-assign]
    track = _track(total_laps=8, safety_car_probability=0.0)
    results = simulator.simulate_race(
        [driver_a, driver_b],
        cars,
        track,
        Weather(),
        starting_grid=["A", "B"],
    )
    result_a = next(result for result in results if result.driver_id == "A")
    result_b = next(result for result in results if result.driver_id == "B")

    assert result_a.pit_stops >= 1
    assert result_a.total_time > result_b.total_time
    # Starting compounds are now sampled from the dry strategy distribution;
    # the regression only requires that the actual opening stint is recorded.
    assert result_a.strategy[0] in {"soft", "medium", "hard"}
    assert len(result_a.strategy) >= 2


def test_overtake_incident_deploying_sc_keeps_penalty_and_fastest_lap() -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(41))
    driver_a = _driver("A", team_id="a")
    driver_b = _driver("B", team_id="b")
    cars = {
        "a": Car(team_id="a", team_name="A", reliability=1.0),
        "b": Car(team_id="b", team_name="B", reliability=1.0),
    }
    lap_seen: list[int] = []
    bunch_snapshots: list[tuple[dict[str, float], dict[str, float]]] = []

    simulator.lap_simulator.calculate_lap_time = (  # type: ignore[method-assign]
        lambda **kwargs: 90.0
    )
    simulator.overtaking_model.should_attempt_overtake = (  # type: ignore[method-assign]
        lambda *args, **kwargs: True
    )
    simulator.overtaking_model.attempt_overtake = (  # type: ignore[method-assign]
        lambda **kwargs: (False, True)
    )

    event = RaceEvent(
        event_type=EventType.SAFETY_CAR,
        lap=1,
        duration_laps=2,
        description="Safety car after on-track incident",
    )

    def fake_process_lap(**kwargs: object) -> list[RaceEvent]:
        lap_seen.append(int(kwargs["incidents_this_lap"]))
        if kwargs["lap"] == 1:
            simulator.event_manager.safety_car_active = True
            simulator.event_manager.safety_car_laps_remaining = 2
            return [event]
        return []

    simulator.event_manager.process_lap = fake_process_lap  # type: ignore[method-assign]
    original_bunch = simulator.event_manager.bunch_field

    def capture_bunch(states: list[object]) -> None:
        before = {state.driver.id: state.total_time for state in states}  # type: ignore[attr-defined]
        original_bunch(states)
        after = {state.driver.id: state.total_time for state in states}  # type: ignore[attr-defined]
        bunch_snapshots.append((before, after))

    simulator.event_manager.bunch_field = capture_bunch  # type: ignore[method-assign]
    results = simulator.simulate_race(
        [driver_a, driver_b],
        cars,
        _track(total_laps=3, safety_car_probability=0.0),
        Weather(),
        starting_grid=["A", "B"],
    )

    assert lap_seen[0] == 1
    assert bunch_snapshots
    before, after = bunch_snapshots[0]
    ordered_after = sorted(after.values())
    assert 0.8 <= ordered_after[1] - ordered_after[0] <= 1.2
    assert {result.position for result in results} == {1, 2}
    assert all(result.fastest_lap > 90.0 for result in results)


def test_bunching_closes_clean_gap_after_incident_position_loss() -> None:
    drivers = [
        _driver("LEAD", team_id="lead"),
        _driver("VICTIM", team_id="victim"),
        _driver("CLEAN", team_id="clean"),
    ]
    states = [
        DriverRaceState(
            driver=drivers[0],
            car=Car(team_id="lead", team_name="Lead"),
            position=1,
            total_time=100.0,
        ),
        DriverRaceState(
            driver=drivers[1],
            car=Car(team_id="victim", team_name="Victim"),
            position=2,
            total_time=104.0 + 30.0,
        ),
        DriverRaceState(
            driver=drivers[2],
            car=Car(team_id="clean", team_name="Clean"),
            position=3,
            total_time=130.0,
        ),
    ]
    simulator = RaceSimulator(rng=np.random.default_rng(43))

    # The incident's elapsed-time penalty moves VICTIM behind CLEAN before
    # the safety-car gap reset.  The large ordinary gap to LEAD should still
    # be closed by bunching rather than retained via a max() safeguard.
    simulator._classify_positions_before_neutralization(states, {"VICTIM"})
    assert [state.driver.id for state in sorted(states, key=lambda s: s.position)] == [
        "LEAD",
        "CLEAN",
        "VICTIM",
    ]

    simulator.event_manager.bunch_field(states)
    ordered = sorted(states, key=lambda state: state.position)
    gaps = [
        ordered[index].total_time - ordered[index - 1].total_time
        for index in range(1, len(ordered))
    ]
    assert all(0.8 <= gap <= 1.2 for gap in gaps)
    assert ordered[-1].driver.id == "VICTIM"


def test_neutralization_preserves_authoritative_overtake_order() -> None:
    drivers = [
        _driver("PASSER", team_id="passer"),
        _driver("FORMER_LEAD", team_id="former"),
    ]
    states = [
        DriverRaceState(
            driver=drivers[0],
            car=Car(team_id="passer", team_name="Passer"),
            position=2,
            # A valid on-track pass can occur before the slower car's
            # cumulative clock catches up; position remains authoritative.
            total_time=101.0,
        ),
        DriverRaceState(
            driver=drivers[1],
            car=Car(team_id="former", team_name="Former lead"),
            position=1,
            total_time=100.0,
        ),
    ]
    simulator = RaceSimulator(rng=np.random.default_rng(47))

    simulator.overtaking_model.should_attempt_overtake = (  # type: ignore[method-assign]
        lambda *args, **kwargs: True
    )
    simulator.overtaking_model.attempt_overtake = (  # type: ignore[method-assign]
        lambda **kwargs: (True, False)
    )
    simulator._process_overtakes(
        states,
        _track(),
        Weather(),
        lap=3,
        overtake_mode_allowed=False,
    )
    assert [state.driver.id for state in sorted(states, key=lambda s: s.position)] == [
        "PASSER",
        "FORMER_LEAD",
    ]

    # There is no material incident to justify changing the result of the
    # pass.  Bunching must compact the gap without re-sorting by elapsed time.
    simulator._classify_positions_before_neutralization(states)
    simulator.event_manager.bunch_field(states)

    assert [state.driver.id for state in sorted(states, key=lambda s: s.position)] == [
        "PASSER",
        "FORMER_LEAD",
    ]

    # Red-flag bunching uses the same authoritative ordering and must not
    # reclassify the cars merely because their cumulative clocks disagree.
    simulator._handle_red_flag_stop(states, Weather(), _track(), 3)
    assert [state.driver.id for state in sorted(states, key=lambda s: s.position)] == [
        "PASSER",
        "FORMER_LEAD",
    ]


def test_final_sc_lap_suppresses_green_flag_incidents_and_intervention() -> None:
    manager = EventManager(rng=np.random.default_rng(44))
    manager.safety_car_active = True
    manager.safety_car_laps_remaining = 1
    manager.set_forced_safety_car(3)
    drivers = [_driver("A", team_id="a"), _driver("B", team_id="b")]
    cars = {
        "a": Car(team_id="a", team_name="A", reliability=1.0),
        "b": Car(team_id="b", team_name="B", reliability=1.0),
    }

    def unexpected_call(*args: object, **kwargs: object) -> None:
        pytest.fail("green-flag event sampling ran on the final SC lap")

    manager._check_random_incident = unexpected_call  # type: ignore[method-assign]
    manager._deploy_safety_measure = unexpected_call  # type: ignore[method-assign]
    events = manager.process_lap(3, drivers, cars, _track(safety_car_probability=1.0), Weather())

    assert events == []
    assert manager.safety_car_deployments == 0
    assert not manager.safety_car_active


def test_final_vsc_lap_suppresses_green_flag_incidents_and_intervention() -> None:
    manager = EventManager(rng=np.random.default_rng(45))
    manager.vsc_active = True
    manager.vsc_laps_remaining = 1
    manager.set_forced_safety_car(3)
    drivers = [_driver("A", team_id="a"), _driver("B", team_id="b")]
    cars = {
        "a": Car(team_id="a", team_name="A", reliability=1.0),
        "b": Car(team_id="b", team_name="B", reliability=1.0),
    }

    def unexpected_call(*args: object, **kwargs: object) -> None:
        pytest.fail("green-flag event sampling ran on the final VSC lap")

    manager._check_random_incident = unexpected_call  # type: ignore[method-assign]
    manager._deploy_safety_measure = unexpected_call  # type: ignore[method-assign]
    events = manager.process_lap(3, drivers, cars, _track(safety_car_probability=1.0), Weather())

    assert events == []
    assert manager.vsc_deployments == 0
    assert not manager.vsc_active


def test_repeated_compound_stints_are_preserved() -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(42))
    state = DriverRaceState(
        driver=_driver(),
        car=Car(team_id="team", team_name="Team"),
        position=1,
            current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
    )
    # Once two distinct slicks have actually been used, repeating a compound
    # is legal and should remain represented as a real new stint.
    state.tire_compound_history = ["medium", "hard"]
    simulator._choose_compound_for_next_stint = (  # type: ignore[method-assign]
        lambda *args: TireCompound.MEDIUM
    )
    simulator._execute_pit_stop(state, _track(), Weather(), current_lap=20)
    assert state.tire_compound_history == ["medium", "hard", "medium"]


def test_full_sc_race_rate_is_calibrated_and_vsc_is_separate() -> None:
    track = _track(total_laps=50, safety_car_probability=0.35)
    races = 400
    sc_races = 0
    vsc_races = 0
    for seed in range(races):
        manager = EventManager(rng=np.random.default_rng(1000 + seed))
        for lap in range(1, track.total_laps + 1):
            manager.process_lap(
                lap=lap,
                drivers=[],
                cars={},
                track=track,
                weather=Weather(),
            )
        sc_races += int(any(event.event_type == EventType.SAFETY_CAR for event in manager.events))
        vsc_races += int(
            any(event.event_type == EventType.VIRTUAL_SAFETY_CAR for event in manager.events)
        )

    # SC probability is the track's full-race input.  VSC is an independent,
    # lower-risk process (currently derived at half the SC risk).
    assert abs(sc_races / races - 0.35) < 0.08
    assert abs(vsc_races / races - 0.175) < 0.07


def test_driver_state_tire_history_tracks_actual_stop_compound() -> None:
    simulator = RaceSimulator(rng=np.random.default_rng(5))
    state = DriverRaceState(
        driver=_driver(),
        car=Car(team_id="team", team_name="Team"),
        position=1,
        current_tire=Tire(
            compound=TireCompound.MEDIUM,
            initial_grip=1.0,
            degradation_rate=0.015,
            cliff_threshold=30,
            cliff_multiplier=3.0,
        ),
    )
    simulator._choose_compound_for_next_stint = lambda *args: TireCompound.HARD  # type: ignore[method-assign]
    simulator._execute_pit_stop(state, _track(), Weather(), current_lap=20)
    assert state.tire_compound_history == ["medium", "hard"]
