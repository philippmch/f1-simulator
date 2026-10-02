"""Custom compulsory choices retain observed traffic and expected box waits."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_custom_pit_replacements import execution_costs, inputs, snapshot

from f1sim.models import ActiveAeroZone, Car, Driver, TireCompound, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.custom_pit_strategy import CustomPitFinishContext
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.weather_schedule import WeatherForecastContext


def test_forced_finite_custom_choice_prices_observed_rejoin_lap():
    simulator, state, track = inputs(True, [], laps=4)
    clean = simulator._plan_inventory(state, track, Weather(), 3, force_stop=True)
    traffic = simulator._plan_inventory(state, track, Weather(), 3, force_stop=True,
                                        current_traffic_gaps=(1., .3))
    expected = execution_costs(simulator, deepcopy(state), track, Weather(), 3,
                               current_traffic_gaps=(1., .3))
    assert traffic.pit_now_cost == pytest.approx(min(expected.values())[-1], rel=0, abs=1.e-8)
    assert traffic.pit_now_cost - clean.pit_now_cost == pytest.approx(.425, rel=0, abs=1.e-8)


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("due_request", [False, True])
@pytest.mark.parametrize("control", ["green", "safety_car"])
@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("zones", [0, 14, 22])
def test_custom_context_matches_executed_laps_and_services(
    finite, free, due_request, control, scheduled, zones,
):
    plan = (([dict(lap=3, compound="hard")] if free else [])
            + [dict(lap=4, compound="soft")]
            if due_request else [])
    simulator, state, track = inputs(finite, plan, laps=5)
    state.driver.skill_rating = state.car.base_pace = state.car.straight_line_speed = 1.
    track.active_aero_zones = [ActiveAeroZone(zone_id=i + 1, sector=1, time_gain=1.)
                               for i in range(zones)]
    simulator.event_manager.safety_car_active = control == "safety_car"
    weather = Weather(track_wetness=.19 if scheduled else 0.)
    options = dict(free_fit=free, physical_total_laps=40,
                   current_traffic_gaps=(1., .3), additional_current_stop_cost=7.)
    if scheduled:
        simulator.weather_forecast_context = WeatherForecastContext.from_schedule(
            [dict(lap=4, rain_intensity=.7)], leading_lap=3,
        )
        options.update(weather_intervals=(0, 1, 3),
                       weather_clock=StrategyWeatherClock((0., 90., 180.),
                                                         90., 90., 4, 18., 11.))
    before = snapshot(state, simulator), deepcopy(track), deepcopy(weather)
    expected = execution_costs(simulator, deepcopy(state), track, weather, 3, **options)
    choice = simulator._custom_plan_replacement_choice(state, track, weather, 3, **options)
    assert (-choice.instructions, choice.cost) == pytest.approx(min(expected.values()),
                                                             rel=0, abs=1.e-8)
    assert expected[choice.set_id or choice.compound] == pytest.approx(
        min(expected.values()), rel=0, abs=1.e-8,
    )
    assert (snapshot(state, simulator), track, weather) == before


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("free", [False, True])
def test_public_lap_extensions_receive_current_gap_without_mutating_inputs(monkeypatch, finite,
                                                                         free):
    simulator, state, track = inputs(finite, [], laps=5)
    calculate = LapSimulator.calculate_lap_time
    observed = []

    def extended(self, driver, car, track, tire, weather, lap, *args, **kwargs):
        observed.append((lap, kwargs.get("gap_to_car_ahead")))
        driver.total_race_time = 12345.
        return calculate(self, driver, car, track, tire, weather, lap, *args, **kwargs)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", extended)
    before = snapshot(state, simulator)
    options = dict(free_fit=free, current_traffic_gaps=(1., .3))
    expected = execution_costs(simulator, deepcopy(state), track, Weather(), 3, **options)
    observed.clear()
    choice = simulator._custom_plan_replacement_choice(state, track, Weather(), 3, **options)
    assert choice.cost == pytest.approx(min(expected.values())[-1], rel=0, abs=1.e-8)
    assert any(lap == 3 for lap, _ in observed)
    assert all(gap == (1. if free else .3) for lap, gap in observed if lap == 3)
    assert all(gap is None for lap, gap in observed if lap > 3)
    assert snapshot(state, simulator) == before


@pytest.mark.parametrize("gaps", [(True, 0), (-1, 0), (0, float("inf")), (0,), "1,2"])
def test_custom_gap_validation_matches_other_strategy_forecasts(gaps):
    simulator, state, track = inputs(True, [])
    with pytest.raises(ValueError, match="[Tt]raffic|current_traffic_gaps"):
        simulator._custom_plan_replacement_choice(state, track, Weather(), 3,
                                                 current_traffic_gaps=gaps)


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("free", [False, True])
def test_unknown_gaps_preserve_clean_forecast_defaults(finite, free):
    simulator, state, track = inputs(finite, [], laps=5)
    default = simulator._custom_plan_replacement_choice(state, track, Weather(), 3, free_fit=free)
    explicit = simulator._custom_plan_replacement_choice(
        state, track, Weather(), 3, free_fit=free, current_traffic_gaps=(None, None),
    )
    assert explicit == default


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("queue", [0., 12.])
@pytest.mark.parametrize("deadline", [280., 300., 400.])
def test_observed_context_preserves_timed_distance_and_requested_stops(
    monkeypatch, finite, free, queue, deadline,
):
    monkeypatch.setattr("f1sim.simulation.race_timing.RACING_TIME_LIMIT_SECONDS", deadline)
    simulator, state, track = inputs(finite, [dict(lap=5, compound="hard")], laps=8)
    if finite:
        state.tire_inventory.unavailable_ids.add("H-used")
    finish = state.strategy_finish_context = CustomPitFinishContext(100., deadline)
    options = dict(free_fit=free, current_traffic_gaps=(1., .3),
                   additional_current_stop_cost=queue)
    expected = execution_costs(simulator, deepcopy(state), track, Weather(), 3,
                               finish_context=finish, **options)
    choice = simulator._custom_plan_replacement_choice(state, track, Weather(), 3, **options)
    rank = (-choice.laps - 2, -choice.instructions, choice.cost)
    assert rank == pytest.approx(min(expected.values()), rel=0, abs=1.e-8)
    assert expected[choice.set_id or choice.compound] == pytest.approx(rank, rel=0, abs=1.e-8)


def test_free_fit_and_immediate_paid_correction_keep_distinct_first_lap_costs():
    simulator, state, track = inputs(True, [], laps=3)
    state.tire_compound_history = ["medium"]
    state.tire_inventory.unavailable_ids.update({"S", "H-used", "I", "W"})
    options = dict(free_fit=True, current_traffic_gaps=(1., .3), additional_current_stop_cost=7.)
    expected = execution_costs(simulator, deepcopy(state), track, Weather(), 3, **options)
    choice = simulator._custom_plan_replacement_choice(state, track, Weather(), 3, **options)
    assert choice.set_id == "H-fresh"
    assert choice.cost == pytest.approx(expected["H-fresh"][-1], rel=0, abs=1.e-8)
    assert expected["M"][-1] - choice.cost == pytest.approx(
        track.pit_lane_delta + expected_stationary_time(state.car) + 7. + .175,
        rel=0, abs=1.e-8,
    )


@pytest.mark.parametrize("free,deadline,selected", [(False, 223., "H-fresh"),
                                                   (True, 389., "H-fresh"),
                                                   (True, 469., "M"),
                                                   (True, 480., "H-fresh")])
def test_current_traffic_can_change_timed_distance_or_request_fulfilment(
    monkeypatch, free, deadline, selected,
):
    monkeypatch.setattr("f1sim.simulation.race_timing.RACING_TIME_LIMIT_SECONDS", deadline)
    simulator, state, track = inputs(True, [dict(lap=5, compound="hard")], laps=8)
    state.tire_inventory.unavailable_ids.add("H-used")
    finish = state.strategy_finish_context = CustomPitFinishContext(100., deadline)
    options = dict(free_fit=free, additional_current_stop_cost=12.)
    clean = simulator._custom_plan_replacement_choice(state, track, Weather(), 3, **options)
    observed = simulator._custom_plan_replacement_choice(state, track, Weather(), 3,
                                                        current_traffic_gaps=(1., .3), **options)
    outcomes = execution_costs(simulator, deepcopy(state), track, Weather(), 3,
                               finish_context=finish, current_traffic_gaps=(1., .3), **options)
    assert observed.set_id == selected != clean.set_id
    assert outcomes[observed.set_id][:2] < outcomes[clean.set_id][:2]
    assert outcomes[observed.set_id] == min(outcomes.values())
    assert (-observed.laps - 2, -observed.instructions, observed.cost) == pytest.approx(
        outcomes[observed.set_id], rel=0, abs=1.e-8,
    )


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("sampled_service", [4., 100.])
def test_standard_custom_refits_keep_frozen_context_when_actual_service_differs(
    monkeypatch, finite, sampled_service,
):
    simulator, a, track = inputs(finite, [], laps=4)
    b, c = deepcopy(a), deepcopy(a)
    for item, key, clock, team in ((a, "A", 100., "team"), (b, "B", 101., "team"),
                                   (c, "C", 125.7, "other")):
        item.driver.id = item.driver.name = key
        item.car.team_id = item.driver.team_id = team
        item.car.pit_stop_avg, item.car.pit_stop_std = 2.75, .1
        item.total_time, item.position = clock, ord(key) - ord("A") + 1
        item.force_pit_next_lap = key != "C"
    track.pit_lane_delta = 20.
    frozen = [replace(item) for item in (a, b, c)]
    choose = simulator._custom_plan_replacement_choice
    forecasts = {}

    def inspect(item, track, weather, lap, **kwargs):
        choice = choose(item, track, weather, lap, **kwargs)
        forecasts[item.driver.id] = kwargs, choice
        expected = execution_costs(simulator, deepcopy(item), track, weather, lap, **kwargs)
        assert choice.cost == pytest.approx(min(expected.values())[-1], rel=0, abs=1.e-8)
        return choice

    monkeypatch.setattr(simulator, "_custom_plan_replacement_choice", inspect)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        lambda car: sampled_service)
    stopped = simulator._process_pit_stops([c, b, a], frozen, track, Weather(), 3)
    assert stopped == [a, b]
    assert forecasts["A"][0]["additional_current_stop_cost"] == 0.
    assert forecasts["B"][0]["additional_current_stop_cost"] == pytest.approx(
        expected_stationary_time(a.car) - 1.,
    )
    assert forecasts["B"][0]["current_traffic_gaps"] == pytest.approx((None, .3))
    assert b.pit_stop_details[-1]["queue_time"] == sampled_service - 1.
    assert all(item.pit_stops == 1 for item in (a, b))
    assert c.pit_stops == 0


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("sampled_service", [4., 100.])
def test_chronological_custom_refits_use_observed_context_before_service(
    monkeypatch, finite, sampled_service,
):
    simulator, _, track = inputs(laps=8)
    engine = ChronologicalRace(simulator)
    track.pit_lane_delta = 20.
    drivers = [Driver(id=key, name=key, team_id="team") for key in "AB"]
    car = Car(team_id="team", team_name="team", pit_stop_avg=2.75, pit_stop_std=.1)
    records = {key: [dict(id=c.value, compound=c.value, age=0) for c in TireCompound]
               for key in "AB"}
    traffic = engine._strategy_traffic
    choose = simulator._custom_plan_replacement_choice
    observed, choices = {}, {}

    def inspect_traffic(state, now, delay):
        result = traffic(state, now, delay)
        observed[state.driver.id, state.laps_completed + 1] = result, delay
        return result

    def inspect_choice(state, planning, weather, lap, **kwargs):
        traffic_snapshot, delay = observed[state.driver.id, lap]
        assert kwargs["current_traffic_gaps"] == traffic_snapshot.current_traffic_gaps
        assert kwargs["additional_current_stop_cost"] == delay
        before = snapshot(state, simulator)
        choice = choose(state, planning, weather, lap, **kwargs)
        assert snapshot(state, simulator) == before
        expected = execution_costs(simulator, deepcopy(state), planning, weather, lap, **kwargs)
        assert choice.cost == pytest.approx(min(expected.values())[-1], rel=0, abs=1.e-8)
        choices[state.driver.id, lap] = choice
        return choice

    monkeypatch.setattr(engine, "_strategy_traffic", inspect_traffic)
    monkeypatch.setattr(simulator, "_custom_plan_replacement_choice", inspect_choice)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        lambda car: sampled_service)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *args, **kwargs: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *args: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *args, **kwargs: (False, False))
    rows = engine.run(drivers, {"team": car}, track, Weather(change_probability=0.), list("AB"),
                      starting_tires={key: TireCompound.WET for key in "AB"},
                      pit_plans={key: [dict(lap=4, compound="hard")] for key in "AB"},
                      tire_inventory=records if finite else None)
    assert {("A", 1), ("B", 1)} <= choices.keys()
    assert observed["B", 1][1] == expected_stationary_time(car)
    follower = next(row for row in rows if row.driver_id == "B")
    assert follower.pit_stop_details[0]["queue_time"] == sampled_service
    assert all(row.pit_plan_history[0]["status"] == "executed" for row in rows)
