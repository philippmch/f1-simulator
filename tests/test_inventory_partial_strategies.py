"""Incomplete elective plans agree with independently enumerated physical uses."""

from copy import deepcopy
from dataclasses import replace
from math import inf, isfinite

import pytest
from test_controlled_dry_strategy import single_car_context
from test_incomplete_inventory_continuations import DIAGNOSTIC

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_finish import ChronologicalFinishContext
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race_timing import RaceFinishTimeline
from f1sim.simulation.strategy_control_clock import ObservedStandardField, StrategyControlContext
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext


def cases(name):
    records, water, age, forecast = {
        "drying": ([dict(id="I", compound="intermediate", remaining_laps=4),
                    dict(id="W", compound="wet", remaining_laps=1)], .21, 1, None),
        "dry_rule": ([dict(id="H", compound="hard", age=50, remaining_laps=4),
                      dict(id="fresh", compound="hard", remaining_laps=4)], 0., 51, None),
        "limited_dry": ([dict(id="H", compound="hard", remaining_laps=1),
                         dict(id="S", compound="soft", remaining_laps=2)], 0., 0, None),
        "scheduled": ([dict(id="I", compound="intermediate", remaining_laps=2),
                       dict(id="W", compound="wet", remaining_laps=2),
                       dict(id="H", compound="hard", remaining_laps=1)], .3, 0,
                      WeatherForecastContext.from_schedule([
                          dict(lap=2, rain_intensity=.85), dict(lap=4, rain_intensity=0.)])),
        "finishable": ([dict(id="I", compound="intermediate", remaining_laps=2),
                        dict(id="W", compound="wet", remaining_laps=1),
                        dict(id="H", compound="hard", remaining_laps=7)], .21, 0, None),
    }[name]
    driver = Driver(id="A", name="Synthetic", team_id="A")
    car = Car(team_id="A", team_name="Synthetic")
    track = Track(id="T", name="Synthetic", country="Test", total_laps=7,
                  base_lap_time=90., pit_lane_delta=20.)
    stock = TireInventory.from_sets(records)
    stock.fit(records[0]["id"])
    return (driver, car, track, Weather(track_wetness=water, change_probability=0), stock,
            age, forecast)


def independent_schedules(models, options):
    """Enumerate IDs and actual crossings, without any planner or pruning bound.

    External weather advances at explicitly counted events, including paid
    delays and fitting fees. A supplied observed field crosses every lap, even
    after control ends; no green strategy delegation supplies this oracle.
    """
    driver, car, track, weather, inventory = deepcopy(models)
    physics = LapSimulator()
    ids = tuple(key for key in inventory.sets if key not in inventory.unavailable_ids)
    ages = {key: item.age for key, item in inventory.sets.items()}
    initial = inventory.current_set_id
    ages[initial] = options["tire_age"]
    ends = {key: inf if item.remaining_laps is None else item.age + item.remaining_laps
            for key, item in inventory.sets.items()}
    surfaces = [weather]
    forecast = options.get("forecast_context")
    clock = options.get("weather_clock")
    observed = options.get("control_context")
    events = (() if clock is None else clock.update_offsets if clock.update_offsets is not None
              else tuple(clock.first_update_after + i * clock.update_interval
                         for i in range(clock.max_updates)))
    unavailable = (False, -1, -inf)
    best = {False: unavailable, True: unavailable}
    first_costs = {}
    current_lap = options.get("current_lap", 1)
    horizon = track.total_laps - current_lap + 1

    def legal(used):
        return (not options.get("require_compound_rule", True)
                or bool(used & {TireCompound.WET, TireCompound.INTERMEDIATE}) or len(used) > 1)

    def surface(count):
        while len(surfaces) <= count:
            previous = surfaces[-1]
            surfaces.append(previous.project_surface() if forecast is None else
                            forecast.advanced(len(surfaces) - 1).project_next(previous))
        return surfaces[count]

    def weather_at(offset, delay, field):
        if field is not None:
            return surface(field.updates)
        if clock is not None:
            time = clock.lap_start_offsets[offset] + delay
            return surface(sum(event <= time for event in events))
        return surface(options.get("weather_intervals", tuple(range(horizon)))[offset])

    def record(finished, offset, total, first, stopped):
        if not offset:
            return
        score = finished, offset, -total
        best[stopped] = max(best[stopped], score)
        if options.get("free_fit", False) and first == initial:
            best[False] = max(best[False], score)
        if stopped:
            first_costs[first] = max(first_costs.get(first, unavailable), score)

    def visit(offset, current, wear, used, left, dry, damp, delay, total, first, stopped, field):
        finished = offset == horizon or field is not None and field.finished
        record(finished and legal(used), offset, total, first, stopped)
        if finished:
            return
        before = weather_at(offset, delay, field)
        for target in ids:
            if wear[target] >= ends[target]:
                continue
            compound = inventory.sets[target].compound
            changing = target != current
            free = offset == 0 and options.get("free_fit", False)
            if before.tire_mismatch(compound) == "critical":
                continue
            if offset == 0 and options.get("force_stop", False) and not changing and not free:
                continue
            if changing and not free and current in ids:
                old = inventory.sets[current].compound
                allowance = dry if before.track_wetness < .08 and before.rain_intensity < .15 \
                    else damp
                elective = left > 0 and (old in (TireCompound.WET, TireCompound.INTERMEDIATE)
                    or before.track_wetness > .3 or allowance is None or allowance > 0)
                forced = (wear[current] >= ends[current]
                          or before.tire_mismatch(old) == "critical"
                          or offset == 0 and options.get("force_stop", False))
                correction = not legal(used) and compound not in used
                if not (elective or forced or correction):
                    continue
            after_used = used | {compound}
            # Execution rejects a known illegal last crossing before service.
            if offset + 1 == horizon and not legal(after_used):
                continue
            paid = changing and not free
            branch = field.fork() if field is not None else None
            after_delay = delay
            if paid and clock is not None:
                after_delay += clock.current_stop_delay if offset == 0 else clock.future_stop_delay
            fee = (options.get("tire_warmup", {}).get(compound.value, 0.)
                   if changing or offset == 0 and options.get("current_fit_pending", False)
                   else 0.)
            driver.current_tire_laps = wear[target]
            if branch is not None:
                factor = .55 if branch.controlled and branch.safety_car else \
                    .75 if branch.controlled else 1.
                stop = (observed.current_stop_delay if offset == 0 else
                        track.pit_lane_delta * factor + expected_stationary_time(car))
                branch.enter(stop if paid else None)
                reference = (1. if type(branch) is ObservedStandardField else
                             branch.free_paces[branch.identifier] * branch.running_modifier)
                gap, aero = branch.gap_ahead(reference), not branch.controlled
            else:
                gaps = options.get("current_traffic_gaps")
                gap = gaps[int(paid)] if gaps is not None and offset == 0 else None
                aero = options.get("active_aero_enabled", True) if offset == 0 else True
            running_surface = weather_at(offset, after_delay, branch)
            running = physics.calculate_lap_time(
                driver, car, track, TIRE_COMPOUNDS[compound], running_surface,
                current_lap + offset, options.get("physical_total_laps", track.total_laps),
                sample_variation=False, active_aero_enabled=aero, gap_to_car_ahead=gap)
            if branch is not None:
                branch.cross(running, fee)
                if branch.finished and not legal(after_used):
                    continue
                value = branch.now - field.now
            else:
                value = running * (options.get("current_lap_time_modifier", 1.)
                                   if offset == 0 else 1.) + fee
                if paid:
                    value += track.pit_lane_delta * (options.get("pit_lane_factor", 1.)
                                                     if offset == 0 else 1.) \
                        + expected_stationary_time(car)
                    if offset == 0:
                        value += options.get("additional_current_stop_cost", 0.)
            updated = dict(wear)
            updated[target] += 1
            visit(offset + 1, target, updated, after_used, max(0, left - paid),
                  None if dry is None else max(0, dry - paid),
                  None if damp is None else max(0, damp - paid),
                  after_delay + fee, total + value, target if offset == 0 else first,
                  (changing or free) if offset == 0 else stopped, branch)

    visit(0, initial, ages, set(options.get("used_compounds", ())),
          options.get("remaining_stops", 3), options.get("remaining_dry_stops"),
          options.get("remaining_damp_stops"), 0., 0., None, False,
          observed.new_field() if observed is not None else None)
    return best, first_costs


def assert_continuation(actual, expected):
    finished, laps, negative_time = expected
    assert actual.laps == laps
    assert actual.finished == finished
    assert actual.seconds == pytest.approx(-negative_time, abs=1.e-8)
    assert actual.cost == (pytest.approx(-negative_time, abs=1.e-8) if finished else inf)


@pytest.mark.parametrize("name", ["drying", "dry_rule", "limited_dry", "scheduled", "finishable"])
@pytest.mark.parametrize("cadence,budget,free", [
    (cadence, budget, free)
    for cadence in ("own", "external", "standard_sc", "standard_vsc",
                    "chronological_sc", "chronological_vsc")
    for budget, free in ((0, False), (2, False), (0, True))
    if not free or cadence in {"own", "external"}
])
def test_partial_distance_time_and_legal_finish_match_all_physical_schedules(name, cadence,
                                                                          budget, free):
    driver, car, track, weather, stock, age, forecast = cases(name)
    options = dict(tire_age=age, remaining_stops=budget,
                   remaining_dry_stops=budget, remaining_damp_stops=budget,
                   physical_total_laps=20, free_fit=free, current_fit_pending=True,
                   tire_warmup={"intermediate": 3., "wet": 7., "hard": .2, "soft": .4},
                   current_traffic_gaps=(.3, .8), current_lap_time_modifier=1.2,
                   active_aero_enabled=False, pit_lane_factor=.75,
                   additional_current_stop_cost=11., forecast_context=forecast)
    if cadence == "external":
        green = track.pit_lane_delta + expected_stationary_time(car)
        options["weather_clock"] = StrategyWeatherClock(
            tuple(90. * i for i in range(track.total_laps)), 35., 100., 10,
            track.pit_lane_delta * .75 + expected_stationary_time(car) + 11., green)
    elif cadence != "own":
        engine, control = cadence.split("_")
        observed = single_car_context(track, car, control, 2)
        if engine == "chronological":
            field = ChronologicalFinishContext(
                "A", RaceFinishTimeline(20, ["A"]), ("A",), (), 90.,
                1.4 if control == "sc" else 1.2, control == "sc", control_intervals=2)
        else:
            field = replace(observed.field, stop_delay=observed.current_stop_delay + 11.)
        options["control_context"] = StrategyControlContext(
            field, 50., observed.current_stop_delay + 11.)
    models = driver, car, track, weather, stock
    before = deepcopy((models, options))
    expected, costs = independent_schedules(models, options)
    result = plan_inventory_strategy(driver, car, track, weather, stock, 1, **options)
    assert_continuation(result.continuation(True), expected[True])
    assert_continuation(result.continuation(False), expected[False])
    if expected[True][1] < 0:
        assert result.set_id is None
    else:
        assert costs[result.set_id] == pytest.approx(expected[True], abs=1.e-8)
    assert result.should_pit() == (expected[True] > expected[False])
    assert models[:-1] == before[0][:-1]
    assert stock.__dict__ == before[0][-1].__dict__
    assert options == before[1]


@pytest.mark.parametrize("warmup", [58., 60.])
def test_legal_shorter_timed_finish_precedes_longer_retirement(warmup):
    driver, car, track, weather, _, _, _ = cases("drying")
    weather.track_wetness = .23
    track.total_laps = 8
    stock = TireInventory.from_sets([
        dict(id="old", compound="hard"),
        dict(id="I", compound="intermediate", remaining_laps=3),
        dict(id="W", compound="wet", age=1000, remaining_laps=2)])
    stock.fit("old")
    stock.mark_current_unavailable(0)
    timeline = RaceFinishTimeline(8, ["A"])
    timeline.observe_crossing("A", 1, 3300., is_leader=True)
    timeline.observe_crossing("A", 2, 6800., is_leader=True)
    context = ChronologicalFinishContext("A", timeline, ("A",), (), 90., 1.2, False,
                                        control_intervals=6)
    options = dict(current_lap=3, tire_age=0, remaining_stops=0,
                   physical_total_laps=8, force_stop=True, tire_warmup={"wet": warmup},
                   control_context=StrategyControlContext(
                       context, 6890., track.pit_lane_delta * .75 + expected_stationary_time(car)))
    models = driver, car, track, weather, stock
    expected, costs = independent_schedules(models, options)
    current_lap = options.pop("current_lap")
    result = plan_inventory_strategy(*models, current_lap, **options)
    assert costs["W"][0] and not costs["I"][0]
    assert costs["W"][1] < costs["I"][1]
    assert result.set_id == "W" and result.should_pit()
    assert_continuation(result.continuation(True), expected[True])
    assert isfinite(result.pit_now_cost) and result.pit_now_partial_time == inf


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("warmup", [None, {"intermediate": 3., "wet": 7.}])
def test_actual_elective_switch_preserves_the_fifth_lap_and_reuses_physical_wear(
    engine, reverse, warmup,
):
    run = DIAGNOSTIC["run_continuation"]
    options = dict(elective=True, reverse=reverse, warmup=warmup)
    selected = run(engine, **options)
    early = run(engine, forced_set="W", **options)
    held = run(engine, forced_set="hold", **options)
    assert selected["status"] == early["status"] == held["status"] == "dnf"
    assert selected["laps_completed"] == early["laps_completed"] == 5
    assert held["laps_completed"] == 4
    assert selected["total_seconds"] == pytest.approx(early["total_seconds"], abs=1.e-8)
    assert selected["pit_laps"] == [2, 3]
    first, wet, reused = selected["tire_set_history"]
    assert [stint["set_id"] for stint in (first, wet, reused)] == ["I", "W", "I"]
    assert first["laps_used"] == wet["laps_used"] == 1
    assert reused["age_at_fit"] == 1 and reused["remaining_laps_at_fit"] == 3
    assert reused["laps_used"] == 3 and reused["remaining_laps_at_end"] == 0
    assert all(item["remaining_laps"] == 0 for item in selected["tire_inventory"])
    stop, correction = selected["pit_stop_details"]
    assert stop["decision_reason"] == "inventory_forecast"
    assert stop["forecast_saving_seconds"] is None  # Neither forecast could finish.
    assert correction["decision_reason"] == "critical_weather"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("plan", [[], [dict(lap=2, compound="wet")]])
def test_custom_requests_keep_their_elective_authority(engine, plan):
    result = DIAGNOSTIC["run_continuation"](engine, elective=True, plan=plan)
    assert result["status"] == "dnf"
    assert result["laps_completed"] == (5 if plan else 4)
    assert result["pit_laps"] == ([2, 3] if plan else [])
    if plan:
        assert result["pit_plan_history"][0]["status"] == "executed"


def test_elective_diagnostic_reports_zero_distance_and_time_gaps():
    rows = DIAGNOSTIC["compare_elective_continuations"]()
    assert {row["engine"] for row in rows} == {"standard", "chronological"}
    for row in rows:
        assert row["best_alternative"] == "W" and row["distance_gap"] == 0
        assert row["time_gap_seconds"] == pytest.approx(0., abs=1.e-8)
        assert row["selected"]["laps_completed"] == 5
        assert row["alternatives"]["hold"]["laps_completed"] == 4


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_elective_partial_policy_survives_workers_and_saved_replay(tmp_path, engine):
    from test_inventory_analysis import save_result

    from f1sim.analysis.montecarlo import MonteCarloRunner
    from f1sim.analysis.replay import replay_saved_simulation

    driver, car, track, weather, _, _, _ = cases("drying")
    track.total_laps = 8
    weather.track_wetness = .24
    runner = MonteCarloRunner([driver], {"A": car}, track, weather,
                              race_engine=engine, seed=7,
                              starting_tires={"A": TireCompound.INTERMEDIATE},
                              tire_inventory={"A": list(DIAGNOSTIC["ELECTIVE_POOL"])})
    serial = runner.run(2, parallel=False)
    parallel = runner.run(2, parallel=True, max_workers=2)
    assert serial.race_results == parallel.race_results
    assert serial.input_snapshot == parallel.input_snapshot
    assert serial.input_snapshot["schema_version"] == 9
    assert any(row.laps_completed == 5 and [stint["set_id"] for stint in row.tire_set_history]
               == ["I", "W", "I"] for race in serial.race_results for row in race)
    replay = replay_saved_simulation(save_result(tmp_path, serial), simulation=2)
    assert replay.race_results[0] == serial.race_results[1]
    assert replay.input_snapshot == serial.input_snapshot


@pytest.mark.parametrize("cadence", ["own", "external"])
@pytest.mark.parametrize("free", [False, True])
def test_identical_native_partial_sets_keep_multiplicity_and_first_input_identity(cadence, free):
    driver, car, track, weather, _, _, _ = cases("drying")
    track.total_laps = 25
    weather.track_wetness = 0.
    stock = TireInventory.from_sets([
        dict(id=f"S{index}", compound="soft", remaining_laps=1) for index in range(20)])
    stock.fit("S0")
    options = dict(tire_age=1, free_fit=free, remaining_stops=0, require_compound_rule=False)
    if cadence == "external":
        green = track.pit_lane_delta + expected_stationary_time(car)
        options["weather_clock"] = StrategyWeatherClock(
            tuple(90. * i for i in range(25)), 50., 100., 25, green, green)
    result = plan_inventory_strategy(driver, car, track, weather, stock, 1, **options)
    assert result.set_id == "S1" and result.pit_now_laps == 19
    assert result.pit_now_cost == inf and isfinite(result.pit_now_partial_time)
    assert result.wait_laps == -1 and result.should_pit()
    assert stock.current_remaining_laps(1) == 0
    assert all(item.remaining_laps == 1 for item in stock.sets.values())


@pytest.mark.parametrize("cadence", ["own", "external", "standard_sc", "chronological_vsc"])
def test_partial_public_physics_cannot_mutate_inputs_or_shared_tires(monkeypatch, cadence):
    driver, car, track, weather, stock, age, _ = cases("drying")
    options = dict(tire_age=age, physical_total_laps=20, remaining_stops=2)
    if cadence == "external":
        green = track.pit_lane_delta + expected_stationary_time(car)
        options["weather_clock"] = StrategyWeatherClock(
            tuple(90. * i for i in range(7)), 35., 100., 10, green, green)
    elif cadence != "own":
        engine, control = cadence.split("_")
        context = single_car_context(track, car, control, 2)
        if engine == "chronological":
            field = ChronologicalFinishContext(
                "A", RaceFinishTimeline(20, ["A"]), ("A",), (), 90.,
                1.2, False, control_intervals=2)
            context = StrategyControlContext(field, 50., context.current_stop_delay)
        options["control_context"] = context

    def mutate(self, driver, car, track, tire, surface, *args, **kwargs):
        age = driver.current_tire_laps
        driver.current_tire_laps = 1000
        car.pit_stop_avg = 99.
        track.pit_lane_delta = 999.
        tire.initial_grip = .8
        surface.track_wetness = 0.
        return 100. + age

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", mutate)
    before = deepcopy((driver, car, track, weather, stock.__dict__, TIRE_COMPOUNDS))
    result = plan_inventory_strategy(driver, car, track, weather, stock, 1, **options)
    assert result.set_id == "W" and result.pit_now_laps == 4
    assert result.pit_now_cost == inf and isfinite(result.pit_now_partial_time)
    assert (driver, car, track, weather, stock.__dict__, TIRE_COMPOUNDS) == before
