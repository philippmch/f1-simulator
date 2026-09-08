"""Compare synthetic dry stint choices against the actual deterministic lap model.

No live data is fetched. Traffic, incidents and pit loss are excluded: this
compares fresh compounds at the same already-decided stop, not pit schedules.
"""

import json

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.models.track import ActiveAeroZone
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.race import DriverRaceState, RaceSimulator


class MeanPace:
    def normal(self, _mean, _std):
        return 0.0


def compare_stints() -> list[dict]:
    rows = []
    for stress in (0.3, 0.9):
        for horizon in (5, 20, 45):
            driver = Driver(id="synthetic", name="Synthetic", team_id="synthetic")
            car = Car(team_id="synthetic", team_name="Synthetic")
            track = Track(id="synthetic", name="Synthetic", country="Synthetic",
                          total_laps=60, base_lap_time=90, tire_stress=stress)
            state = DriverRaceState(
                driver=driver, car=car, position=1,
                current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
            )
            simulator = RaceSimulator(rng=np.random.default_rng(42))
            chosen = simulator._choose_compound_for_next_stint(state, track, 61 - horizon)
            lap_model = LapSimulator(rng=MeanPace())
            totals = {}
            for compound in (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD):
                total = 0.0
                for age in range(horizon):
                    driver.current_tire_laps = age
                    total += lap_model.calculate_lap_time(
                        driver, car, track, TIRE_COMPOUNDS[compound], Weather(),
                        61 - horizon + age, 60,
                    )
                totals[compound.value] = total
            rows.append({
                "tire_stress": stress, "stint_laps": horizon, "chosen": chosen.value,
                "total_seconds": {key: round(value, 3) for key, value in totals.items()},
                "choice_cost_vs_fastest_seconds": round(
                    totals[chosen.value] - min(totals.values()), 3,
                ),
            })
    return rows


def compare_damp_fallbacks() -> list[dict]:
    """Exercise high-aero clipping and a shortened horizon with original fuel."""
    rows = []
    for finish in (90, 100):
        physical = 100
        lap = 81
        driver = Driver(id="synthetic", name="Synthetic", team_id="synthetic", skill_rating=1)
        car = Car(team_id="synthetic", team_name="Synthetic")
        track = Track(
            id="synthetic", name="Synthetic high aero", country="Synthetic",
            total_laps=finish, base_lap_time=70,
            active_aero_zones=[ActiveAeroZone(zone_id=i+1, sector=1, time_gain=1)
                               for i in range(6)],
        )
        weather = Weather(track_wetness=.08, change_probability=0)
        state = DriverRaceState(
            driver=driver, car=car, position=1, tire_laps=20,
            current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
            tire_compound_history=["soft", "medium"], pit_stops=1,
        )
        simulator = RaceSimulator(rng=np.random.default_rng(4))
        simulator._execute_pit_stop(
            state, track, weather, lap, sample_service=False, physical_total_laps=physical,
        )
        chosen = state.current_tire.compound
        totals = {}
        for compound in (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD):
            projected = weather.model_copy(deep=True)
            probe = driver.model_copy(deep=True)
            total = 0.0
            for age in range(finish-lap+1):
                probe.current_tire_laps = age
                total += simulator.lap_simulator.calculate_lap_time(
                    probe, car, track, TIRE_COMPOUNDS[compound], projected, lap+age, physical,
                    sample_variation=False,
                )
                projected = projected.project_surface()
            totals[compound.value] = total
        tolerance = .05 * (finish-lap+1)
        extra = totals[chosen.value] - min(totals.values())
        assert extra <= tolerance + 1e-9, (finish, chosen, totals)
        rows.append({
            "planning_final_lap": finish, "physical_fuel_laps": physical,
            "stop_lap": lap, "chosen": chosen.value,
            "total_seconds": {key: round(value, 6) for key, value in totals.items()},
            "choice_cost_vs_fastest_seconds": round(extra, 6),
            "allowed_style_cost_seconds": tolerance,
        })
    return rows


if __name__ == "__main__":
    print(json.dumps({"ordinary_stints": compare_stints(),
                      "damp_floor_cases": compare_damp_fallbacks()}, indent=2))
