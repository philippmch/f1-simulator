"""Compare synthetic dry stint choices against the actual deterministic lap model.

No live data is fetched. Traffic, incidents and pit loss are excluded: this
compares fresh compounds at the same already-decided stop, not pit schedules.
"""

import json

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
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


if __name__ == "__main__":
    print(json.dumps(compare_stints(), indent=2))
