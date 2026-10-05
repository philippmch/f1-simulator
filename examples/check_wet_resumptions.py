"""Show native compulsory-wet running and physical-set use without live feeds."""

import json

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import native_physics


def main():
    evidence = {}
    for engine in ("standard", "chronological"):
        keys = ("A", "B")
        result = MonteCarloRunner(
            [Driver(id=key, name=key, team_id=key, consistency=1.) for key in keys],
            {key: Car(team_id=key, team_name=key, reliability=1.) for key in keys},
            Track(id="test", name="Wet resumption example", country="Synthetic",
                  total_laps=8, base_lap_time=90., safety_car_probability=0.),
            Weather(change_probability=0.), seed=91, race_engine=engine,
            rng_policy="isolated_race_v1", starting_tires={key: "medium" for key in keys},
            tire_inventory={key: [
                {"id": "M", "compound": "medium"}, {"id": "H", "compound": "hard"},
                {"id": "W", "compound": "wet", "age": 9, "remaining_laps": 1},
            ] for key in keys},
            pit_plans={key: [] for key in keys},
            control_schedule=[{"lap": 2, "control": "red_flag", "action": "resume_wet"}],
        ).run(1, parallel=False)
        assert native_physics()
        drivers = {}
        for row in result.race_results[0]:
            wet = next(stint for stint in row.tire_set_history if stint["set_id"] == "W")
            assert (wet["age_at_fit"], wet["age_at_end"], wet["laps_used"]) == (9, 10, 1)
            assert wet["kind"] == "red_flag" and wet["remaining_laps_at_end"] == 0
            drivers[row.driver_id] = {"full_wet_stint": wet, "paid_pit_laps": row.pit_laps}
        evidence[engine] = {"schema_version": result.input_snapshot["schema_version"],
                            "control_history": result.control_schedule_histories,
                            "drivers": drivers}
    print(json.dumps({"native": native_physics(), "engines": evidence}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
