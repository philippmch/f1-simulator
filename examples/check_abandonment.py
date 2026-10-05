"""Show native countback classification and early no-result outcomes offline."""

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather


def main():
    for engine in ("standard", "chronological"):
        for after in (1, 4):
            drivers = [Driver(id=key, name=key, team_id=key, consistency=1.) for key in ("A", "B")]
            cars = {key: Car(team_id=key, team_name=key, reliability=1.) for key in ("A", "B")}
            result = MonteCarloRunner(
                drivers, cars,
                Track(id="test", name="Countback example", country="Synthetic",
                      total_laps=8, base_lap_time=90., safety_car_probability=0.),
                Weather(change_probability=0.), seed=91, race_engine=engine,
                rng_policy="isolated_race_v1",
                starting_tires={key: "medium" for key in ("A", "B")},
                pit_plans={key: [] for key in ("A", "B")},
                control_schedule=[{"lap": after, "control": "red_flag", "action": "abandon"}],
            ).run(1, parallel=False)
            context = result.get_race_abandonment_context()
            print(f"\n{engine}, red flag after crossing {after}")
            print(context["description"] if context else "No verified abandonment record")
            for row in result.race_results[0]:
                penalty = (row.abandonment_tire_rule.penalty_seconds
                           if row.abandonment_tire_rule else 0)
                print(f"  {row.driver_id}: {row.status.value}, {row.laps_completed} laps, "
                      f"{row.total_time:.3f}s, +{penalty}s penalty, {row.points_awarded} points")


if __name__ == "__main__":
    main()
