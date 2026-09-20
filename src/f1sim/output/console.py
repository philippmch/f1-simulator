"""Console output formatting."""

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.output.timing import (
    finite_time,
    format_lap_deficit,
    format_seconds,
    race_suspension_seconds,
    suspension_statistics,
)
from f1sim.simulation.qualifying import QualifyingResult
from f1sim.simulation.race import RaceResult, result_is_classified
from f1sim.simulation.race_points import points_for_result


class ConsoleOutput:
    """Formats simulation results for console display."""

    @staticmethod
    def print_paired_comparison(
        results: dict[str, SimulationResults], reference_scenario: str,
        driver_id: str | None = None,
    ) -> None:
        """Show differences over comparable seeded trials with their own counts."""
        paired = paired_comparison_statistics(results, reference_scenario)
        print(f"\nPaired changes compared with {reference_scenario}")
        print("Positive points changes mean more points; "
              "positive DNF changes mean more retirements.")
        print("SE estimates sampling error, not a confidence interval. "
              "Zero SE does not prove equality.")
        ConsoleOutput._print_suspension_context(results)
        for label, comparison in paired["variants"].items():
            print(f"{label}:")
            if comparison["status"] == "unavailable":
                print(f"  Unavailable: {comparison['reason']}")
                continue
            print("Driver       Pairs  Excluded  Points change     SE   "
                  "More/equal/fewer   DNF change")
            for driver, stats in comparison["driver_statistics"].items():
                if driver_id is not None and driver != driver_id:
                    continue
                if not stats["paired_races"]:
                    print(f"{driver:<12} No usable paired results "
                          f"({stats['excluded_pairs']} excluded pairs)")
                    continue
                error = stats["points_difference_standard_error"]
                error_text = f"{error:.3f}" if error is not None else "n/a"
                counts = (f"{stats['more_points_races']}/{stats['equal_points_races']}/"
                          f"{stats['fewer_points_races']}")
                print(f"{driver:<12} {stats['paired_races']:>5} {stats['excluded_pairs']:>9} "
                      f"{stats['mean_points_difference']:>+14.3f} {error_text:>6} "
                      f"{counts:>18} {stats['dnf_rate_difference_percentage_points']:>+11.1f} pp")

    @staticmethod
    def print_qualifying_results(results: list[QualifyingResult]) -> None:
        """Print qualifying results to console.

        Args:
            results: Qualifying results sorted by position
        """
        print("\n" + "=" * 88)
        print("QUALIFYING RESULTS")
        print("=" * 88)
        print(f"{'Pos':<4} {'Driver':<20} {'Q1':<11} {'Q2':<11} {'Q3':<11} {'Best':<11} Status")
        print("-" * 88)

        ordered = sorted(results, key=lambda r: r.position)

        for result in ordered:
            time = finite_time(result.best_time)
            time_text = f"{time:.3f}s" if time is not None else "No time"
            session_times = []
            for value in (result.q1_time, result.q2_time, result.q3_time):
                session_time = finite_time(value)
                session_times.append(f"{session_time:.3f}s" if session_time is not None else "--")

            eliminated = f"out in {result.eliminated_in}" if result.eliminated_in else ""
            times_text = " ".join(f"{value:<11}" for value in session_times)

            print(
                f"{result.position:<4} "
                f"{result.driver_name:<20} "
                f"{times_text} {time_text:<11} "
                f"{eliminated}"
            )

        print("Best is the fastest lap across sessions; grid order follows session classification.")
        print("=" * 88)

    @staticmethod
    def print_race_results(results: list[RaceResult]) -> None:
        """Print race results to console.

        Args:
            results: Race results sorted by position
        """
        print("\n" + "=" * 70)
        print("RACE RESULTS")
        suspension = race_suspension_seconds(results)
        print(f"Completed race suspension: {format_seconds(suspension)}")
        print("Race-wide collection + restart pause; this elapsed-race context is "
              "already in finish clocks, not an individual driver's stopped or driving time.")
        if any(getattr(result, "race_time_limited", False) for result in results):
            print("Race shortened by the two-hour limit.")
        print("=" * 70)
        print(f"{'Pos':<4} {'Driver':<20} {'Team':<18} {'Time/Gap':<15} "
              f"{'Pits':<5} {'Laps':<5} {'Status':<25} {'Points'}")
        print("-" * 70)

        leader_time = None
        leader_laps = None
        for result in sorted(results, key=lambda r: r.position):
            if result.status.value == "finished":
                if leader_time is None:
                    leader_time = result.total_time
                    leader_laps = getattr(result, "laps_completed", None)
                    time_str = f"{result.total_time:.3f}s"
                else:
                    gap = result.total_time - leader_time
                    lap_gap = format_lap_deficit(getattr(result, "laps_completed", None),
                                                 leader_laps)
                    if lap_gap is not None:
                        time_str = lap_gap
                    elif gap < 60:
                        time_str = f"+{gap:.3f}s"
                    else:
                        mins = int(gap // 60)
                        secs = gap % 60
                        time_str = f"+{mins}:{secs:05.2f}"
            else:
                time_str = result.dnf_reason or "DNF"

            classified = result_is_classified(result)
            position = str(result.position) if classified else "NC"
            completed = getattr(result, "laps_completed", None)
            laps = str(completed) if completed is not None else "-"
            status_str = result.status.value.upper()
            if result.status.value != "finished" or not classified:
                status_str += " / " + ("Classified" if classified else "Not classified")

            print(
                f"{position:<4} "
                f"{result.driver_name:<20} "
                f"{result.team:<18} "
                f"{time_str:<15} "
                f"{result.pit_stops:<5} "
                f"{laps:<5} "
                f"{status_str:<25} {points_for_result(result)}"
            )

        print("=" * 70)

    @staticmethod
    def print_monte_carlo_summary(results: SimulationResults, top_n: int = 10) -> None:
        """Print Monte Carlo simulation summary.

        Args:
            results: Aggregated simulation results
            top_n: Top-N threshold for finish probability table
        """
        print("\n" + "=" * 80)
        print(f"MONTE CARLO SIMULATION RESULTS - {results.track_name}")
        print(f"({results.num_simulations} simulations)")
        if results.seed is not None:
            print(
                f"seed={results.seed} "
                f"parallel={results.parallel} "
                f"max_workers={results.max_workers if results.max_workers is not None else 'auto'}"
            )
        print("=" * 80)

        # Win probabilities
        print("\nWIN PROBABILITIES:")
        print("-" * 50)
        win_probs = results.get_win_probabilities()
        for i, (driver_id, prob) in enumerate(win_probs.items()):
            if prob > 0 or i < 10:
                stats = results.driver_stats[driver_id]
                bar = "#" * int(prob / 2)
                print(f"{stats.driver_name:<20} {prob:5.1f}% {bar}")
            if i >= 9 and prob == 0:
                break

        # Podium probabilities
        print("\nPODIUM PROBABILITIES:")
        print("-" * 50)
        podium_sorted = sorted(
            results.driver_stats.items(),
            key=lambda x: x[1].podium_rate,
            reverse=True,
        )
        for i, (driver_id, stats) in enumerate(podium_sorted[:10]):
            bar = "#" * int(stats.podium_rate / 2)
            print(f"{stats.driver_name:<20} {stats.podium_rate:5.1f}% {bar}")

        # Average positions
        print("\nAVERAGE FINISHING POSITION:")
        print("-" * 50)
        avg_sorted = sorted(
            results.driver_stats.items(),
            key=lambda x: x[1].avg_position,
        )
        for driver_id, stats in avg_sorted:
            if stats.positions:
                print(
                    f"{stats.driver_name:<20} "
                    f"Avg: {stats.avg_position:5.2f}  "
                    f"Best: {stats.best_position:2d}  "
                    f"Worst: {stats.worst_position:2d}  "
                    f"DNF: {stats.dnf_rate:4.1f}%"
                )

        # Driver points projection
        print("\nDRIVER POINTS PROJECTION (per race):")
        print("Means use observed driver races; unobserved drivers are omitted.")
        print("-" * 50)
        points_proj = results.get_championship_projection()
        for driver_id, points in list(points_proj.items())[:10]:
            stats = results.driver_stats[driver_id]
            bar = "#" * int(points)
            print(f"{stats.driver_name:<20} {points:5.2f} pts {bar}")

        # Team points projection
        print("\nTEAM POINTS PROJECTION (per race):")
        print("Sum of listed drivers' observed means; incompletely observed teams are omitted.")
        print("-" * 50)
        team_proj = results.get_team_championship_projection()
        for team, points in list(team_proj.items())[:10]:
            bar = "#" * int(points)
            print(f"{team:<20} {points:6.2f} pts {bar}")

        # Top-N finish probabilities
        print(f"\nTOP-{top_n} FINISH PROBABILITIES:")
        print("-" * 50)
        top_finish = results.get_top_n_finish_probabilities(top_n)
        for driver_id, prob in list(top_finish.items())[:10]:
            stats = results.driver_stats[driver_id]
            bar = "#" * int(prob / 2)
            print(f"{stats.driver_name:<20} {prob:5.1f}% {bar}")

        # Event statistics
        event_stats = results.event_stats
        print("\nRACE EVENT STATISTICS:")
        print("-" * 50)
        sims = results.get_event_rate_trials()
        print(f"  Event-rate denominator: {sims} trials")
        if sims > 0:
            sc_rate = event_stats.races_with_safety_car / sims * 100
            rf_rate = event_stats.races_with_red_flag / sims * 100
            avg_sc = event_stats.safety_car_count / sims
            avg_vsc = event_stats.vsc_count / sims
            avg_rf = event_stats.red_flag_count / sims
            avg_incidents = event_stats.total_incidents / sims
        else:
            sc_rate = rf_rate = avg_sc = avg_vsc = avg_rf = avg_incidents = 0

        print(
            f"  Safety Cars:     {event_stats.safety_car_count:4d} total "
            f"({avg_sc:.2f}/race, {sc_rate:.1f}% of races)"
        )
        print(f"  Virtual SC:      {event_stats.vsc_count:4d} total ({avg_vsc:.2f}/race)")
        print(
            f"  Red Flags:       {event_stats.red_flag_count:4d} total "
            f"({avg_rf:.2f}/race, {rf_rate:.1f}% of races)"
        )
        print(
            f"  Total Incidents: {event_stats.total_incidents:4d} total "
            f"({avg_incidents:.2f}/race)"
        )

        print("\n  Mechanical Failure Breakdown:")
        if event_stats.mechanical_failure_breakdown:
            component_rates = results.get_mechanical_failure_component_rates()
            for component, count in sorted(
                event_stats.mechanical_failure_breakdown.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                observed_pct = component_rates.get(component)
                share_text = (
                    f"{observed_pct * 100:5.1f}% observed share"
                    if observed_pct is not None else
                    "share unknown"
                )
                print(
                    f"    - {component:<10} {count:4d} "
                    f"({share_text})"
            )
            if not component_rates:
                print("    Component shares: unknown (no mechanical failures observed)")
        else:
            print("    No mechanical failures observed; component shares are unknown")
        print(
            "    No reference component shares are configured; observed shares are "
            "descriptive only."
        )

        ConsoleOutput._print_suspension_context(results)

        print("=" * 80)

    @staticmethod
    def print_driver_deep_dive(results: SimulationResults, driver_id: str) -> None:
        """Print detailed analysis for a specific driver.

        Args:
            results: Simulation results
            driver_id: Driver to analyze
        """
        if driver_id not in results.driver_stats:
            print(f"Driver {driver_id} not found in results")
            return

        stats = results.driver_stats[driver_id]

        print("\n" + "=" * 60)
        print(f"DETAILED ANALYSIS: {stats.driver_name} ({stats.team})")
        print("=" * 60)

        print(f"\nOverall Statistics ({results.num_simulations} races):")
        print(f"  Wins:           {stats.wins:4d} ({stats.win_rate:.1f}%)")
        print(f"  Podiums:        {stats.podiums:4d} ({stats.podium_rate:.1f}%)")
        print(f"  Points finishes:{stats.points_finishes:4d}")
        print(f"  DNFs:           {stats.dnfs:4d} ({stats.dnf_rate:.1f}%)")
        print(f"  Total points:   {stats.total_points:.0f}")

        print("\nPosition Statistics:")
        print(f"  Average:  {stats.avg_position:.2f}")
        print(f"  Best:     {stats.best_position}")
        print(f"  Worst:    {stats.worst_position}")
        print(f"  Avg Quali:{stats.avg_qualifying:.2f}")

        print("\nPosition Distribution:")
        dist = results.get_position_distribution(driver_id)
        for pos in sorted(dist):
            bar = "#" * int(dist[pos] / 2)
            print(f"  P{pos:2d}: {dist[pos]:5.1f}% {bar}")

        pct = results.get_position_percentiles(driver_id)
        if pct:
            print("\nPosition Percentiles:")
            print(
                f"  P10: {pct.get(10, 0):.2f}  "
                f"P50: {pct.get(50, 0):.2f}  "
                f"P90: {pct.get(90, 0):.2f}"
            )

        top_5 = results.get_top_n_finish_probabilities(5).get(driver_id, 0.0)
        top_10 = results.get_top_n_finish_probabilities(10).get(driver_id, 0.0)
        print(f"\nTop-5 finish probability:  {top_5:.1f}%")
        print(f"Top-10 finish probability: {top_10:.1f}%")

        print("=" * 60)

    @staticmethod
    def print_event_calibration(
        results: SimulationResults,
        expected_sc_race_rate: float,
    ) -> None:
        """Print event calibration diagnostics for one run."""
        rates = results.get_event_rates()
        observed = rates["safety_car_race_rate"]
        delta = results.get_safety_car_calibration_delta(expected_sc_race_rate)

        print("\nEVENT CALIBRATION")
        print(f"Event-rate denominator: {results.get_event_rate_trials()} trials")
        print("-" * 50)
        print(f"Expected SC race rate: {expected_sc_race_rate * 100:5.1f}%")
        print(f"Observed SC race rate: {observed * 100:5.1f}%")
        print(f"Calibration delta:     {delta * 100:5.1f}%")
        print(f"Observed red-flag rate:{rates['red_flag_race_rate'] * 100:5.1f}%")
        print(f"Avg SC/VSC per race:   {rates['avg_safety_cars']:.2f} / {rates['avg_vsc']:.2f}")

    @staticmethod
    def print_scenario_comparison(
        scenario_results: dict[str, SimulationResults],
        top_n: int = 10,
    ) -> None:
        """Show every scenario's entrants, ordered by top-N finish probabilities."""
        if not scenario_results:
            print("No scenario results to compare")
            return

        print("\n" + "=" * 80)
        print("SCENARIO COMPARISON (WIN PROBABILITIES)")
        print("All scenario entrants; -- means no recorded results for that driver.")
        print("=" * 80)

        scenario_names = list(scenario_results.keys())
        header = "Driver".ljust(18) + " ".join(name.rjust(14) for name in scenario_names)
        print(header)
        print("-" * 80)

        top_finish_probabilities = {
            name: result.get_top_n_finish_probabilities(top_n)
            for name, result in scenario_results.items()
        }
        win_probabilities = {
            name: result.get_win_probabilities()
            for name, result in scenario_results.items()
        }
        drivers = []
        seen_drivers = set()
        for probabilities in top_finish_probabilities.values():
            for driver_id in probabilities:
                if driver_id not in seen_drivers:
                    seen_drivers.add(driver_id)
                    drivers.append(driver_id)

        for driver_id in drivers:
            row = driver_id.ljust(18)
            for scenario in scenario_names:
                stats = scenario_results[scenario].driver_stats.get(driver_id)
                if stats is None or not stats.positions:
                    row += f"{'--':>14} "
                else:
                    row += f"{win_probabilities[scenario].get(driver_id, 0.0):13.1f}% "
            print(row)

        ConsoleOutput._print_suspension_context(scenario_results)

        print("=" * 80)

    @staticmethod
    def _print_suspension_context(
        results: SimulationResults | dict[str, SimulationResults],
    ) -> None:
        """Print race-wide suspension observations with their own denominator."""
        print("\nRACE SUSPENSION CONTEXT:")
        print("Race-wide collection + restart pause are already in finish clocks; "
              "they are not an individual driver's stopped or driving time.")
        items = results.items() if isinstance(results, dict) else [(results.track_name, results)]
        for name, result in items:
            stats = suspension_statistics(result)
            recorded = stats["recorded_races"]
            known = stats["races_with_recorded_suspension"]
            unit = "race" if recorded == 1 else "races"
            print(
                f"  {name}: Mean completed race suspension: "
                f"{format_seconds(stats['mean_completed_suspension_seconds'])}; "
                f"{recorded} recorded {unit}; {known} with suspension"
            )
