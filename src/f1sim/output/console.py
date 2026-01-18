"""Console output formatting."""

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.simulation.qualifying import QualifyingResult
from f1sim.simulation.race import RaceResult


class ConsoleOutput:
    """Formats simulation results for console display."""

    @staticmethod
    def print_qualifying_results(results: list[QualifyingResult]) -> None:
        """Print qualifying results to console.

        Args:
            results: Qualifying results sorted by position
        """
        print("\n" + "=" * 60)
        print("QUALIFYING RESULTS")
        print("=" * 60)
        print(f"{'Pos':<4} {'Driver':<20} {'Team':<15} {'Time':<12} {'Gap':<10}")
        print("-" * 60)

        pole_time = results[0].best_time if results else 0

        for result in sorted(results, key=lambda r: r.position):
            gap = ""
            if result.position > 1:
                gap_secs = result.best_time - pole_time
                gap = f"+{gap_secs:.3f}"

            eliminated = f" (out in {result.eliminated_in})" if result.eliminated_in else ""

            print(
                f"{result.position:<4} "
                f"{result.driver_name:<20} "
                f"{'':<15} "
                f"{result.best_time:.3f}s "
                f"{gap:<10}"
                f"{eliminated}"
            )

        print("=" * 60)

    @staticmethod
    def print_race_results(results: list[RaceResult]) -> None:
        """Print race results to console.

        Args:
            results: Race results sorted by position
        """
        print("\n" + "=" * 70)
        print("RACE RESULTS")
        print("=" * 70)
        print(f"{'Pos':<4} {'Driver':<20} {'Team':<18} {'Time/Gap':<15} {'Pits':<5} {'Status':<10}")
        print("-" * 70)

        leader_time = None
        for result in sorted(results, key=lambda r: r.position):
            if result.status.value == "finished":
                if leader_time is None:
                    leader_time = result.total_time
                    time_str = f"{result.total_time:.3f}s"
                else:
                    gap = result.total_time - leader_time
                    if gap < 60:
                        time_str = f"+{gap:.3f}s"
                    else:
                        mins = int(gap // 60)
                        secs = gap % 60
                        time_str = f"+{mins}:{secs:05.2f}"
            else:
                time_str = result.dnf_reason or "DNF"

            status_str = result.status.value.upper()
            if result.status.value == "finished":
                status_str = ""

            print(
                f"{result.position:<4} "
                f"{result.driver_name:<20} "
                f"{result.team:<18} "
                f"{time_str:<15} "
                f"{result.pit_stops:<5} "
                f"{status_str:<10}"
            )

        print("=" * 70)

    @staticmethod
    def print_monte_carlo_summary(results: SimulationResults) -> None:
        """Print Monte Carlo simulation summary.

        Args:
            results: Aggregated simulation results
        """
        print("\n" + "=" * 80)
        print(f"MONTE CARLO SIMULATION RESULTS - {results.track_name}")
        print(f"({results.num_simulations} simulations)")
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

        # Points projection
        print("\nPOINTS PROJECTION (per race):")
        print("-" * 50)
        points_proj = results.get_championship_projection()
        for driver_id, points in list(points_proj.items())[:10]:
            stats = results.driver_stats[driver_id]
            bar = "#" * int(points)
            print(f"{stats.driver_name:<20} {points:5.2f} pts {bar}")

        # Event statistics
        event_stats = results.event_stats
        print("\nRACE EVENT STATISTICS:")
        print("-" * 50)
        sc_rate = (event_stats.races_with_safety_car / results.num_simulations * 100) if results.num_simulations > 0 else 0
        rf_rate = (event_stats.races_with_red_flag / results.num_simulations * 100) if results.num_simulations > 0 else 0
        avg_sc = event_stats.safety_car_count / results.num_simulations if results.num_simulations > 0 else 0
        avg_vsc = event_stats.vsc_count / results.num_simulations if results.num_simulations > 0 else 0
        avg_rf = event_stats.red_flag_count / results.num_simulations if results.num_simulations > 0 else 0
        avg_incidents = event_stats.total_incidents / results.num_simulations if results.num_simulations > 0 else 0

        print(f"  Safety Cars:     {event_stats.safety_car_count:4d} total ({avg_sc:.2f}/race, {sc_rate:.1f}% of races)")
        print(f"  Virtual SC:      {event_stats.vsc_count:4d} total ({avg_vsc:.2f}/race)")
        print(f"  Red Flags:       {event_stats.red_flag_count:4d} total ({avg_rf:.2f}/race, {rf_rate:.1f}% of races)")
        print(f"  Total Incidents: {event_stats.total_incidents:4d} total ({avg_incidents:.2f}/race)")

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

        print(f"\nPosition Statistics:")
        print(f"  Average:  {stats.avg_position:.2f}")
        print(f"  Best:     {stats.best_position}")
        print(f"  Worst:    {stats.worst_position}")
        print(f"  Avg Quali:{stats.avg_qualifying:.2f}")

        print(f"\nPosition Distribution:")
        dist = results.get_position_distribution(driver_id)
        for pos in range(1, 21):
            if pos in dist:
                bar = "#" * int(dist[pos] / 2)
                print(f"  P{pos:2d}: {dist[pos]:5.1f}% {bar}")

        print("=" * 60)
