from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults


def _stats(
    driver_id: str,
    positions: list[int],
    wins: int = 0,
    total_points: float = 0.0,
) -> DriverStatistics:
    return DriverStatistics(
        driver_id=driver_id,
        driver_name=driver_id,
        team="Test",
        wins=wins,
        total_points=total_points,
        positions=positions,
    )


def test_position_distribution_percentages() -> None:
    stats = {"VER": _stats("VER", [1, 1, 2, 3])}
    results = SimulationResults(
        num_simulations=4,
        track_name="Bahrain",
        driver_stats=stats,
        race_results=[],
        qualifying_results=[],
    )

    dist = results.get_position_distribution("VER")
    assert dist == {1: 50.0, 2: 25.0, 3: 25.0}


def test_win_probabilities_sorted_desc() -> None:
    stats = {
        "VER": _stats("VER", [1, 2], wins=1),
        "PER": _stats("PER", [2, 1], wins=1),
        "HAM": _stats("HAM", [3, 3], wins=0),
    }
    results = SimulationResults(
        num_simulations=2,
        track_name="Bahrain",
        driver_stats=stats,
        race_results=[],
        qualifying_results=[],
    )

    probs = results.get_win_probabilities()
    assert probs["VER"] == 50.0
    assert probs["PER"] == 50.0
    assert probs["HAM"] == 0.0
