import pytest

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


def test_position_distribution_unknown_driver() -> None:
    results = SimulationResults(
        num_simulations=1,
        track_name="Bahrain",
        driver_stats={"VER": _stats("VER", [1])},
        race_results=[],
        qualifying_results=[],
    )

    assert results.get_position_distribution("HAM") == {}


def test_championship_projection_sorted_desc() -> None:
    stats = {
        "VER": DriverStatistics(
            driver_id="VER",
            driver_name="VER",
            team="Red Bull",
            positions=[1],
            total_points=250.0,
        ),
        "NOR": DriverStatistics(
            driver_id="NOR",
            driver_name="NOR",
            team="McLaren",
            positions=[2],
            total_points=180.0,
        ),
        "HAM": DriverStatistics(
            driver_id="HAM",
            driver_name="HAM",
            team="Ferrari",
            positions=[3],
            total_points=100.0,
        ),
    }
    results = SimulationResults(
        num_simulations=10,
        track_name="Bahrain",
        driver_stats=stats,
        race_results=[],
        qualifying_results=[],
    )

    projection = results.get_championship_projection()
    assert list(projection.keys()) == ["VER", "NOR", "HAM"]
    assert projection["VER"] == 25.0
    assert projection["NOR"] == 18.0
    assert projection["HAM"] == 10.0


def test_team_championship_projection_aggregates_teammates() -> None:
    stats = {
        "VER": DriverStatistics(
            driver_id="VER",
            driver_name="VER",
            team="Red Bull",
            positions=[1],
            total_points=250.0,
        ),
        "PER": DriverStatistics(
            driver_id="PER",
            driver_name="PER",
            team="Red Bull",
            positions=[2],
            total_points=180.0,
        ),
        "NOR": DriverStatistics(
            driver_id="NOR",
            driver_name="NOR",
            team="McLaren",
            positions=[3],
            total_points=220.0,
        ),
    }
    results = SimulationResults(
        num_simulations=10,
        track_name="Bahrain",
        driver_stats=stats,
        race_results=[],
        qualifying_results=[],
    )

    projection = results.get_team_championship_projection()
    assert list(projection.keys()) == ["Red Bull", "McLaren"]
    assert projection["Red Bull"] == 43.0
    assert projection["McLaren"] == 22.0


def test_top_n_finish_probabilities() -> None:
    stats = {
        "VER": DriverStatistics(
            driver_id="VER",
            driver_name="VER",
            team="Red Bull",
            positions=[1, 2, 11, 12],
        ),
        "NOR": DriverStatistics(
            driver_id="NOR",
            driver_name="NOR",
            team="McLaren",
            positions=[3, 4, 5, 6],
        ),
    }
    results = SimulationResults(
        num_simulations=4,
        track_name="Bahrain",
        driver_stats=stats,
        race_results=[],
        qualifying_results=[],
    )

    top_10 = results.get_top_n_finish_probabilities(10)
    assert top_10["NOR"] == 100.0
    assert top_10["VER"] == 50.0


def test_top_n_finish_probabilities_rejects_invalid_n() -> None:
    results = SimulationResults(
        num_simulations=1,
        track_name="Bahrain",
        driver_stats={},
        race_results=[],
        qualifying_results=[],
    )

    with pytest.raises(ValueError, match="n must be greater than 0"):
        results.get_top_n_finish_probabilities(0)


def test_position_percentiles_for_driver() -> None:
    stats = {
        "VER": DriverStatistics(
            driver_id="VER",
            driver_name="VER",
            team="Red Bull",
            positions=[1, 2, 3, 10],
        )
    }
    results = SimulationResults(
        num_simulations=4,
        track_name="Bahrain",
        driver_stats=stats,
        race_results=[],
        qualifying_results=[],
    )

    pct = results.get_position_percentiles("VER")
    assert pct[10] <= pct[50] <= pct[90]
    assert pct[50] == 2.5


def test_position_percentiles_unknown_driver() -> None:
    results = SimulationResults(
        num_simulations=1,
        track_name="Bahrain",
        driver_stats={},
        race_results=[],
        qualifying_results=[],
    )

    assert results.get_position_percentiles("VER") == {}
