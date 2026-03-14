from typing import cast

import pytest

from f1sim.analysis.montecarlo import POINTS_SYSTEM, DriverStatistics, MonteCarloRunner


def test_points_system_top10_values() -> None:
    assert POINTS_SYSTEM[1] == 25
    assert POINTS_SYSTEM[2] == 18
    assert POINTS_SYSTEM[3] == 15
    assert POINTS_SYSTEM[10] == 1


def test_driver_statistics_rates() -> None:
    stats = DriverStatistics(
        driver_id="VER",
        driver_name="Max Verstappen",
        team="Red Bull",
        wins=2,
        podiums=3,
        dnfs=1,
        positions=[1, 1, 2, 12],
    )

    assert stats.win_rate == 50.0
    assert stats.podium_rate == 75.0
    assert stats.dnf_rate == 25.0


def test_driver_statistics_rates_empty_positions() -> None:
    stats = DriverStatistics(
        driver_id="VER",
        driver_name="Max Verstappen",
        team="Red Bull",
    )

    assert stats.win_rate == 0
    assert stats.podium_rate == 0
    assert stats.dnf_rate == 0


def test_montecarlo_rejects_non_positive_simulation_count() -> None:
    with pytest.raises(ValueError, match="num_simulations must be greater than 0"):
        MonteCarloRunner.run(cast(MonteCarloRunner, object()), num_simulations=0)
