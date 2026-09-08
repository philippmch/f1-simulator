"""Weather policy request validation happens before contacting live providers."""

import pytest

from f1sim.web import server


@pytest.mark.parametrize("mode", [None, True, 1, [], {}, "fixed", "EVOLVING"])
def test_invalid_dashboard_weather_mode_precedes_live_fetch(monkeypatch, mode):
    def unexpected_loader(*args, **kwargs):
        pytest.fail("Invalid weather mode reached the live provider")

    monkeypatch.setattr(server, "CurrentSeasonDataLoader", unexpected_loader)
    with pytest.raises(ValueError, match="weather_mode"):
        server.run_dashboard_simulation(server.DashboardRunRequest(weather_mode=mode))
