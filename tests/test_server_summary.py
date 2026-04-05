from f1sim.web.server import _summarize_scenario_results, build_dashboard_html


class _FakeResults:
    def __init__(self, seed: int) -> None:
        self.num_simulations = 100
        self.seed = seed

    def get_win_probabilities(self):
        return {"VER": 55.0, "NOR": 25.0, "HAM": 20.0}

    def get_event_rates(self):
        return {"safety_car_race_rate": 0.4}

    def get_team_championship_projection(self):
        return {"Red Bull": 35.0, "McLaren": 28.0}


def test_dashboard_summary_shape() -> None:
    summary = _summarize_scenario_results(
        {
            "dry": _FakeResults(seed=42),
            "heavy_rain": _FakeResults(seed=1042),
        }
    )

    assert set(summary["scenarios"].keys()) == {"dry", "heavy_rain"}
    assert summary["scenarios"]["dry"]["seed"] == 42
    assert summary["scenarios"]["dry"]["top3_win_probabilities"][0][0] == "VER"
    assert "win_probabilities" in summary["scenarios"]["dry"]
    assert "runtime_seconds" in summary["scenarios"]["dry"]
    assert "simulations_per_second" in summary["scenarios"]["dry"]
    assert "driver_statistics" in summary["scenarios"]["dry"]
    assert "sample_race" in summary["scenarios"]["dry"]
    assert "sample_qualifying" in summary["scenarios"]["dry"]
    assert summary["scenarios"]["dry"]["qualifying_mode"] == "simulated"


def test_dashboard_html_contains_controls() -> None:
    html = build_dashboard_html()
    assert "Run Simulation" in html
    assert "scenarioSetInput" in html
    assert "panel-scenarios" in html
    assert "renderScenarioLab" in html
    assert "renderScenarioCards" in html
    assert "renderScenarioWinChart" in html
    assert "renderScenarioTrends" in html
    assert "renderDriverMatrix" in html
    assert "compareTopN" in html
    assert "compareTrendMetric" in html
    assert "compareTrendScale" in html
    assert "compareMatrixSort" in html
    assert "compareMatrixHighlight" in html
    assert "compareDriverFilter" in html
    assert "compareScenarioFilter" in html
    assert "downloadScenarioJsonBtn" in html
    assert "downloadResultJson" in html
    assert "exportScenarioMatrixCsv" in html
    assert "presetDryBtn" in html
    assert "presetMixedBtn" in html
    assert "presetChaosBtn" in html
    assert "LAST_PAYLOAD_KEY" in html
    assert "f1sim:lastPayload" in html
    assert "f1sim:uiPrefs" in html
    assert "scenarios" in html
