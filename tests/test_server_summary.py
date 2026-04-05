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


def test_dashboard_html_contains_controls() -> None:
    html = build_dashboard_html()
    assert "Run Simulation" in html
    assert "/api/run" in html
    assert "/api/runs" in html
    assert "/output/" in html
    assert "renderScenarioCards" in html
    assert "renderWinChart" in html
    assert "renderScenarioTrends" in html
    assert "renderDriverMatrix" in html
    assert "resultChart" in html
    assert "resultTrends" in html
    assert "trendMetric" in html
    assert "trendScale" in html
    assert "topNSelector" in html
    assert "resultMatrix" in html
    assert "matrixSort" in html
    assert "matrixHighlight" in html
    assert "matrixDriverFilter" in html
    assert "matrixScenarioFilter" in html
    assert "matrixExportBtn" in html
    assert "exportMatrixCsv" in html
    assert "exampleConfig" in html
    assert "applyExampleBtn" in html
    assert "runExampleBtn" in html
    assert "downloadResultBtn" in html
    assert "downloadResultJson" in html
    assert "loadSelectedExample" in html
    assert "EXAMPLE_CONFIGS" in html
    assert "renderQuickActions" in html
    assert "presetDry" in html
    assert "presetMixed" in html
    assert "presetWet" in html
    assert "rerunBtn" in html
    assert "clearBtn" in html
    assert "runsFilter" in html
    assert "f1sim:lastPayload" in html
    assert "f1sim:uiPrefs" in html
    assert "Open latest report" in html
    assert "scenarios" in html
