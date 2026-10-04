"""Dashboard weighted rival-plan selection stays preflighted and replayable."""

from types import SimpleNamespace

import pytest

from f1sim.models import Car, Driver, Track
from f1sim.web import server

_HARD_STOP = [{"lap": 2, "compound": "hard"}]


class RivalLoader:
    def resolve_race_identifier(self, *args):
        return 1

    def list_available_events(self, *args):
        return [{"round": 1, "race": "Synthetic"}]

    def get_weighted_driver_stats(self, **kwargs):
        return {}

    def get_track_stats(self, *args):
        return SimpleNamespace(
            track_name="Synthetic", country="Test", total_laps=3, avg_lap_time=90,
        )

    def create_drivers_from_stats(self, *args):
        return [Driver(id="A", name="A", team_id="T"), Driver(id="B", name="B", team_id="U")]

    def create_cars_from_stats(self, *args):
        return {
            "T": Car(team_id="T", team_name="T"),
            "U": Car(team_id="U", team_name="U"),
        }

    def create_track_from_stats(self, *args):
        return Track(
            id="synthetic", name="Synthetic", country="Test", total_laps=3,
            base_lap_time=90,
        )

    def get_provenance(self):
        return {"source": "test"}


def _selection(**overrides):
    value = {
        "plans": {"reference": None, "planned": _HARD_STOP},
        "reference_label": "reference",
        "driver_id": "A",
        "training_simulations": 1,
        "validation_simulations": 1,
        "rival_scenarios": {
            "inherit": {"weight": 2, "pit_plans": {}},
            "automatic": {"weight": 1, "pit_plans": {"B": None}},
            "no_stops": {"weight": 1, "pit_plans": {"B": []}},
        },
    }
    value.update(overrides)
    return value


def _request(**overrides):
    values = {
        "year": 2026,
        "simulations": 10,
        "scenarios": "dry",
        "parallel": False,
        "starting_tires": {"A": "medium", "B": "medium"},
        "tire_inventory": {
            driver_id: [
                {"id": f"{driver_id}-medium", "compound": "medium", "age": 0},
                {"id": f"{driver_id}-hard", "compound": "hard", "age": 0},
            ]
            for driver_id in ("A", "B")
        },
        "pit_plans": {"A": _HARD_STOP, "B": _HARD_STOP},
        "pit_plan_selection": _selection(),
    }
    values.update(overrides)
    return server.DashboardRunRequest(**values)


def _install_loader(monkeypatch):
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "_get_loader", RivalLoader)


@pytest.mark.parametrize("objective", ["win", "podium"])
def test_weighted_dashboard_probabilities_are_frozen_and_exported(monkeypatch, objective):
    _install_loader(monkeypatch)
    result = server.run_dashboard_simulation(_request(pit_plan_selection=_selection(
        objective=objective,
    )))
    selection = result["strategy_selections"]["dry"]["selection"]
    assert selection["objective"] == objective
    assert selection["score_unit"] == "probability"
    assert all("mean_score" in row for row in selection["training_score_table"])
    assert all("score_difference_standard_error" in metrics
               for metrics in selection["validation_scenario_metrics"].values())
    assert result["request"]["pit_plan_selection"]["objective"] == objective
    assert "Held-out objective probabilities" in (
        result["strategy_selections"]["dry"]["validation_report_html"]
    )


def test_weighted_dashboard_response_keeps_source_and_exposes_frozen_plan_maps(monkeypatch):
    _install_loader(monkeypatch)

    result = server.run_dashboard_simulation(_request())

    entry = result["strategy_selections"]["dry"]
    assert set(entry) == {
        "selection", "plans", "plans_by_rival_scenario", "source",
        "training_by_rival_scenario", "validation_by_rival_scenario",
        "validation_report_html",
    }
    assert entry["plans"] == {"reference": None, "planned": _HARD_STOP}
    assert entry["plans_by_rival_scenario"]["inherit"]["reference"] == {"B": _HARD_STOP}
    assert entry["plans_by_rival_scenario"]["automatic"]["reference"] == {}
    assert entry["plans_by_rival_scenario"]["no_stops"]["reference"] == {"B": []}
    assert entry["plans_by_rival_scenario"]["inherit"]["planned"] == {
        "A": _HARD_STOP,
        "B": _HARD_STOP,
    }
    assert set(entry["training_by_rival_scenario"]) == {"inherit", "automatic", "no_stops"}
    assert set(entry["validation_by_rival_scenario"]) == {"inherit", "automatic", "no_stops"}
    assert all(
        set(summary) == {"scenarios"}
        for summary in entry["training_by_rival_scenario"].values()
    )
    assert "Rival strategy selection report" in entry["validation_report_html"]
    assert result["scenarios"]["dry"]["simulation_inputs"]["pit_plans"] == {
        "A": _HARD_STOP,
        "B": _HARD_STOP,
    }
    assert "training" not in entry and "validation" not in entry
    assert "rival_scenarios" in result["request"]["pit_plan_selection"]


def test_weighted_budget_counts_every_assumption_before_live_loading(monkeypatch):
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live load"))
    scenarios = {
        f"rival_{index}": {"weight": 1, "pit_plans": {}}
        for index in range(2)
    }

    with pytest.raises(ValueError, match="worst-case work.*at most 1000"):
        server.run_dashboard_simulation(_request(
            pit_plan_selection=_selection(
                plans={"reference": None, "planned": None},
                rival_scenarios=scenarios,
                training_simulations=245,
                validation_simulations=4,
            ),
        ))


@pytest.mark.parametrize("override,message", [
    ({"weight": 0}, "positive finite"),
    ({"weather": {"rain_intensity": True}}, "Invalid race weather"),
    ({"weather": {"unknown": 1}}, "unknown"),
    ({"weather_schedule": [{"lap": 1, "rain_intensity": .5}]}, "lap"),
])
def test_http_rejects_invalid_assumptions_before_capacity_admission(
    monkeypatch, override, message,
):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class NoCapacity:
        @classmethod
        def from_environment(cls):
            return cls()

        def acquire(self):
            pytest.fail("invalid rival request must fail before capacity admission")

    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "RunCapacity", NoCapacity)
    selection = _selection(rival_scenarios={
        "invalid": {"weight": 1, "pit_plans": {}, **override},
    })
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={
            "year": 2026,
            "simulations": 10,
            "scenarios": "dry",
            "pit_plan_selection": selection,
        })

    assert response.status_code == 422
    assert message in response.text


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_dashboard_weather_cases_freeze_qualifying_and_export_actual_inputs(monkeypatch, engine):
    _install_loader(monkeypatch)
    assumptions = {
        "inherited": {"weight": 1, "pit_plans": {}},
        "cleared dry": {"weight": 2, "pit_plans": {}, "weather_schedule": [],
                        "weather": {"condition": "dry", "track_wetness": 0,
                                    "rain_intensity": 0}},
        "timed rain": {"weight": 1, "pit_plans": {"B": []},
                       "weather": {"condition": "heavy_rain", "track_wetness": .8,
                                   "rain_intensity": .8},
                       "weather_schedule": [{"lap": 2, "rain_intensity": 0,
                                             "condition": "dry"}]},
    }
    result = server.run_dashboard_simulation(_request(
        race_engine=engine, scenarios="dry,light_rain", weather_mode="fixed_rainfall",
        weather_schedule=[{"lap": 3, "rain_intensity": .5}],
        qualifying_weather={"Q1": {"condition": "cloudy"}},
        pit_plan_selection=_selection(rival_scenarios=assumptions),
    ))
    assert result["request"]["pit_plan_selection"]["rival_scenarios"] == assumptions
    for entry in result["strategy_selections"].values():
        selection = entry["selection"]
        assert "weather_and_rival" in selection["method"]
        assert selection["frozen_qualifying_weather"]["Q1"]["condition"] == "cloudy"
        assert entry["source"]["simulation_inputs"]["weather_schedule"] == [
            {"lap": 3, "rain_intensity": .5},
        ]
        for case in selection["rival_scenarios"]:
            for phase in ("training", "validation"):
                variants = entry[f"{phase}_by_rival_scenario"][case["name"]]["scenarios"]
                for summary in variants.values():
                    inputs = summary["simulation_inputs"]
                    assert inputs["weather"] == case["weather"]
                    assert inputs.get("weather_schedule", []) == case["weather_schedule"]
                    assert inputs["qualifying_weather"] == selection["frozen_qualifying_weather"]
        assert "Race weather and schedule" in entry["validation_report_html"]
        assert "Shared qualifying weather (frozen)" in entry["validation_report_html"]


def test_weather_distance_preflight_happens_before_source_trials(monkeypatch):
    _install_loader(monkeypatch)
    monkeypatch.setattr(server.MonteCarloRunner, "run", lambda *a, **kw: pytest.fail("trial work"))
    with pytest.raises(ValueError, match="through the scheduled distance"):
        server.run_dashboard_simulation(_request(pit_plan_selection=_selection(
            rival_scenarios={"too late": {"weight": 1, "pit_plans": {},
                                          "weather_schedule": [{"lap": 4,
                                                                "rain_intensity": .8}]}},
        )))


@pytest.mark.parametrize(
    ("rivals", "message"),
    [
        ({}, "1 to 10 named scenarios"),
        ({"bad": {"weight": 0, "pit_plans": {}}}, "positive finite"),
        ({"bad": {"weight": 1, "pit_plans": {}, "extra": True}}, "extra"),
        ({"bad": {"weight": 1, "pit_plans": {"B": "automatic"}}}, "list or null"),
        ({" ": {"weight": 1, "pit_plans": {}}}, "at most 80"),
    ],
)
def test_invalid_rival_request_shape_fails_before_live_loading(monkeypatch, rivals, message):
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live load"))

    with pytest.raises(ValueError, match=message):
        server.run_dashboard_simulation(_request(
            pit_plan_selection=_selection(rival_scenarios=rivals),
        ))


def test_late_weather_seed_overflow_preflights_before_source_trials(monkeypatch):
    _install_loader(monkeypatch)
    calls = []
    monkeypatch.setattr(
        server.MonteCarloRunner,
        "run",
        lambda self, *args, **kwargs: calls.append(self.base_seed),
    )

    with pytest.raises(ValueError, match="must not exceed"):
        server.run_dashboard_simulation(_request(
            seed=(2**32 - 1) - 1010,
            scenarios="dry,light_rain",
            pit_plan_selection=_selection(
                rival_scenarios={
                    "one": {"weight": 1, "pit_plans": {}},
                    "two": {"weight": 1, "pit_plans": {"B": []}},
                },
            ),
        ))
    assert calls == []


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"A": None}, "cannot override target"),
        ({"UNKNOWN": None}, "Unknown or nonrunnable"),
        ({"B": [{"lap": 2, "compound": "soft"}]}, "absent from"),
    ],
)
def test_loaded_target_roster_and_inventory_preflight_before_source_run(
    monkeypatch, override, message,
):
    _install_loader(monkeypatch)
    calls = []
    monkeypatch.setattr(
        server.MonteCarloRunner,
        "run",
        lambda self, *args, **kwargs: calls.append(self.base_seed),
    )

    with pytest.raises(ValueError, match=message):
        server.run_dashboard_simulation(_request(
            pit_plan_selection=_selection(
                rival_scenarios={"invalid": {"weight": 1, "pit_plans": override}},
            ),
        ))
    assert calls == []
