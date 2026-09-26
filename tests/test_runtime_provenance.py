"""Saved runtime metadata is compared safely and remains informational."""

import json
import socket

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.provenance import (
    compare_saved_runtime,
    format_saved_runtime_status,
    saved_runtime_status,
    simulation_runtime,
)
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter


@pytest.fixture
def saved(tmp_path):
    result = MonteCarloRunner(
        [Driver(id="A", name="A", team_id="T")],
        {"T": Car(team_id="T", team_name="T")},
        Track(id="t", name="Track", country="Test", total_laps=3, base_lap_time=90),
        Weather(change_probability=0), seed=41,
    ).run(1, parallel=False)
    return Exporter(tmp_path).export_statistics_json(result)


def test_installed_runtime_matches_new_saved_input(saved):
    assert saved_runtime_status(saved) == ("match", ())
    assert format_saved_runtime_status(saved_runtime_status(saved)) == (
        "Runtime provenance (installed vs saved): match"
    )


@pytest.mark.parametrize(
    "field",
    ["f1sim", "python", "numpy", "pydantic", "simulation_source_sha256"],
)
def test_reports_only_the_name_of_each_mismatched_runtime_field(saved, field):
    data = json.loads(saved.read_text(encoding="utf-8"))
    runtime = data["simulation_inputs"]["runtime"]
    runtime[field] = "0" * 64 if field == "simulation_source_sha256" else "private-old-value"
    saved.write_text(json.dumps(data), encoding="utf-8")

    status = saved_runtime_status(saved)

    assert status == ("mismatch", (field,))
    summary = format_saved_runtime_status(status)
    assert field in summary
    assert "private-old-value" not in summary
    assert "0" * 64 not in summary


@pytest.mark.parametrize(
    "mutate",
    [
        lambda runtime: runtime.pop("numpy"),
        lambda runtime: runtime.update(python=1),
        lambda runtime: runtime.update(python=""),
        lambda runtime: runtime.update(python="   "),
        lambda runtime: runtime.update(python="3 .12"),
        lambda runtime: runtime.update(python="v" * 129),
        lambda runtime: runtime.update(simulation_source_sha256="broken"),
        lambda runtime: runtime.update(simulation_source_sha256="g" * 64),
    ],
)
def test_missing_or_malformed_runtime_is_unavailable(saved, mutate):
    data = json.loads(saved.read_text(encoding="utf-8"))
    mutate(data["simulation_inputs"]["runtime"])
    saved.write_text(json.dumps(data), encoding="utf-8")

    assert saved_runtime_status(saved) == ("unavailable", ())
    assert format_saved_runtime_status(("unavailable", ())) == (
        "Runtime provenance (installed vs saved): unavailable"
    )


@pytest.mark.parametrize("runtime", [None, [], {"python": "3.12"}])
def test_runtime_comparison_requires_all_five_fields(runtime):
    assert compare_saved_runtime(runtime) == ("unavailable", ())


def test_status_formatter_only_includes_known_field_names():
    assert format_saved_runtime_status(("mismatch", ("secret-value", "python"))) == (
        "Runtime provenance (installed vs saved): mismatch (python)"
    )


def test_uppercase_sha256_is_a_valid_matching_digest():
    runtime = simulation_runtime()
    runtime["simulation_source_sha256"] = runtime["simulation_source_sha256"].upper()
    assert compare_saved_runtime(runtime) == ("match", ())


def test_named_scenario_status_uses_replay_selection_rules(saved):
    source = json.loads(saved.read_text(encoding="utf-8"))
    scenarios = {
        "dry": {**source["metadata"], "simulation_inputs": source["simulation_inputs"]},
        "wet": {**source["metadata"], "simulation_inputs": source["simulation_inputs"]},
    }
    scenarios["wet"]["simulation_inputs"] = {
        **source["simulation_inputs"], "runtime": {"malformed": True},
    }
    saved.write_text(json.dumps({"scenarios": scenarios}), encoding="utf-8")

    assert saved_runtime_status(saved, "dry") == ("match", ())
    assert saved_runtime_status(saved, "wet") == ("unavailable", ())
    assert saved_runtime_status(saved) == ("unavailable", ())


def test_status_lookup_never_contacts_network(saved, monkeypatch):
    def unexpected_connect(*args, **kwargs):
        raise AssertionError("Runtime status must remain offline")

    monkeypatch.setattr(socket.socket, "connect", unexpected_connect)
    assert saved_runtime_status(saved) == ("match", ())


def test_malformed_json_status_is_unavailable(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")
    assert saved_runtime_status(path) == ("unavailable", ())
