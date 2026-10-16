"""The portable historical verifier never invents missing joint outcomes."""

import gzip
import importlib.util
import json
import socket
from pathlib import Path

import pytest
from test_learned_forecast import _stock_fixture


@pytest.fixture(scope="module")
def verifier():
    path = Path(__file__).resolve().parents[1] / "examples/verify_historical_learned_afterqual.py"
    spec = importlib.util.spec_from_file_location("portable_historical_learned", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_paths(result):
    return [[{"driver_id": row.driver_id, "position": row.position,
              "status": row.status.value, "classified": row.classified}
             for row in race] for race in result.race_results]


def test_portable_reconstruction_preserves_late_dnf_classification_and_no_winner(verifier):
    result, features, artifact = _stock_fixture()
    native, guarded = verifier.forecast_from_paths(_minimal_paths(result), features, artifact)
    assert native["wins"] == {"A": 1, "B": 2, "C": 0}
    assert native["no_classified_winner_count"] == 1
    assert guarded["winner_probabilities"][verifier.NULL_WINNER] == .25
    assert guarded["marginals"]["C"]["podium_probability"] == 0
    assert guarded["marginals"]["C"]["nonfinish_probability"] == pytest.approx(1)
    assert sum(row["expected_position"] for row in guarded["marginals"].values()) == (
        pytest.approx(6)
    )


def test_aggregate_only_identity_does_not_synthesize_paths_or_weight_arrays(verifier):
    result, features, _ = _stock_fixture()
    aggregate, _ = verifier.aggregate_paths(_minimal_paths(result), features["codes"])
    guarded = verifier.forecast_identity_aggregate(aggregate, {"A": .1, "B": .8, "C": .1})
    assert guarded["joint_rows_available"] is False
    assert guarded["ESS"] == 4.
    assert "normalized_trial_weights" not in guarded
    assert "trial_winner_categories" not in guarded
    assert guarded["uniform_weight_formula_only"].startswith("1/4;")
    assert guarded["winner_probabilities"]["B"] == .5
    assert guarded["marginals"]["A"]["expected_position"] == 2.25
    with pytest.raises(ValueError, match="exact identity"):
        verifier.forecast_identity_aggregate(aggregate, {"A": .8, "B": .1, "C": .1})


def test_portable_comparison_tolerates_float_roundoff_but_keeps_discrete_state_exact(verifier):
    verifier.close_values({"q": .4 + 1e-15, "active": ["A"]}, {"q": .4, "active": ["A"]})
    with pytest.raises(ValueError, match="discrete"):
        verifier.close_values({"q": .4, "active": ["B"]}, {"q": .4, "active": ["A"]})
    with pytest.raises(ValueError, match="numerical"):
        verifier.close_values(.4 + 1e-5, .4)


def test_portable_archive_requires_a_valid_content_seal(verifier):
    with pytest.raises(ValueError, match="content seal"):
        verifier.verify_capsule({"schema_version": 1, "kind": verifier.KIND,
                                 "content_sha256": "0" * 64})


def test_saved_count_bootstrap_replay_separates_identity_projection_and_fallback(verifier):
    aggregate = {"trials": 400, "wins": {"A": 100, "B": 200, "C": 0},
                 "no_classified_winner_count": 100}
    analytic = {"A": .7, "B": .2, "C": .1}
    point_q, diagnostic = verifier.project_winner_probabilities(
        {"A": .25, "B": .5, "C": 0., verifier.NULL_WINNER: .25}, analytic,
    )
    summary = verifier.replay_saved_count_draws(
        [[200, 100, 0, 100], [100, 200, 0, 100], [0, 300, 0, 100]],
        ["A", "B", "C", verifier.NULL_WINNER], aggregate, analytic,
        {"winner_probabilities": point_q, "diagnostic": diagnostic}, ["A", "B", "C"],
    )
    assert summary["identity_count"] == 1
    assert summary["nonidentity_count"] == 2
    assert summary["actual_projection_count"] == summary["projection_count"] == 1
    assert summary["fallback_counts"] == {"native_j_zero": 1}
    assert summary["primary_pick_counts"] == {"A": 2, "B": 1}
    assert summary["sampled_trigger_discontinuity"] is True
    assert summary["maximum_absolute_q_deviation_quantiles"][-1] == pytest.approx(7 / 12)


def test_complete_portable_capsule_replays_without_network_or_simulation(verifier, monkeypatch):
    from f1sim.analysis.montecarlo import MonteCarloRunner
    from f1sim.data.current import CurrentSeasonDataLoader

    def forbidden(*args, **kwargs):
        raise AssertionError("historical verification must remain offline without races")

    monkeypatch.setattr(MonteCarloRunner, "run", forbidden)
    monkeypatch.setattr(CurrentSeasonDataLoader, "__init__", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    path = (Path(__file__).resolve().parents[1] / "evidence"
            / "historical-learned-afterqual-2026-10-16.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        result = verifier.verify_capsule(json.load(stream))
    assert result["verified"] and result["both_complete400_cohorts_pass"]
    assert (result["events"], result["full_joint400_events"],
            result["aggregate_only_identity400_events"]) == (45, 11, 34)
    assert result["races_run"] == result["fits"] == result["network_requests"] == 0
    assert result["bootstrap_stability"]["saved_count_draws_replayed"] == 45_000
    assert result["bootstrap_stability"]["new_resampling"] == 0
    assert len(result["bootstrap_stability"]["events_with_trigger_discontinuity"]) == 5
    for cohort in result["cohorts"]["400"].values():
        summary = cohort["summaries"]["all"]
        assert summary["models"]["native"]["correct"] == 25
        assert summary["models"]["guard"]["correct"] == 28
        assert summary["grid18"]["correct"] == 27
