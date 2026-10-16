from copy import deepcopy

import pytest

from f1sim.analysis.learned_afterqual import canonical_digest, load_frozen_afterqual_model
from f1sim.analysis.learned_forecast import (
    NULL_WINNER,
    PRIMARY_TIE_POLICY,
    build_learned_forecast,
    project_winner_probabilities,
    validate_saved_learned_forecast,
)
from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.simulation.race import DriverStatus, RaceResult


def test_projection_matches_independent_mass_root_and_preserves_null():
    native = {"A": .1, "B": .6, "C": .25, NULL_WINNER: .05}
    analytic = {"A": .6, "B": .3, "C": .1}
    q, diagnostic = project_winner_probabilities(native, analytic)
    low, high = 0., 1.
    for _ in range(100):
        middle = (low + high) / 2
        mass = middle + sum(min(native[code], analytic[code] / analytic["A"] * middle)
                            for code in ("B", "C"))
        if mass < 1 - native[NULL_WINNER]:
            low = middle
        else:
            high = middle
    assert q["A"] == pytest.approx((low + high) / 2, abs=1e-14)
    assert q[NULL_WINNER] == native[NULL_WINNER]
    assert sum(q.values()) == pytest.approx(1)
    assert diagnostic["active"] == ["C", "B"]
    assert abs(diagnostic["j_stationarity_residual"]) < 1e-14


def test_native_top_tie_remains_identity_even_when_ratio_constraints_would_fail():
    native = {"A": .4, "B": .4, "C": .2, NULL_WINNER: 0.}
    analytic = {"A": .9, "B": .08, "C": .02}
    q, diagnostic = project_winner_probabilities(native, analytic)
    assert q == native
    assert diagnostic["status"] == "identity"


@pytest.mark.parametrize("native,analytic,reason", [
    ({"A": .5, "B": .5, NULL_WINNER: 0.}, {"A": .5, "B": .5},
     "nonunique_analytic_top"),
    ({"A": 0., "B": 1., NULL_WINNER: 0.}, {"A": .8, "B": .2}, "native_j_zero"),
    ({"A": .2, "B": .8, NULL_WINNER: 0.}, {"A": 1., "B": 0.},
     "zero_ratio_positive_native"),
])
def test_projection_reports_unsupported_support_without_inventing_paths(native, analytic, reason):
    q, diagnostic = project_winner_probabilities(native, analytic)
    assert q == native
    assert diagnostic["status"] == "unsupported"
    assert diagnostic["fallback"] == reason


def _stock_fixture():
    codes = ["A", "B", "C"]
    x = [[1 - i / 2, 1 - i / 2, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          .5, .5, 0., 0., 0.] for i in range(3)]
    snapshot = {"year": 2027, "round": 1, "ids": ["a", "b", "c"], "codes": codes,
                "grid_ids": ["a", "b", "c"], "tree_x": x, "risk": [[1., .15, .15]] * 3}
    artifact = load_frozen_afterqual_model().to_dict()
    artifact["model"]["trees"] = [{"leaf": 0.}] * 100
    artifact["content_sha256"] = canonical_digest(
        {key: value for key, value in artifact.items() if key != "content_sha256"}
    )

    def row(code, position, status=DriverStatus.FINISHED, classified=True):
        return RaceResult(code, code, "team", position, 100., 0., 1, 10., status,
                          classified=classified)

    races = [
        [row("A", 1, DriverStatus.DNF), row("B", 2), row("C", 3, DriverStatus.DNF, False)],
        [row("A", 3), row("B", 1), row("C", 2, DriverStatus.DNF, False)],
        [row("A", 2), row("B", 1), row("C", 3, DriverStatus.DNF, False)],
        [row("A", 3), row("B", 2, DriverStatus.DNF), row("C", 1, DriverStatus.DNF, False)],
    ]
    stats = {code: DriverStatistics(code, code, "team", wins={"A": 1, "B": 2, "C": 0}[code],
                                    positions=[r[i].position for r in races])
             for i, code in enumerate(codes)}
    return SimulationResults(4, "Test", stats, races, []), snapshot, artifact


def test_joint_weighting_uses_one_measure_for_every_marginal_and_keeps_native_results():
    result, snapshot, artifact = _stock_fixture()
    original = deepcopy(result)
    estimate = build_learned_forecast(result, snapshot, artifact,
                                      eligibility={"eligible": True, "scope": "test"})
    assert result == original
    assert estimate["status"] == "projected"
    q = estimate["win_probabilities"]
    assert q["A"] > q["B"]
    assert q["C"] == 0
    assert estimate["no_classified_winner_probability"] == .25
    assert estimate["native_winner_counts"] == {"A": 1, "B": 2, "C": 0, NULL_WINNER: 1}
    weights = estimate["trial_weights"]
    assert weights == pytest.approx([q["A"], q["B"] / 2, q["B"] / 2, .25])
    assert estimate["nonfinish_probabilities"]["A"] == pytest.approx(q["A"])
    assert estimate["nonfinish_probabilities"]["B"] == .25
    assert estimate["nonfinish_probabilities"]["C"] == pytest.approx(1)
    assert sum(estimate["podium_probabilities"].values()) == pytest.approx(2)
    assert sum(estimate["expected_positions"].values()) == pytest.approx(6)
    assert estimate["expected_positions"]["A"] == pytest.approx(
        q["A"] + 2.5 * q["B"] + .75
    )
    assert estimate["effective_sample_size"] == pytest.approx(1 / sum(w * w for w in weights))
    assert validate_saved_learned_forecast(estimate) == estimate


@pytest.mark.parametrize("mutation", ["missing_row", "duplicate_position", "raw_count",
                                    "allocation", "eligibility", "unsupported"])
def test_joint_forecast_refuses_incomplete_or_mixed_native_evidence(mutation):
    result, snapshot, artifact = _stock_fixture()
    eligibility = {"eligible": True}
    if mutation == "missing_row":
        result.race_results[0].pop()
    elif mutation == "duplicate_position":
        result.race_results[0][0].position = 2
    elif mutation == "raw_count":
        result.driver_stats["A"].wins += 1
    elif mutation == "allocation":
        result.winner_allocation = {}
    elif mutation == "eligibility":
        eligibility["eligible"] = False
    elif mutation == "unsupported":
        result.race_results[0][0].classified = False
        result.driver_stats["A"].wins = 0
    with pytest.raises(ValueError):
        build_learned_forecast(result, snapshot, artifact, eligibility=eligibility)


def test_saved_replay_recomputes_marginals_even_after_a_tamper_is_resealed():
    result, snapshot, artifact = _stock_fixture()
    estimate = build_learned_forecast(result, snapshot, artifact, eligibility={"eligible": True})
    estimate["nonfinish_probabilities"]["A"] += .01
    estimate["content_sha256"] = canonical_digest(
        {key: value for key, value in estimate.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="native joint replay"):
        validate_saved_learned_forecast(estimate)


def test_public_builder_requires_a_positive_eligibility_proof():
    result, snapshot, artifact = _stock_fixture()
    with pytest.raises(ValueError, match="eligibility proof"):
        build_learned_forecast(result, snapshot, artifact)


def test_saved_replay_tolerates_only_derived_roundoff_and_preserves_saved_points(monkeypatch):
    from f1sim.analysis import learned_afterqual

    result, snapshot, artifact = _stock_fixture()
    estimate = build_learned_forecast(result, snapshot, artifact, eligibility={"eligible": True})
    original = deepcopy(estimate)
    baseline = learned_afterqual._joint_marginals

    def changed_roundoff(strength, risk):
        winner, podium, expected, all_retired = baseline(strength, risk)
        winner[0] += 1e-15
        winner[1] -= 1e-15
        return winner, podium, expected, all_retired

    monkeypatch.setattr(learned_afterqual, "_joint_marginals", changed_roundoff)
    assert validate_saved_learned_forecast(estimate) == original
    estimate["win_probabilities"]["A"] += 1e-15
    with pytest.raises(ValueError, match="content seal"):
        validate_saved_learned_forecast(estimate)


def test_reported_identity_tie_uses_grid_even_when_learned_favourite_is_countergrid(monkeypatch):
    result, snapshot, artifact = _stock_fixture()
    snapshot["grid_ids"] = ["b", "a", "c"]
    snapshot["tree_x"][0][0] = snapshot["tree_x"][0][1] = .5
    snapshot["tree_x"][1][0] = snapshot["tree_x"][1][1] = 1.
    for index, race in enumerate(result.race_results):
        for row in race:
            row.position = (1 if row.driver_id == ("A" if index < 2 else "B") else
                            3 if row.driver_id == "C" else 2)
            row.status, row.classified = DriverStatus.FINISHED, True
    for code, stats in result.driver_stats.items():
        stats.wins = 2 if code in ("A", "B") else 0
        stats.positions = [next(row.position for row in race if row.driver_id == code)
                           for race in result.race_results]
    monkeypatch.setattr("f1sim.analysis.learned_forecast.predict_after_qualifying",
                        lambda *args: {"winner_probabilities": {"A": .7, "B": .2, "C": .1}})
    estimate = build_learned_forecast(result, snapshot, artifact, eligibility={"eligible": True})
    assert estimate["status"] == "identity"
    assert estimate["projection"]["j"] == "A"
    assert estimate["win_probabilities"] == {"A": .5, "B": .5, "C": 0.}
    assert estimate["winner_top_set"] == ["B", "A"]
    assert estimate["primary_winner"] == "B"
    assert estimate["primary_tie_policy"] == PRIMARY_TIE_POLICY


@pytest.mark.parametrize("classified_wins", [1, 2])
def test_reported_winner_accounts_for_null_and_places_it_after_driver_ties(classified_wins):
    result, snapshot, artifact = _stock_fixture()
    for index, race in enumerate(result.race_results):
        for row in race:
            row.position = {"A": 1, "B": 2, "C": 3}[row.driver_id]
            row.classified = index < classified_wins
    for code, stats in result.driver_stats.items():
        stats.wins = classified_wins if code == "A" else 0
        stats.positions = [{"A": 1, "B": 2, "C": 3}[code]] * 4
    estimate = build_learned_forecast(result, snapshot, artifact, eligibility={"eligible": True})
    expected = ["A", NULL_WINNER] if classified_wins == 2 else [NULL_WINNER]
    assert estimate["winner_top_set"] == expected
    assert estimate["primary_winner"] == expected[0]
    assert estimate["no_classified_winner_probability"] == (4 - classified_wins) / 4
    assert validate_saved_learned_forecast(estimate) == estimate
