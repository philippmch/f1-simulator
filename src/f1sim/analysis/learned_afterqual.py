"""Frozen after-qualifying inference; no fitting, I/O feeds or random draws.

This model supplies a separate experimental forecast.  Its scores do not alter
the drivers, cars, qualifying, strategy or native simulated race paths.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from math import isfinite
from numbers import Integral, Real
from typing import Mapping

import numpy as np

FEATURE_SCHEMA = "afterqual_grid_history_16_v1"
FEATURE_NAMES = (
    "grid", "qualifying_rank", "pit_lane", "q1_gap", "q2_gap", "q3_gap",
    "q1_available", "q2_available", "q3_available", "teammate_latest_gap",
    "team_q1_gap", "previous_team_race", "previous_team_qualifying",
    "previous_driver_race_residual", "previous_driver_qualifying_residual",
    "qualifying_minus_grid",
)
ARTIFACT_KIND = "frozen_afterqual_grid_residual_v1"
DEFAULT_ARTIFACT = "learned_afterqual_2026_16.json"


def canonical_digest(value) -> str:
    """Digest the portable JSON representation used by the frozen studies."""
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _detached(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def _identities(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty identity list")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")
    return tuple(values)


def _matrix(value, n, width, name):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{name} must have {n} rows")
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError(f"{name} must have {width} columns")
        result.append(tuple(_number(item, name) for item in row))
    return tuple(result)


@dataclass(frozen=True)
class AfterQualifyingFeatures:
    """Ordered, validated provider identities and physical simulation codes."""

    year: int
    round: int
    ids: tuple[str, ...]
    codes: tuple[str, ...]
    grid_ids: tuple[str, ...]
    tree_x: tuple[tuple[float, ...], ...]
    risk: tuple[tuple[float, ...], ...]
    context: dict

    @classmethod
    def from_mapping(cls, value: Mapping):
        if not isinstance(value, Mapping):
            raise ValueError("feature snapshot must be an object")
        if "content_sha256" in value and value["content_sha256"] != canonical_digest(
            {key: item for key, item in value.items() if key != "content_sha256"}
        ):
            raise ValueError("feature snapshot content seal mismatch")
        year = _integer(value.get("year"), "year")
        number = _integer(value.get("round"), "round")
        ids = _identities(value.get("ids"), "ids")
        codes = _identities(value.get("codes"), "codes")
        grid = _identities(value.get("grid_ids"), "grid_ids")
        n = len(ids)
        if n < 2 or len(codes) != n or len(grid) != n or set(grid) != set(ids):
            raise ValueError("ids, codes and grid_ids must be one-to-one complete fields")
        if value.get("feature_schema", FEATURE_SCHEMA) != FEATURE_SCHEMA:
            raise ValueError("unsupported after-qualifying feature schema")
        tree_x = _matrix(value.get("tree_x"), n, 16, "tree_x")
        risk = _matrix(value.get("risk"), n, 3, "risk")
        if "x" in value and _matrix(value["x"], n, 15, "x") != tuple(
            row[:15] for row in tree_x
        ):
            raise ValueError("x must match the first fifteen tree_x columns")
        ordinal = {identity: index for index, identity in enumerate(grid)}
        for identity, row, risk_row in zip(ids, tree_x, risk, strict=True):
            if abs(row[0] - (1 - ordinal[identity] / (n - 1))) > 1e-12:
                raise ValueError("grid feature must match the published grid ordinal")
            # The frozen field can exclude a nonstarter whose qualifying place
            # still appears in the published classification (2025 R9: P20 in
            # a nineteen-driver race field).  Preserve that negative rank.
            if not -1 <= row[1] <= 1 or any(row[i] not in (0, 1) for i in (2, 6, 7, 8)):
                raise ValueError("invalid qualifying rank or binary feature")
            if any(not -5 <= row[i] <= 5 for i in (3, 4, 5, 9, 10)):
                raise ValueError("qualifying gaps must lie within the frozen clipping bounds")
            if any(not 0 <= row[i] <= 1 for i in (11, 12)):
                raise ValueError("team history levels must be probabilities")
            if any(not -1 <= row[i] <= 1 for i in (13, 14, 15)):
                raise ValueError("driver residuals must lie within [-1, 1]")
            if abs(row[15] - (row[1] - row[0])) > 1e-12:
                raise ValueError("qualifying_minus_grid must match the rank features")
            if risk_row[0] != 1 or any(not 0 <= item <= 1 for item in risk_row[1:]):
                raise ValueError("risk must contain an intercept and two nonfinish histories")
        numerical = {"year", "round", "ids", "codes", "grid_ids", "tree_x", "x", "risk",
                     "feature_schema", "content_sha256"}
        context = _detached({key: item for key, item in value.items() if key not in numerical})
        return cls(year, number, ids, codes, grid, tree_x, risk, context)

    def to_dict(self):
        return {
            **_detached(self.context), "feature_schema": FEATURE_SCHEMA,
            "year": self.year, "round": self.round, "ids": list(self.ids),
            "codes": list(self.codes), "grid_ids": list(self.grid_ids),
            "x": [list(row[:15]) for row in self.tree_x],
            "tree_x": [list(row) for row in self.tree_x],
            "risk": [list(row) for row in self.risk],
        }


def _validate_tree(tree, depth=0):
    if not isinstance(tree, dict):
        raise ValueError("model tree must be an object")
    if "leaf" in tree:
        _number(tree["leaf"], "tree leaf")
        if "left" in tree or "right" in tree or "feature" in tree:
            raise ValueError("leaf cannot also be a branch")
        return
    if depth >= 2:
        raise ValueError("frozen model trees must have depth at most two")
    feature = _integer(tree.get("feature"), "tree feature", 0)
    if feature >= len(FEATURE_NAMES):
        raise ValueError("invalid tree feature")
    _number(tree.get("threshold"), "tree threshold")
    _validate_tree(tree.get("left"), depth + 1)
    _validate_tree(tree.get("right"), depth + 1)


@dataclass(frozen=True)
class FrozenAfterQualifyingModel:
    """A sealed artifact and its fixed training boundary; never fitted here."""

    document: dict

    @classmethod
    def from_mapping(cls, value: Mapping):
        if not isinstance(value, Mapping):
            raise ValueError("model artifact must be an object")
        document = _detached(dict(value))
        seal = document.get("content_sha256")
        if seal != canonical_digest({k: v for k, v in document.items()
                                     if k != "content_sha256"}):
            raise ValueError("model artifact content seal mismatch")
        if document.get("kind") != ARTIFACT_KIND or document.get("schema_version") != 1:
            raise ValueError("unsupported model artifact")
        if document.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("unsupported model feature schema")
        for name in ("source_model_sha256", "source_input_sha256", "source_dataset_sha256"):
            seal = document.get(name)
            if (not isinstance(seal, str) or len(seal) != 64
                    or any(character not in "0123456789abcdef" for character in seal)):
                raise ValueError(f"artifact requires a valid {name}")
        model = document.get("model", {})
        if (model.get("feature_names") != list(FEATURE_NAMES)
                or model.get("learning_rate") != .1 or model.get("residual_bound") != 1
                or model.get("grid_scale") != 18):
            raise ValueError("model must retain the frozen inference specification")
        trees = model.get("trees")
        if not isinstance(trees, list) or len(trees) != 100:
            raise ValueError("model must contain the frozen one hundred trees")
        for tree in trees:
            _validate_tree(tree)
        beta = model.get("risk_beta")
        if not isinstance(beta, list) or len(beta) != 3:
            raise ValueError("model must retain its three-coefficient risk head")
        for item in beta:
            _number(item, "risk coefficient")
        keys = model.get("training_event_keys")
        if not isinstance(keys, list) or not keys:
            raise ValueError("model training event keys are required")
        checked = []
        for key in keys:
            if not isinstance(key, list) or len(key) != 2:
                raise ValueError("invalid training event key")
            checked.append(tuple(_integer(item, "training event key") for item in key))
        if checked != sorted(set(checked)):
            raise ValueError("training keys must be unique and increasing")
        if document.get("training_boundary") != list(checked[-1]):
            raise ValueError("artifact boundary does not match its training history")
        return cls(document)

    @property
    def model(self):
        return self.document["model"]

    def to_dict(self):
        return _detached(self.document)


def load_frozen_afterqual_model() -> FrozenAfterQualifyingModel:
    """Load the packaged candidate; callers still enforce its live cutoff."""
    document = json.loads(files("f1sim.data").joinpath(DEFAULT_ARTIFACT).read_text("utf-8"))
    return FrozenAfterQualifyingModel.from_mapping(document)


def stable_sigmoid(values):
    """The expit function without an optimizer or SciPy runtime dependency."""
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1 + negative_exp)
    return result


@lru_cache(maxsize=1)
def _quadrature():
    return np.polynomial.legendre.leggauss(512)


def _joint_marginals(strength, risk):
    """Exact port of the sealed independent-survival/PL quadrature."""
    n = len(strength)
    log_rates = strength - max(strength)
    if np.ptp(log_rates) >= 150:
        raise ValueError("competitive strength range exceeds the frozen quadrature")
    rates = np.exp(log_rates)
    survives = 1 - risk
    quad_x, quad_w = _quadrature()
    low, high = -40., max(40., -min(log_rates) + 30.)
    u = (quad_x + 1) * (high - low) / 2 + low
    t = np.exp(u)
    w = quad_w * (high - low) / 2 * t
    exp_times = np.exp(-t[:, None] * rates)
    a = risk + survives * exp_times
    b = survives * (1 - exp_times)
    odds = b / a
    p0 = np.prod(a, axis=1)[:, None] / a
    e1 = odds.sum(axis=1)[:, None] - odds
    e2 = .5 * (e1 ** 2 - (np.square(odds).sum(axis=1)[:, None] - odds ** 2))
    density = survives * rates * exp_times * w[:, None]
    winner = np.sum(density * p0, axis=0)
    podium = np.sum(density * p0 * (1 + e1 + e2), axis=0)
    all_retired = float(np.prod(risk))
    winner += all_retired / n
    retired_odds = survives / risk
    r1 = retired_odds.sum() - retired_odds
    r2 = .5 * (r1 * r1 - (np.square(retired_odds).sum() - retired_odds ** 2))
    for count, multiplier in enumerate((np.ones(n), r1, r2)):
        if count < n:
            podium += all_retired * multiplier * (min(3, n) - count) / (n - count)
    pair = rates[None, :] / (rates[:, None] + rates[None, :])
    np.fill_diagonal(pair, 0.)
    expected = survives * (1 + (pair * survives[None, :]).sum(axis=1))
    expected += risk * ((n + 1) / 2 + (survives.sum() - survives) / 2)
    if abs(winner.sum() - 1) >= 1e-8 or abs(podium.sum() - min(3, n)) >= 1e-8:
        raise ValueError("frozen numerical integration failed its mass checks")
    winner /= winner.sum()
    podium *= min(3, n) / podium.sum()
    return winner, np.clip(podium, 0., 1.), expected, all_retired


def predict_after_qualifying(features, artifact=None) -> dict:
    """Infer from ordered frozen features, rejecting target/training overlap."""
    features = AfterQualifyingFeatures.from_mapping(
        features.to_dict() if isinstance(features, AfterQualifyingFeatures) else features
    )
    if artifact is None:
        artifact = load_frozen_afterqual_model()
    else:
        artifact = FrozenAfterQualifyingModel.from_mapping(
            artifact.to_dict() if isinstance(artifact, FrozenAfterQualifyingModel) else artifact
        )
    if (features.year, features.round) <= tuple(artifact.document["training_boundary"]):
        raise ValueError("forecast target must be strictly later than every training event")
    model = artifact.model
    x = np.asarray(features.tree_x)
    residual = np.zeros(len(features.ids))
    for tree in model["trees"]:
        leaves = []
        for row in x:
            node = tree
            while "leaf" not in node:
                node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            leaves.append(node["leaf"])
        residual = np.clip(residual + model["learning_rate"] * np.asarray(leaves),
                           -model["residual_bound"], model["residual_bound"])
    strength = model["grid_scale"] * x[:, 0] + residual
    risk = np.clip(stable_sigmoid(np.asarray(features.risk) @ np.asarray(model["risk_beta"])),
                   .01, .6)
    winner, podium, expected, all_retired = _joint_marginals(strength, risk)
    return {
        "driver_ids": list(features.ids), "codes": list(features.codes),
        "strengths": strength.tolist(), "winner": winner.tolist(), "podium": podium.tolist(),
        "expected_position": expected.tolist(), "nonfinish": risk.tolist(),
        "winner_probabilities": dict(zip(features.codes, winner.tolist(), strict=True)),
        "joint_distribution": "independent_survival_pl_competitive_order_v1",
        "numerical_method": "512_node_log_time_gauss_legendre",
        "all_nonfinish_probability": all_retired,
        "artifact_sha256": artifact.document["content_sha256"],
        "source_model_sha256": artifact.document["source_model_sha256"],
        "training_boundary": list(artifact.document["training_boundary"]),
        "feature_schema": FEATURE_SCHEMA,
    }
