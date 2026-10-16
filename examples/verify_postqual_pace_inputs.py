"""Rebuild both historical pace policies with production code, offline.

This checks actual model inputs and strategy inference against the saved
400-trial sensitivity. It does not rerun races or authenticate source vintages.
"""

import argparse
import gzip
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from f1sim.data.current import CurrentSeasonDataLoader
from f1sim.models import Car, Driver, Track
from f1sim.simulation.race import RaceSimulator

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/postqual-pace-policy-inputs-2026-10-16.json.gz"
CONTENT_SHA256 = "f859f389cbc571dadc3d31dd56afb3ab468b83ae0fe657f5fee6e06f9d92b8e3"


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path=EVIDENCE):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def same_model(actual, expected):
    return actual.keys() == expected.keys() and all(
        math.isclose(actual[k], value, rel_tol=0., abs_tol=1e-12)
        if isinstance(value, float) else actual[k] == value
        for k, value in expected.items()
    )


def verify_data(document, *, expected_seal=CONTENT_SHA256):
    body = {k: v for k, v in document.items() if k != "content_sha256"}
    require(digest(body) == document.get("content_sha256"), "Input fixture seal differs")
    require(expected_seal is None or document["content_sha256"] == expected_seal,
            "Pinned input fixture identity differs")
    require(document.get("kind") == "historical_postqual_pace_builder_fixtures"
            and document.get("schema_version") == 1, "Unsupported input fixtures")
    events = document["events"]
    require(len(events) == 40 and len({(e["year"], e["round"]) for e in events}) == 40,
            "Expected exactly 40 unique historical events")

    def no_network(*args, **kwargs):
        raise AssertionError("Historical model-input verification must not fetch data")

    loader = CurrentSeasonDataLoader(http_getter=no_network)
    simulator = RaceSimulator()
    strategy_checks = 0
    for event in events:
        args = event["builder_args"]
        require(args["year"] == event["year"]
                and int(args["target_event"]["round"]) == event["round"],
                "Target identity differs")
        require(all(0 < int(r["round"]) < event["round"]
                    for r in args["race_rows"] + args["quali_rows"]),
                "Target/future performance appears in history")
        require(all(int(r["round"]) == event["round"] for r in args["target_qualifying_rows"]),
                "Qualifying belongs to another target")
        require(set(event["policies"]) == {"season_points", "qualifying_led"},
                "Pace policy identities differ")
        expected_drivers = {r["id"]: r for r in event["drivers"]}
        context = event["race_context"]
        track = Track.model_validate(context["track"])
        for policy, saved in event["policies"].items():
            stats = loader._build_driver_stats(**deepcopy(args), postqual_pace_policy=policy)
            drivers = {d.id: d.model_dump() for d in loader.create_drivers_from_stats(stats)}
            cars = {k: c.model_dump() for k, c in loader.create_cars_from_stats(stats).items()}
            require(drivers.keys() == expected_drivers.keys()
                    and cars.keys() == saved["cars"].keys(),
                    "Historical roster changed")
            # Readable names differed between source normalizers. Physics never
            # uses these labels; preserve archived labels when rebuilding hashes.
            for identity, expected in expected_drivers.items():
                drivers[identity]["name"] = expected["name"]
                require(same_model(drivers[identity], expected), "Historical driver traits changed")
            for identity, expected in saved["cars"].items():
                cars[identity]["team_name"] = expected["team_name"]
                require(same_model(cars[identity], expected), "Historical car inputs changed")
            # Hash the original recorded values after verifying numerical parity.
            # Different process dictionary ordering can move a mean by one ULP.
            supplied = {"drivers": event["drivers"], "cars": saved["cars"], **context}
            require(digest(supplied) == saved["input_sha256"], "400-trial model input hash differs")
            # The only later simulator changes affect strategy anchoring and a
            # floating threshold. Prove identity for this full-field cohort.
            driver_models = [Driver.model_validate(d) for d in supplied["drivers"]]
            car_models = {k: Car.model_validate(c) for k, c in cars.items()}
            views = simulator._strategy_car_views(
                driver_models, car_models, context["starting_grid"],
            )
            require(max(c.base_pace for c in views.values()) == 1.,
                    "Historical field anchor differs")
            for identity, car in car_models.items():
                old = ("conservative" if car.reliability < .9 or car.tire_degradation_factor > 1.08
                       else "aggressive" if car.base_pace > .82 and track.overtake_difficulty > .55
                       else "balanced")
                require(simulator._infer_team_strategy(views[identity], track).value == old,
                        "Historical strategy inference changed")
                strategy_checks += 1
        pre = {**args, "target_qualifying_rows": []}
        legacy = loader._build_driver_stats(**deepcopy(pre), postqual_pace_policy="season_points")
        timing = loader._build_driver_stats(**deepcopy(pre), postqual_pace_policy="qualifying_led")
        require(legacy == timing, "Pre-qualifying models differ")
    return {"events": len(events), "policies_per_event": 2,
            "verified_archived_400_trial_input_hashes": 2*len(events),
            "rebuilt_model_absolute_tolerance": 1e-12,
            "strategy_identity_checks": strategy_checks,
            "pre_qualifying_unchanged": True, "simulations_run": 0,
            "content_sha256": document["content_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    print(json.dumps(verify_data(read(args.path)), indent=2))


if __name__ == "__main__":
    main()
