"""Rerun historical after-qualifying forecasts offline, then score their saved trials."""

import argparse
import hashlib
import json
import math
import os
import sys
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import monotonic

import numpy as np

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.provenance import simulation_runtime
from f1sim.data.current import CurrentSeasonDataLoader
from f1sim.data.event_tire_rules import get_event_tire_rules
from f1sim.models import Track, Weather
from f1sim.simulation.race import result_is_classified

try:
    from . import verify_historical_qualifying_led as labels_verifier
    from . import verify_postqual_pace_inputs as inputs_verifier
except ImportError:  # Direct CLI execution puts examples/ on the module search path.
    import verify_historical_qualifying_led as labels_verifier
    import verify_postqual_pace_inputs as inputs_verifier


# The compact input fixture has no calendar dates. Use a conservative start of
# qualifying day for these two sourced exceptions, rather than today's clock.
# These calendar pages verify the dates, not original publication receipts.
_TIRE_RULE_CUTOFFS = {
    (2025, 8): ("2025-05-24T00:00:00Z", "https://www.formula1.com/en/racing/2025/monaco"),
    (2025, 23): ("2025-11-29T00:00:00Z", "https://www.formula1.com/en/racing/2025/qatar"),
}


def no_network(*args, **kwargs):
    raise RuntimeError("Historical backtests must remain offline")


def offline_audit(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo", "urllib.Request"}:
        no_network()


def serial(value):
    if is_dataclass(value):
        fields = asdict(value)
        if fields.get("race_tire_usage") is None:
            fields.pop("race_tire_usage", None)
        return serial(fields)
    if isinstance(value, Enum):
        return serial(value.value)
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(item) for item in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": "nan" if math.isnan(value) else
                "positive_infinity" if value > 0 else "negative_infinity"}
    return value


def sealed(value):
    body = serial(value)
    return {**body, "content_sha256": inputs_verifier.digest(body)}


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checkpoint(path, value):
    temporary = path.with_name(path.name + ".tmp")
    write_new(temporary, sealed(value))
    os.replace(temporary, path)


def forecast(event, policy, trials, workers, *, event_tire_rules=False):
    loader = CurrentSeasonDataLoader(http_getter=no_network)
    stats = loader._build_driver_stats(
        **deepcopy(event["builder_args"]), postqual_pace_policy=policy,
    )
    drivers = loader.create_drivers_from_stats(stats)
    cars = loader.create_cars_from_stats(stats)
    # Preserve the fixture's readable labels after rebuilding all physics traits.
    names = {d["id"]: d["name"] for d in event["drivers"]}
    for driver in drivers:
        driver.name = names[driver.id]
    for identity, car in cars.items():
        car.team_name = event["policies"][policy]["cars"][identity]["team_name"]
    context = deepcopy(event["race_context"])
    rules, rule_sources = {}, {}
    if event_tire_rules:
        cutoff = _TIRE_RULE_CUTOFFS.get((event["year"], event["round"]))
        rules, rule_sources = get_event_tire_rules(
            event["year"], event["builder_args"]["target_event"],
            cutoff=cutoff[0] if cutoff else None,
        )
        if rules and cutoff is None:
            raise ValueError("Historical event tyre rules require a verified qualifying cutoff")
        for source in rule_sources.values():
            source.update({"historical_cutoff": cutoff[0], "schedule_source_url": cutoff[1],
                           "cutoff_basis": "start of qualifying day, UTC; conservative"})
    supplied = {"drivers": [d.model_dump() for d in drivers],
                "cars": {k: c.model_dump() for k, c in cars.items()}, **context,
                **({"race_tire_rules": rules} if rules else {})}
    runner = MonteCarloRunner(
        drivers, cars, Track.model_validate(context["track"]),
        Weather.model_validate(context["weather"]), seed=context["seed"],
        race_engine="chronological", rng_policy="isolated_weather_v1",
        starting_grid=context["starting_grid"], pit_lane_starters=context["pit_lane_starters"],
        **({"race_tire_rules": rules} if rules else {}),
    )
    result = runner.run(trials, parallel=workers > 1 and trials > 1, max_workers=workers)
    if result.num_simulations != trials or result.winner_allocation is not None:
        raise ValueError("Native trial identity changed")
    identifiers = set(context["starting_grid"])
    marginals = {d: {"position_sum": 0, "podium_count": 0, "nonfinish_count": 0}
                 for d in identifiers}
    for race in result.race_results:
        if {row.driver_id for row in race} != identifiers or sorted(
            row.position for row in race
        ) != list(range(1, len(identifiers) + 1)):
            raise ValueError("Joint trial does not preserve the complete field")
        for row in race:
            value = marginals[row.driver_id]
            value["position_sum"] += row.position
            value["podium_count"] += int(row.position <= 3 and result_is_classified(row))
            value["nonfinish_count"] += int(row.status.value != "finished")
    wins = {d: int(result.driver_stats[d].wins) for d in sorted(identifiers)}
    return serial({"policy": policy, "trials": trials, "seed": context["seed"],
                   "trial_seeds": list(range(context["seed"], context["seed"] + trials)),
                   "input_sha256": inputs_verifier.digest(supplied),
                   "archived_input_sha256": event["policies"][policy]["input_sha256"],
                   "input_snapshot": result.input_snapshot, "wins": wins,
                   "no_classified_winner_count": trials - sum(wins.values()),
                   "marginals": marginals, "joint_race_results": result.race_results,
                   "joint_qualifying_results": result.qualifying_results,
                   "race_scoring_contexts": result.get_race_scoring_contexts(),
                   "weather_histories": result.weather_histories,
                   "control_schedule_histories": result.control_schedule_histories,
                   **({"race_tire_rules_provenance": rule_sources} if rules else {})})


def score(forecast_value, observed):
    n = forecast_value["trials"]
    winner = observed["observed_winner"]
    labels = observed["observed_drivers"]
    identifiers = set(forecast_value["wins"])
    if (set(forecast_value["marginals"]) != identifiers or set(labels) != identifiers
            or winner not in identifiers):
        raise ValueError("Scoring labels and forecast must contain the same complete field")
    probability = forecast_value["wins"][winner] / n
    value = {"winner_brier": sum((v / n - (d == winner)) ** 2
                                  for d, v in forecast_value["wins"].items())
             + (forecast_value["no_classified_winner_count"] / n) ** 2,
             "winner_log_loss": -math.log(probability) if probability else None,
             "winner_log_loss_infinite": probability == 0,
             "zero_probability_observed_winner": probability == 0}
    for name, field, target in (("position_mae", "position_sum", "position"),
                                ("podium_brier", "podium_count", "podium"),
                                ("nonfinish_brier", "nonfinish_count", "dnf")):
        errors = [abs(v[field] / n - labels[d][target]) for d, v in
                  forecast_value["marginals"].items()]
        value[name] = sum(e if name == "position_mae" else e * e for e in errors) / len(errors)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2025, 2026))
    parser.add_argument("--round", type=int)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--event-tyre-rules", action="store_true",
        help="Apply sourced pre-qualifying 2025 Monaco/Qatar rules; preserve original archives.",
    )
    parser.add_argument("--policy", choices=("both", "season_points", "qualifying_led"),
                        default="both")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.trials <= 10000 or not 1 <= args.workers <= 32:
        parser.error("trials must be 1..10000 and workers 1..32")
    if args.round is not None and not 1 <= args.round <= 24:
        parser.error("round must be 1..24")
    if not args.list and (args.year is None or args.output is None):
        parser.error("--year and --output are required unless --list is used")
    partial = None if args.output is None else args.output.with_name(args.output.name + ".partial")
    if args.output is not None and any(p.exists() for p in (
        args.output, partial, partial.with_name(partial.name + ".tmp"),
    )):
        parser.error("Refusing to overwrite an output or its partial checkpoint")
    fixture = inputs_verifier.read()
    body = {k: v for k, v in fixture.items() if k != "content_sha256"}
    if inputs_verifier.digest(body) != fixture["content_sha256"] or fixture[
        "content_sha256"
    ] != inputs_verifier.CONTENT_SHA256:
        raise ValueError("Pinned historical input fixture identity differs")
    events = [e for e in fixture["events"] if (args.year is None or e["year"] == args.year)
              and (args.round is None or e["round"] == args.round)]
    if not events:
        parser.error("No available historical event matches the requested year/round")
    if args.list:
        print(json.dumps([{"year": e["year"], "round": e["round"]} for e in events]))
        return None
    inputs_verifier.verify_data(fixture)
    started = monotonic()
    policies = ("season_points", "qualifying_led") if args.policy == "both" else (args.policy,)
    saved = {"kind": "offline_historical_after_qualifying_native_backtest", "format_version": 1,
             "started_at": datetime.now(timezone.utc).isoformat(), "runtime": simulation_runtime(),
             "runner_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "fixture_content_sha256": fixture["content_sha256"], "workers": args.workers,
             "trials_per_model_event": args.trials, "scoring_performed": False,
             "limits": {"fixed_dry": True, "current_2026_rules_on_2025": True,
                        "legacy_rules_reconstructed": False,
                        "authentic_historical_availability": False,
                        "historical_distance_faults": ["2025R5 Jeddah57vs50", "2025R7 Imola57vs63"],
                        "prospective_skill_established": False}, "events": []}
    if args.event_tyre_rules:
        saved["event_tire_rules"] = "sourced_pre_qualifying_registry_v1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(partial, sealed(saved))
    for event in events:
        record = {"year": event["year"], "round": event["round"],
                  "starting_grid": event["race_context"]["starting_grid"].copy(), "forecasts": {}}
        saved["events"].append(record)
        for policy in policies:
            record["forecasts"][policy] = forecast(
                event, policy, args.trials, args.workers,
                **({"event_tire_rules": True} if args.event_tyre_rules else {}),
            )
            checkpoint(partial, saved)
            print(json.dumps({"year": event["year"], "round": event["round"],
                              "policy": policy, "trials_saved": args.trials}), flush=True)
    # Outcomes are opened only after every forecast has been checkpointed.
    labels = json.loads(labels_verifier.EVIDENCE.read_text(encoding="utf-8"))
    labels_verifier.assess(labels)
    lookup = {(e["year"], e["round"]): e for e in labels["events"]}
    for event in saved["events"]:
        observed = lookup[event["year"], event["round"]]
        event["observed_winner"] = observed["observed_winner"]
        event["observed_drivers"] = observed["observed_drivers"]
        grid = observed["starting_grid"]
        if grid != event["starting_grid"]:
            raise ValueError("Scoring reference grid differs from the forecast's frozen grid")
        event["grid_references"] = {}
        for scale in (6, 12, 18):
            weights = [math.exp(-scale * i / (len(grid) - 1)) for i in range(len(grid))]
            probability = [weight / sum(weights) for weight in weights]
            p = probability[grid.index(observed["observed_winner"])]
            event["grid_references"][str(scale)] = {
                "winner_brier": sum(v * v for v in probability) + 1 - 2 * p,
                "winner_log_loss": -math.log(p),
            }
        event["scores"] = {policy: score(value, observed)
                           for policy, value in event["forecasts"].items()}
    saved.update({"scoring_performed": True, "labels_content_sha256": labels["content_sha256"],
                  "elapsed_seconds": monotonic() - started})
    final = sealed(saved)
    write_new(args.output, final)
    partial.unlink()
    return final


if __name__ in {"__main__", "__mp_main__"}:
    sys.addaudithook(offline_audit)
if __name__ == "__main__":
    main()
