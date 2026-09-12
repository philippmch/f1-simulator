"""The benchmark compares outcomes independently of process cache warmth."""

import runpy
from pathlib import Path


def test_benchmark_digest_excludes_timings_and_cache_warmth():
    benchmark = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                  / "examples" / "benchmark_strategy_planning.py"))["benchmark"]
    cold = benchmark(drivers=4, laps=8, trials=2)
    warm = benchmark(drivers=4, laps=8, trials=2)
    changed = benchmark(drivers=4, laps=8, trials=2, seed=43)
    assert cold["outcome_sha256"] == warm["outcome_sha256"]
    assert cold["outcome_sha256"] != changed["outcome_sha256"]
    assert len(cold["outcome_sha256"]) == 64
    assert len(cold["trial_seconds"]) == cold["trials"] == 2
    assert all(value >= 0 for value in cold["trial_seconds"])
    assert benchmark(drivers=1, laps=2, trials=1)["later_trial_mean_seconds"] is None
