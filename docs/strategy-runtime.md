# Native rain strategy runtime

The native transition planner now reuses its already projected weather surfaces
and one prepared lap evaluator within a forecast decision. This removes repeated
model restoration, evaluator preparation and suffix projection on scalar row
cache misses. It preserves the existing global row cache keys and hits, compound
choices, lap calculations, cancellation checks and random streams. Custom model
behavior, observed-surface rows and calls outside a native decision retain the
original path. The workspace holds one current record and expires at the outer
decision boundary, including exceptions and cancellation.

This is a simulation runtime improvement. It does not improve forecast accuracy.

## Frozen comparison, 18 October 2026

Baseline: `65ba7260d31e2dccd101cd2ee35e0d4a22886f2c`. The comparison used the
chronological engine, automatic opening strategy, unlimited tyre inventory,
22 drivers, 53 laps, no random weather changes, and three consecutive trials
with seeds 42, 43 and 44. Each baseline/candidate pair ran in fresh processes;
the second pair reversed execution order. Other project CPU work was paused.
Environment: Windows, Intel Core Ultra 9 285K, Python 3.11.9, NumPy 2.4.6,
Pydantic 2.13.5.

Before editing, acceptance required at least 10% mean paired improvement in both
total and warm steady-damp runtime, with no more than 3% mean paired regression
in dry or drying-race runtime. Warm runtime is the mean of trials 2 and 3 in a
process. Total runtime sums all three trials. State proof work was excluded
from the timing. No thresholds, variants or additional pairs were selected
after reading the results.

| Scenario | Baseline mean, seconds | Candidate mean, seconds | Mean paired total gain | Mean paired warm gain |
| --- | ---: | ---: | ---: | ---: |
| Steady damp | 45.100 | 37.283 | 17.35% | 22.72% |
| Dry | 3.108 | 3.102 | 0.19% | 0.12% |
| Drying / rain transition | 40.181 | 38.073 | 5.22% | 2.04% |

All predeclared gates passed. The dry result is effectively unchanged. These are
local measurements on fixed synthetic workloads, not a guarantee for every
track, custom extension or machine. In particular, finite-inventory and external
weather-clock planners have separate paths.

Raw total times in execution-pair order:

| Scenario | Pair 1 baseline → candidate | Pair 2 baseline → candidate | Pair 3 baseline → candidate |
| --- | ---: | ---: | ---: |
| Steady damp | 45.1378914 → 41.8010479 | 45.9368303 → 36.0296725 | 44.2241837 → 34.0188974 |
| Dry | 3.1022440 → 3.0825902 | 3.1513740 → 3.1168662 | 3.0713357 → 3.1070508 |
| Drying / rain transition | 40.3231523 → 37.4244967 | 38.7600876 → 37.4904598 | 41.4598657 → 39.3042042 |

## Exactness evidence

Every timed baseline/candidate pair produced identical complete race,
qualifying, weather-history and event-statistic output hashes. The existing
`_running_row` cache statistics also matched exactly: damp had 858,350 hits and
91,010 misses; dry had none; drying had 139,830 hits and 4,777 misses. Cache size
and capacity matched. Thus the improvement did not change the search or cache
workload.

A separate untimed replay compared seed 42 in each scenario. It encoded every
instance field of the complete native race engine and simulator, including
nested models, dictionary insertion order, mutable alias relationships, every
RNG object/state and the event serial counter observed without advancing it.
All fields matched exactly. This state proof covers one seed per scenario;
the timed output equality covers all three seeds.

The 16 focused tests compare every returned row cell using `float.hex()` against
the original independent suffix path. They cover ordinary, repeated, skipped
and scheduled weather updates, drainage to zero, equilibrium weather, absolute
lap/fuel distance, warmup, fallback on key mismatches and observed surfaces,
custom scope isolation, nested replacement, cancellation and object release.
An independent run also passed 741 relevant existing tests.

## Reproduction

In both the baseline checkout and the updated checkout, with the same Python
environment, run the following in a fresh process for each sample. Repeat for
`dry` and `rain_transition`; alternate baseline/candidate order across three
pairs. Compare the complete `outcome_sha256` as well as total and later-trial
times. The benchmark performs no network access.

```powershell
python examples/benchmark_strategy_planning.py --engine chronological --scenario steady_damp --opening automatic --inventory unlimited --trials 3 --drivers 22 --laps 53 --seed 42
python -m pytest tests/test_native_prepared_transition_rows.py -q
```

The frozen measured source SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `src/f1sim/models/_native.py` | `05215e7b870a23482c4e46521aac213e030c12f627c7b8e6919039080a443636` |
| `src/f1sim/simulation/rain_strategy.py` | `cba5c95ae2f96d580fc47f018b47615eb066c475f9c485f00646b10e1ad39d3e` |

Full local protocol, raw pairs, source snapshots, state encoder and replay
proofs are retained under
`output/native-prepared-transition-rows-2026-10-18/`. That directory is private
ignored evidence and is not required by the application or tests.
