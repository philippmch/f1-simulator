# Finite race tyre pools

An optional `tire_inventory` constrains each listed driver to named physical
sets throughout the race. The pool includes the opening set. Unlisted drivers
keep the existing unlimited-set policy. Qualifying uses independent sets;
the simulator does not infer real weekend allocations or qualifying usage.

## Inputs

In the dashboard's **Race tyre sets (optional)** field or the CLI:

```powershell
python examples/simulate_race.py --race 1 --tire-inventory "VER=soft@5,medium,hard;NOR=soft,hard" --starting-tyres "VER=soft@5" --export
```

Separate drivers with semicolons or newlines and sets with commas. Omit `@age`
for zero prior laps. Blank input leaves every driver unlimited. Use driver
codes from the loaded roster and canonical compounds: `soft`, `medium`, `hard`,
`intermediate`, or `wet`.

Python `MonteCarloRunner` and HTTP `/api/run` accept the equivalent mapping:

```python
tire_inventory = {
    "VER": [
        {"id": "used-soft", "compound": "soft", "age": 5},
        {"id": "race-medium", "compound": "medium", "age": 0},
        {"compound": "hard"},
    ],
}
```

Each listed driver needs 1–20 sets. Ages must be integers from 0 through 1000;
booleans, fractional values and numeric strings are rejected. This input bound
does not imply a realistic tyre lifetime. IDs must be nonempty and unique per
driver. Omitted IDs become `set-1`, `set-2`, etc.; omitted ages become zero.
Only `id`, `compound` and `age` are accepted in input records. Unknown drivers
are rejected. An explicit `starting_tires`/`starting_tire_ages` choice must have
an exact compound-and-age match in that driver's pool. No replacement set is
fabricated to satisfy an override or a later strategy decision.

## Race behavior and planning

Both race engines retain physical IDs and wear. Removing an undamaged set
makes it available for reuse at its accumulated age. Prior wear affects pace
but does not count as race distance or satisfy compound-use requirements.
A punctured set becomes unavailable. If a required safe replacement cannot be
fitted, or the final mandatory distinct-compound change cannot be satisfied,
the model withdraws the car as a DNF before pit service. This is explicit model
behavior for an exhausted pool, not a prediction of a team's real response.

Automatic opening selection and subsequent decisions evaluate eligible sets
under the planner's bounded stop allowances and compound-use constraints.
The forecast uses deterministic lap costs and projected surface evolution;
actual races still sample events and service variation. It does not anticipate
random future weather changes or incidents, or jointly optimize both teammates.
Its result is conditional on these assumptions, not an exact global optimum
for the stochastic race. Heat cycles and tyre storage effects are not modeled.
Finite-pool planning adds work for each distinct driver, set age and forecast.
Automatic opening selection also compares complete policy paths. Start with a
small trial count when checking a new pool before running a large ensemble.

Red-flag fittings are free changes and can retain the current physical set.
They consume no paid stop. The physical-set ledger includes opening and later
fittings that never complete a lap; these do not earn compound-use credit.

## Search and benchmark

Future planning groups interchangeable physical IDs by compound and age while
retaining their number. Each set accumulates wear when it runs, including after
removal and reuse. Search skips a replacement branch only when an optimistic
remaining-time bound cannot improve the best cost already found.

That bound starts with the cheapest reachable running cost on each future lap.
For each physical set and age, it finds the smallest excess above that lap's
baseline over eligible future surfaces. It then adds the cheapest distinct uses
needed for the remaining distance. A set cannot supply its same age twice;
identical sets each supply their own uses. Ignoring use order, ages already
consumed earlier in the forecast and pit charges makes this a lower bound,
not an executable strategy. It does not assume that older tyres are always
slower, so lap-time floors and nonmonotone cost curves remain supported.
Arithmetic rounds the bound down to protect nearly tied branches. Unchanged
final stints also share a cost within each decision. These calculations remain
local to the forecast and preserve its weather, fuel and race-control context.

For externally timed chronological weather, each paid stop also advances the
candidate's weather clock. The running-cost bound considers the reachable
delayed surfaces, including compulsory stops after the elective allowance is
exhausted. A second bound retains the current set's exact wear and surface
timeline until its first future stop, charges that pit entry, then relaxes the
remaining running costs as above. Taking the stronger bound avoids expanding
unnecessary full-distance stint combinations without approximating tyre ages,
weather timing or the selected strategy. Free restart fits add no elapsed pit
time; future paid stops still advance their weather forecast.

Automatic opening selection evaluates one deterministic policy path per distinct
compound and prior age. Equivalent physical IDs receive the same score in their
original order. The complete pool remains available throughout each path, and
first-input tie-breaking remains unchanged. Actual races still replan as their
traffic, weather, wear and control conditions change.

The offline benchmark can include both finite inventory and automatic openings:

```powershell
python examples/benchmark_strategy_planning.py --engine standard --scenario wetting --inventory finite --opening automatic --drivers 4 --laps 53 --trials 1 --seed 42
```

Repeat with `--engine chronological`, or use `--opening explicit` to isolate
running strategy from opening selection. Increase driver or trial counts after
checking the smaller workload. Compare revisions using the same script and
arguments in fresh processes without competing CPU-heavy work. The outcome hash
covers complete race rows, including physical ledgers and final inventories,
plus qualifying, weather and event results. Check equality before comparing
timings; one matching workload does not prove equivalence for every input.

A local 2026-09-12 checkpoint ran that four-driver, 53-lap command against
revision `6a2d5a7` and the updated search in separate fresh processes,
using the same script, Python and dependencies. Each measurement is one cold
trial, so these are workload observations rather than general timing guarantees:

| Engine | `6a2d5a7` seconds | Updated search seconds | Outcome hash prefix (both) |
|---|---:|---:|---|
| Standard | 62.278 | 37.489 | `f70f5842efd2` |
| Chronological | 64.479 | 36.829 | `fb9b3db4e246` |

The three benchmark sets are distinct, so those gains do not rely on duplicate
opening candidates. Separate native-policy tests compare each equivalent ID's
outcome independently and verify that three identical wet sets need one opening
path while retaining all three physical sets.

## Saved inputs and audit records

The input pool is saved separately from the final pool, in request/scenario
provenance and replay inputs. A nonempty finite pool uses snapshot schema 4;
older supported snapshots retain their existing unlimited-set interpretation.
Unsupported schemas are rejected instead of silently dropping the constraint.

Finite race results include `tire_set_history` records with `lap`, `kind`
(`start`, `pit`, `red_flag`), `set_id`, `compound`, `age_at_fit`, `age_at_end`
and `laps_used`. The result's `tire_inventory` is the final snapshot, containing
`id`, `compound`, `age`, `current`, `available` and `unavailable`. Do not pass
that final snapshot directly as an input pool: its extra state flags are not
input fields. Unlimited or legacy results have no finite ledger.
Wear and `laps_used` count completed laps; an interrupted retirement lap is
not credited as a full lap. A paid fitting on that lap remains in the ledger.

Dashboard and HTML reports display both fittings and final pools. JSON and
race CSV exports preserve the records, while paid-stop details additionally
identify outgoing/incoming sets and incoming wear. Compound strategy sequences
and paid-stop totals remain separate views: a free fitting is not a paid stop.
