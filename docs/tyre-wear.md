# Tyre wear and strategy

Tyre wear continues to cost lap time after the numerical grip indicator reaches
its lower bound. Previously, `time_penalty_per_lap()` derived wear loss from
`grip_at_lap()`, which stops falling at 0.5. For several default compounds that
bound was reached before their configured cliff. A much older set could then
have the same wear cost as a moderately used set, weakening pit incentives.

## Curve and interpretation

For completed tyre age `a`, cliff age `c`, management rating `m`, degradation
rate `d` and cliff multiplier `k`, the accumulated wear loss is:

```text
r = d × (2 − m)
wear = r × min(a, c) + r × k × max(a − c, 0)
wear seconds = reference lap time × 0.03 × wear
```

The penalty starts at zero and is continuous at the cliff. Its slope increases
by `k` after the threshold. Zero configured degradation stays zero. The wear
index is an abstract pace cost, not a fraction of rubber removed. Existing
driver, circuit-stress, car-degradation and weather factors still apply through
the shared lap model. The separate minimum lap-time bound also remains active.
Actual racing, passing comparisons and all strategy planners consume that same
pace calculation. Finite pools retain each physical set's prior and race wear.

`grip_at_lap()` stays between `min(initial_grip, 0.5)` and `initial_grip`. It is
used as a grip indicator by the existing overcut heuristic; it no longer limits the
pace penalty. Custom low-initial-grip sets cannot gain grip or receive a fresh
wear bonus, but their configured degradation can still accumulate.

For a 90-second reference lap and management 0.8, before circuit/car/weather
scaling, the following are **model wear penalties**, in seconds:

| Compound | Age 20 | Age 40 | Age 60 | Former capped maximum |
|---|---:|---:|---:|---:|
| Soft | 1.620 | 7.290 | 12.960 | 1.485 |
| Medium | 0.972 | 2.916 | 5.832 | 1.350 |
| Hard | 0.518 | 1.037 | 2.138 | 1.215 |
| Intermediate | 1.296 | 2.916 | 5.508 | 1.080 |
| Wet | 0.972 | 1.944 | 3.888 | 0.945 |

The rates, thresholds, multipliers and time conversion are the existing model
presets, not newly fitted measurements. Pirelli describes a tradeoff between
peak performance and durability across its
[compound range](https://www.pirelli.com/tyres/en-ww/motorsport/car/formula-1).
That supports the qualitative distinction; it does not establish these
numerical curves or a real tyre's safe lifetime. Temperature history, graining,
damage-dependent pace and age-dependent puncture risk are separate modeling
questions. This change does not calibrate them from reported stint lengths.

## Reproducible strategy evidence

```powershell
python examples/check_tyre_wear.py
```

The diagnostic executes eight synthetic six-lap cases: both engines, finite
and unlimited replacement pools, a soft opening set aged 18 laps in dry weather
and a wet set aged 38 laps in steady rain. It uses a 180-second reference lap,
8-second pit lane, expected service, mean lap pace and no incidents. Those
deliberate inputs put a short remaining race across the configured wear cliffs.

It independently executes every legal schedule with zero to two paid stops
after the opening lap: 90 schedules per dry case and 16 per wet case. Finite
schedules allow removed physical sets to return with their accumulated wear.
JSON records inputs, wear samples, selected and best executed results, and
their time difference. The selected policies currently match the best checked
schedule in all eight cases. The enumeration is bounded; it does not prove
global optimality for traffic, future weather changes or arbitrary races.

Running the same diagnostic against revision `3e684c8` selected no stop in the
wet case. The corrected model selects a replacement on lap two. Executing
both schedules under the corrected physics gives approximately 1358.784 seconds
for staying out and 1346.260 seconds for the replacement: a 12.524-second model
gain, with the same completed distance. Both engines and inventory modes agree.
The dry choice stays at lap two. These are internal model comparisons, not
predicted real-race gains or a target stop count for calibration.

Saved inputs replay using the installed simulator implementation. Runs made
with the former capped curve can therefore produce different races after this
model correction; retain the recorded code revision for historical reproduction.

Finite-pool planning can take longer under the corrected curve because more
replacement schedules remain competitive during exact search. Reusing the cost
of identical final stints within each decision reduces repeated work, but does
not remove this increase. Long forecasts and automatic opening selection still
need particular care when sizing a Monte Carlo run; start with a small trial
count as described in the [finite-pool guide](tyre-inventory.md).
