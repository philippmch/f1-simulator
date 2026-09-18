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

## Temperature and warm-up

The current model prices accumulated wear and weather mismatch, but a newly
fitted tyre receives its configured grip immediately. The `optimal_temp_range`
field is descriptive; lap physics does not track tyre temperature or charge
for bringing a replacement set into its operating window. Formation laps,
thermal recovery after neutralization, pressure and heat cycles are also
outside this model. Track temperature is not a measurement of tyre temperature.

This is a material limitation for comparing undercuts and overcuts. Pirelli
describes fast warm-up as a characteristic of its
[C4 compound](https://www.pirelli.com/tires/en-us/motorsport/car/formula-1).
Its [2026 Canadian weekend comments](https://www.formula1.com/en/latest/article/what-the-teams-said-sprint-day-and-qualifying-in-canada.5K36jKNTRyzhQ6HTirVMgw)
also describe the effect of low ambient temperatures, modest lateral loads
and tyre blankets on early-lap heating. These observations support including
thermal behaviour in future calibration; they do not supply a universal
seconds-per-outlap penalty for the simulator's relative soft/medium/hard sets.

A useful calibration must separate the time needed to heat a tyre from pit-lane
loss, fuel burn, traffic, energy deployment and changing conditions. OpenF1's
[lap fields](https://openf1.org/docs/#laps) identify pit-out laps and their full
duration, with approximate starting timestamps; those times alone do not isolate
warm-up. A 2026-09-12 evidence refresh returned HTTP 401 for both the season
session request and a request restricted to completed races. No new timing
observations or thermal coefficients were obtained from that attempt. This
records access at that checkpoint, not a claim that all historical data requires
authentication.

The official [Formula 1 timing archive](https://livetiming.formula1.com/static/2026/Index.json)
provides an alternative source. An explicit diagnostic reads one completed race
from the current UTC season, keeps the source feeds in memory, and prints JSON:

```powershell
python examples/check_tyre_evidence.py --meeting Canadian
python examples/check_tyre_evidence.py --meeting Italian --include-laps
```

The report associates lap numbers and durations only when the timing feed
reports them together. A missing time remains missing; a previous lap's duration
is never carried forward. Tyre-stint metadata identifies reported compounds and
prior wear, while pit, track-status and rainfall observations identify excluded
laps. Missing required timing, compound, pit, control or rainfall context also
excludes a lap. Optional metadata such as prior wear remains explicitly unknown
when absent. The sparse lap-time fields inside
the tyre-stint feed are not treated as a complete lap history.

The remaining observations are candidates for further analysis, not isolated
thermal measurements. They still contain fuel burn, traffic, energy deployment,
driver variation and possible timing corrections. Feed timestamps describe
reported events rather than precise tyre-fitting or temperature measurements.
The report fits no warm-up or wear coefficients and does not alter race inputs
or tyre presets. Unavailable, unfinished or mismatched archives produce an error
rather than falling back to another season.

On 2026-09-18, the diagnostic found the following coverage. Candidate counts
apply the observational exclusions above and are not calibration sample sizes:

| Race | Reported lap crossings | With a paired duration | Candidate laps |
|---|---:|---:|---:|
| Canada | 1,206 | 1,185 | 934 |
| Italy | 1,052 | 1,029 | 908 |
| Spain | 1,105 | 1,081 | 978 |

The JSON records the session identity and decoded-feed hashes so a later run
can distinguish changed source data from changed normalization.

A follow-up comparison on the same date required six consecutive candidate laps
in the new stint after each reported stint transition. The first following lap,
minus the median of following laps three through six, varied substantially by
race: Canada's median was +1.843 seconds on mediums (13 windows) and +2.211 on
softs (7); Italy's was +0.244 on mediums (6); Spain's was -0.317 on hards (13),
-0.853 on softs (4), and -0.374 on mediums (6). These differences mix early-stint
behaviour with subsequent wear, fuel burn, traffic and changing pace. They do
not measure the initial pit-out warm-up loss.

Requiring green, rain-free transition context and eligible preceding laps two
and three laps before the transition left only one window each in Canada and
Italy, and twelve in Spain. Extrapolating a linear trend from following laps
three through six still produced widely varying early-lap residuals. This
small, confounded sample does not identify a reliable compound-specific warm-up
coefficient. The default physics therefore remains unchanged; adding a fixed
outlap penalty from these comparisons would imply calibration the evidence
does not support.

Physical wear and thermal state must remain separate if this model is extended:
a reused set retains its accumulated wear even when it needs to heat again.
The same thermal response would need to be applied in actual laps, opening and
pit forecasts, qualifying assumptions and replay, rather than adding a cost to
only one strategy path. The current controlled execution diagnostics establish
consistency of the existing physics and do not resolve this calibration gap.

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
replacement schedules remain competitive during exact search. Local reuse of
identical final stints and a bound that respects physical set ages reduce this
work; see the [search and benchmark notes](tyre-inventory.md#search-and-benchmark).
Long forecasts and automatic opening selection still need particular care when
sizing a Monte Carlo run; start with a small trial count.
