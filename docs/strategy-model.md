# Strategy model and limits

The simulator combines dry-race, rain-stint and mixed-weather cost optimization.
It compares modeled remaining tyre
and pit costs within the allowed stop budget, including required corrections.
Rain-stint plans also consider compound transitions under persistent rainfall
and the modeled surface response. These are conditional model calculations,
not claims to find the fastest strategy for a real race.

Unless a section says otherwise, the fresh-set projections below describe
drivers with unlimited replacement sets. An explicit [finite race pool](tyre-inventory.md)
uses a separate physical-set planner that retains set IDs and wear, constrains
opening choices and replacements, and records an auditable fitting history.

All team styles can evaluate up to three paid stops in dry running, using fewer
when further tyre gains do not cover pit loss. Previous paid stops, including
weather stops, count toward this budget; a required compound-correction stop
remains available after it is exhausted. Three is a bounded search limit, not
a requirement to make three stops or a claim that longer races never need more.
Dry red-flag projections use the same budget.

Dry planning scales tyre costs with the same current weather multiplier as
simulated laps, including surface water, rainfall and the car's wet-performance
contribution. It holds that multiplier constant over the projected stint;
this is not a forecast of future weather. Expected service and pit-lane loss
are not weather-scaled. A current SC/VSC running multiplier applies in addition
to weather scaling for this lap only, with later laps assumed green.

Before a timed finish is announced, in-race strategy estimates its distance
from the leader's elapsed time and most recent running pace. Pit service,
incident loss and time spent waiting behind another car are excluded from the
recurring pace estimate. Current control affects the upcoming lap, with later
laps assumed green. The estimate includes the lap following clock expiry and
never exceeds the scheduled distance. Without a pace observation it retains
that schedule. Both engines use this estimate for pit decisions and free tyre
choices; chronological races also map the leader's estimated finish time to
each car's own remaining laps and include completed suspension extensions.

Chronological forecasts resolve equal completed distances using physical
on-track order, ahead of cars still in the pits, rather than stale crossing
times. A pitting car a full lap ahead retains distance priority. A pending stop
uses its expected exit when planning the remaining race; the actual finish
controller continues to use executed crossings.

Rain and weather-stop projections use this shared leading clock to estimate
surface conditions at each car's future lap starts. A slow car can see multiple
surface updates between its own laps; a faster car can see the same surface
twice. The forecast includes no update at the projected chequered crossing.
It uses observed free pace and expected unfinished leader service, with current
control on the upcoming lap and green running thereafter. Without usable pace
observations it retains one update per own lap. Rainfall and condition remain
fixed, and no future weather draws are consumed.

When another car supplies the chronological forecast's leading clock, rain,
weather-stop and finite-pool costs also account for the candidate's own planned
paid stops. Retaining a tyre starts running immediately; fitting a replacement
starts after expected lane, service and any observed queue delay. Each later
paid stop adds its expected green-running pit loss to subsequent projected
starts. Traffic-related cost adjustments do not advance this physical clock.
The compound choice uses conditions before service, while its running cost uses
the projected surface at rejoin; the execution still fits just once.

The external leading clock remains fixed under these comparisons. A candidate
that is itself the projected leader, including a single-car race, retains the
own-lap projection so its stationary pit time cannot invent weather updates.
The forecasts do not resolve future changes of leader, tyre-dependent changes
in free pace, battles or interventions. They retain the current planning
distance and are recalculated at the next decision; stop timing can still change
actual finishing distance. Matching the shared weather cadence does not
establish an optimal complete strategy or predict real weather.

The forecast is recalculated as the race develops. It does not change the
actual finish controller, predict future stops or weather, or guarantee a
globally optimal timed strategy. After a timed announcement, the next leading
crossing remains authoritative even if a lapped driver inherits the lead.
Original scheduled fuel distance remains separate from all strategy horizons.

Before committing an elective stop, the chronological engine also checks whether
the stop would sacrifice a completed lap under the projected leading finish.
It compares a mean-pace continuation on the fitted tyre with an optimistic stop
continuation: expected lane, service and queue time, a mean outlap, then later
laps at the lap model's minimum time without further pit or traffic losses.
Both paths end at their first crossing at or after the same projected flag, or
at the original scheduled distance. A stop is cancelled only when retaining the
tyre remains feasible and completes more laps than this optimistic stop path.
Equal-distance decisions retain the ordinary strategy planner's choice.

This check uses persistent rainfall and the shared leading weather clock at
projected track-entry times, including the expected pit exit. It consumes no
future weather or service draws and does not fit or reserve physical tyre sets.
Forced repairs, critical tyre mismatch, unavailable fitted sets and unresolved
compound-use requirements remain under the existing compulsory-stop rules.
The check also remains inactive under race control, for the projected leader,
or without a usable forecast. It protects distance under these conditional
mean-pace assumptions; it does not guarantee the sampled race outcome or solve
the complete timed strategy problem.

Pit-rejoin traffic uses the same projected finish time, even before the timed
finish is announced. Each rival remains traffic until its own projected final
crossing; lapped cars can therefore remain after the leader takes the flag.
Exact crossing/exit ties retain the scheduler's distance and ordering rules.
This forecast retains current pace/control assumptions and does not predict
later weather, incidents or elective stops.

Explicit starting-tyre overrides are available per driver in the dashboard,
CLI, API and Monte Carlo runner. They replace only the opening choice; unlisted
drivers retain automatic selection and all later pit decisions remain active.
An unsuitable starting set can therefore be replaced before lap one. Overrides
do not change qualifying, grant a compound-use exemption, or specify a fixed
pit schedule. They are included in saved-input replay. Unknown driver codes and
invalid compounds are rejected instead of silently falling back to automatic.

Opening sets can include prior wear: `VER=soft@5` in the CLI or dashboard,
or `starting_tires={"VER": "soft"}, starting_tire_ages={"VER": 5}` in Python
and the corresponding JSON objects in the API. An age requires an explicit
compound for that driver and must be an integer from 0 through 1000; omitted
ages mean fresh sets. The upper bound limits inputs, not realistic tyre life.
Prior laps enter the existing wear and strategy calculations but are separate
from laps driven in this race. They do not add distance, satisfy dry-compound
use or grant a wet-tyre exemption. A used opening set replaced before running
does not create a fictitious stint. Without an explicit finite pool, paid stops
and free red-flag changes fit fresh sets and reset the prior-wear offset.
With a [finite race pool](tyre-inventory.md), fittings select actual available
sets and preserve their wear, including reuse. Qualifying remains independent;
heat cycles and real-world tyre allocations are not inferred.

The optional `fixed_rainfall` weather mode helps isolate strategy behavior under
unchanging rainfall. It prevents random condition transitions while retaining
surface-water response, incidents and adaptive pit decisions. The usual evolving
mode remains the default. Both modes are scenario assumptions, not forecasts;
saved-input tyre comparisons retain whichever weather behavior was exported.

Automatic starting-compound selection on a clearly dry track compares isolated
runs of the existing pit policy for each slick, using noise-free lap pace and
expected service time. These runs follow the race clock, retain the original
fuel distance, and include later paid stops and the distinct-compound rule.
They compare completed distance first and elapsed time second. This avoids
choosing an opening tyre for a scheduled distance that the time limit will cut
short, or rewarding a slower policy merely because it completes fewer laps.
Existing strategy weights choose among equivalent best outcomes, with a
one-billionth-of-a-second tolerance for elapsed-time ties. One-lap races and
cases without a legal projected finish retain the original weighted choice.
Explicit starting-tyre overrides and rain sets selected by the surface/rainfall
crossover retain priority.

The dashboard's Statistics tab and saved HTML reports show recorded tyre
sequence frequencies for each driver across the full run. Expand a driver to
see every sequence, its count and share, and the finished and retired counts.
JSON statistics, scenario comparisons and dashboard downloads retain the same
`strategy_statistics` data. Shares use that driver's observed races with a
recorded sequence, with missing records counted separately; the requested
simulation count is not used as a substitute for observations.

Sequences preserve fitting order and repeated compounds, including free
red-flag changes. A retirement can truncate a sequence, and a fitted set may
not complete a lap. Sequence length is therefore not the paid-stop count.
These are descriptive frequencies from the simulated policy, not controlled
comparisons of which strategy is fastest.

For a fixed-input comparison, `examples/compare_starting_tyres.py` runs one
driver's requested opening choices from an exported simulation snapshot. Each
variant preserves the roster, cars, track, weather, race engine, other drivers'
overrides, including their ages, and base seed range. Labels such as `soft@5`
and `soft` compare a used set with a fresh one. `automatic` clears only the
target driver's saved compound and age overrides. The trial count can differ
from the source run. All later strategy
and weather responses remain enabled, including immediate replacement of an
unsuitable opening set.

The comparison reports outcome rates and points per race, with a 95% Wilson
interval for each win rate. These intervals measure Monte Carlo sampling
uncertainty within the model. Matching seeds preserve qualifying, but changes
in race decisions can consume random draws differently, so incidents and later
events are not held fixed between variants. New runs use an independent weather
stream, so strategy choices cannot alter rainfall simply by consuming different
race draws. With matching seeds and weather inputs, the recorded weather
histories share the same prefix by weather-update interval. They are not aligned
by elapsed seconds, and retirements or timed finishes can shorten the history.
This is a comparison of opening
policies under the saved model, not a ranking of complete pit schedules or a
claim about real-race optimality. Exported variants contain their own inputs
and can be replayed using the installed simulator implementation.

The saved `rng_policy` makes this behavior explicit. `isolated_weather_v1`
retains the usual seeded qualifying/race generator and derives a separate
weather generator from `SeedSequence(seed, spawn_key=(0x57454154,))`. That fixed
namespace separates weather from race draws without advancing the race stream.
`shared_v1` retains the former single generator. New snapshots use schema 2,
schema 3 when opening ages are specified, or schema 4 for nonempty finite race
pools, and require an explicit policy. Schema 4 records the initial inventory.
Schema 3 also requires the age mapping, so older installations reject used-set
runs instead of replaying them fresh. Replay and comparison still accept
schemas 1 and 2 with fresh opening sets; schema 1 infers `shared_v1` when its
policy field is absent. Unknown policies are rejected.
The opt-in `isolated_weather_mechanical_v1` policy retains that weather namespace
and derives mechanical draws separately for each trial seed, stable driver ID
and own lap. The driver ID is UTF-8 encoded and SHA-256 hashed; the full digest
is split into eight little-endian unsigned 32-bit words. The mechanical
generator uses `SeedSequence(seed, spawn_key=(0x4d454348, *driver_words, lap))`.
This versioned layout does not depend on Python's process-specific hash seed.
Unrelated race draws and driver processing order cannot shift those draws.
The hazard check and, when needed, component selection use the derived
generator. The hazard formula and component weights are unchanged. Only laps
actually driven receive checks; changes in heat, risk inputs or exposure can
still change failures. Other race processes continue to share the race stream,
so this does not isolate all causes of a strategy's outcome or guarantee a lower
sampling error. Existing policies and the default remain unchanged.

Comparisons inherit the saved policy unless `--rng-policy`,
`--independent-weather` (the convenience option for `isolated_weather_v1`), or
the Python `rng_policy` override is supplied. The two CLI options are mutually
exclusive. An override changes all variants together and records the policy
for replay. Direct `RaceSimulator` callers retain shared mechanical and weather
draws unless they supply the corresponding generator hooks.

Combined JSON exports retain each scenario's observed driver counts, rate
intervals, points per observed race and paid-stop statistics. The offline HTML
comparison report places scenarios in their supplied order, with expandable
driver tables showing win, podium and retirement intervals. It does not rank
choices or treat an interval for one rate as an interval for a difference.
Absent drivers and zero observed trials display as not recorded; paid-stop
averages use their own recorded-race counts and exclude free tyre changes.
The dashboard's Scenarios tab can download this report for the completed run.
Its weather context includes the condition label and per-lap change probability,
since scenarios with identical initial rain and wetness can still evolve
differently. Their current pace is identical: [numeric water and rainfall](weather-pace.md),
rather than the condition label, determine the weather pace factor. These
settings are not a weather forecast.
The dashboard scenario chart and matrix also display the backend's Wilson
intervals and observed trial counts. Driver groups in the chart can be expanded
independently. Matrix highlights refer only to point estimates. Missing rates
are excluded from sorting averages and remain blank in matrix CSV exports;
older results without interval metadata retain their estimates with an explicit
sampling-range-unavailable label.

Saved-input comparison commands additionally compare each variant against a
reference (the first selected label, or `--reference`). They pair overlapping
recorded effective seeds, `base_seed + trial_index`, only when saved driver,
car, track, weather and runtime inputs and the random-stream policy match.
Model snapshots must also pass the strict JSON type checks used for replay;
booleans and quoted numbers cannot stand in for numeric model fields. Invalid
snapshots make paired statistics unavailable instead of supplying matching evidence.
Starting-tyre overrides and race engine may differ. Each retained trial must
have identical qualifying records. A driver needs exactly one valid final row
in both runs; missing, duplicate and malformed records exclude that driver's
whole pair. Drivers without a matching saved car cannot supply paired observations.
Reported means and retirement rates use only this paired subset.
The source run's overall averages can differ when observations were excluded.
Saved opening-tyre comparisons validate all requested variants before running
any trials, including exact compound-and-age availability in finite set pools.
An invalid later choice therefore cannot consume earlier variants' simulation work.
Console and HTML comparisons show the overlapping seed range and the number
of whole pairs excluded by missing, invalid or different qualifying records.
Each driver's excluded count includes those qualifying exclusions; additional
exclusions reflect missing or invalid final observations for that driver.

For points differences `d_i = variant_points_i - reference_points_i`, the mean
is `sum(d_i) / n` and its estimated standard error is
`sqrt(sum((d_i - mean)^2) / (n - 1)) / sqrt(n)`, following the
[paired-observation statistics described by NIST](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm).
SE is absent for fewer than two pairs. Identical observed differences give
zero estimated SE, which is not evidence that future differences are fixed.
This is a descriptive sampling-error estimate, not a confidence interval or
an optimal-strategy ranking. Points use explicit distance-adjusted awards when
recorded, including classified retirements; legacy rows use classification-based
scoring. Operational DNFs are counted separately. Positive retirement-rate
changes mean more retirements and are expressed in percentage points.
Joint retirement counts distinguish pairs where both finish, both retire, only
the reference retires, or only the variant retires. They sum to the driver's
usable paired count and include classified retirements as DNFs. Matching
retirement rates can otherwise hide completely different trial outcomes.

For the retirement-rate difference, each paired observation is
`d_i = variant_dnf_i - reference_dnf_i`, with values -1, 0 or 1. Its SE is
`100 * stdev(d_i) / sqrt(n)` in percentage points, using the sample standard
deviation and the same usable pairs as the reported rate difference. It is
absent below two pairs. This retains the observed association between paired
outcomes; treating the two marginal retirement rates as independent would lose
that information. Neither the joint counts nor this SE identify retirement
causes or establish a strategy's causal effect. The same sampling and
zero-variation limitations as the points SE apply.
These calculations do not consume simulation randomness or rerun races.

Paired comparisons also report completed-distance changes, since equal points
and retirement outcomes can hide a lost lap. Each driver's `completed_distance`
summary uses only otherwise valid pairs with an integer `laps_completed` in
both results, from zero through the saved scheduled distance. Unknown legacy
distances and invalid values are excluded from this distance subset without
discarding their valid points or retirement observations. Its paired and
excluded counts therefore describe a separate denominator.

The distance summary includes both mean lap counts, the mean variant-minus-reference
change, its sample standard error, and counts of more, equal and fewer completed
laps. A known zero-lap retirement is an observation; missing distance is not zero.
The standard error is absent below two distance pairs. Finishes and retirements
both contribute their recorded distance, so these differences can reflect
lapping, time limits or retirement exposure. They do not isolate a strategy's
causal effect, measure pace at equal distance, or rank complete strategies.
Console and HTML reports display the distance subset alongside points and DNF
comparisons. No race is rerun to supply missing distance.

JSON `paired_comparisons` and the HTML paired tables are opt-in through the
exporter's `reference_scenario`; both saved-comparison commands supply it.
General weather-scenario exports remain unpaired. Summary labels retain supplied
order, and unavailable comparisons state which pairing prerequisite is missing.

With no rainfall, the dry policy projection is deterministic and needs one
private run per slick. With sustained rainfall that could change the surface,
it averages the same eight private reaction seeds used below. Cached scores
include physical inputs, tyre configuration, strategy settings and the race
time limit; driver and team names do not affect them. This compares the
existing single-car policy, not every possible timed pit schedule. It assumes
constant weather condition and rainfall, no traffic or future interruptions,
and unlimited tyre inventory for drivers without an explicit pool. A multi-car
race can have a different horizon as its leader and traffic determine the finish.
Precautionary intermediates must also pass the same mismatch check used during
the race. A rainy condition label with a sufficiently dry surface and low
rainfall does not fit intermediates that would immediately require a paid
replacement before lap one. Explicit starting-tyre overrides remain available.

When intermediates are only a precaution, the selector compares them with all
three slick compounds using isolated, traffic-free runs of the existing pit
policy. These runs cover the full race, use mean lap pace and expected service
time, and advance surface wetness after each lap under constant rainfall and
weather condition. They include later paid stops and the compound-use rule.
Each candidate uses the same eight fixed private reaction seeds; the selector
compares their average completed distance first and average total time second,
so a slower, shorter time-limited race cannot win merely by ending sooner.
It favours the existing intermediate choice in a tie. The real race's random generator and input objects are untouched.

This is an approximate comparison of the current policy under sustained
conditions, not a global wet-strategy optimizer or a forecast of changing rain,
traffic, or incidents. Close choices can depend on the sampled reaction paths.
Results are cached with a bounded capacity, including the driver, car, circuit,
weather, tyre configuration, strategy settings and race time limit. Dry,
precautionary and finite-pool scores share the same input normalization: driver
and team names and previous race state do not create new physics. Changing the
deadline recomputes the completed-distance ranking before a new opening is
selected. Direct selector calls
without driver/car context retain the original precautionary choice.

Run `python examples/check_opening_policy_execution.py` to compare the isolated
opening-policy path with actual execution. The default diagnostic runs seven
synthetic dry, drying, wetting and timed cases, all five opening compounds,
two private reaction seeds (0 and 3), both engines and both inventory modes:
280 comparisons. The finite pool has one set of every compound, with soft
already five laps old. Rainfall stays fixed while the surface evolves; mean
pace, expected service and green running remove incidents and traffic.

The JSON report includes physical inputs, forecast and executed distance/time,
actual compound and stop histories, and finite-set wear ledgers. It checks the
original scheduled fuel distance even when racing finishes early, and accounts
for all completed laps in physical wear. Infeasible forecast costs are explicit
and use JSON null rather than Infinity; invalid numeric results are errors.
The command exits with status 1 for a mismatch and 0 for a successful comparison.
`--engine standard|chronological|both` and `--inventory unlimited|finite|both`
select narrower runs. It makes no network requests or default file writes.

The 2026-09-12 checkpoint matched all 280 comparisons, including 120 timed
finishes. This tests execution of the existing policy, not every alternative
pit schedule, all eight opening reaction seeds, or performance in a stochastic
multi-car race. It does not calibrate tyre behaviour; in particular,
[warm-up remains unmodeled](tyre-wear.md#temperature-and-warm-up).

Aggressive, balanced and conservative profiles influence opening slick choices
and close dry timing decisions. Damp strategy retains the style-based ordinary
stop allowance while comparing projected running and stop costs. Legacy fallback
decisions also respond to traffic, track position, circuit overtaking difficulty
and surface water. Pit loss includes the circuit's pit-lane
delta and sampled stationary service time; safety-car and virtual-safety-car
running reduce the relative pit-lane loss.

Weather-driven stops take precedence over normal stop budgets. This allows a
driver to react when conditions change after the planned stops are exhausted.
The dry-compound check counts distinct slick compounds rather than stops; using
an intermediate or wet compound exempts that driver's modeled dry-use rule.

A fitted set earns compound-use credit when it runs on track in a simulated
lap. Replacing it before any running does not count an extra compound or grant
a wet-tyre exemption; unused fittings are removed from stint history. When
considering staying out, the planner credits the current set's forthcoming lap.
An immediate replacement must instead use only compounds that have already
run. This also prevents a free red-flag fitting from granting credit if it is
replaced again before the restart lap. A distinct free set that can complete
the requirement by running does not force another paid stop or extend the
elective stop budget. In a two-lap race, the mandatory change waits until lap
two so the opening set gets a lap of running.
The unconditional compound-correction safeguard acts on the final lap, since
the replacement runs that lap after its stop. Earlier dry stops follow the cost
comparison, allowing a faster current set to remain on until the last legal
change when that minimizes total time.
One-lap simulations retain a single starting stint because this lap-level
model cannot run two sets within one lap.

The dashboard Statistics view summarizes paid pit stops across all observed
races for each driver: the average and the shares with zero, one, two, or at
least three stops. Retirements remain in these observations, so early failures
can lower the average. Free red-flag changes are excluded. The API and statistics
JSON export include the observed race count, mean, and exact stop-count
frequencies; drivers without race observations have no inferred stop statistics.

Race results expose the actual paid pit laps, shown below each dashboard stop
count and included as `pit_laps` in the API and downloaded scenario JSON.
The race CSV appends a `pit_laps` column containing a JSON array, such as
`[17, 34]`. An empty array means no paid stops; legacy results with unavailable
timing use null in JSON and a blank CSV cell. Free red-flag tyre changes remain
in compound history but do not add a paid pit lap. A stop performed before a
retirement on the same lap remains part of that driver's stop history.

Each paid stop also records `pit_stop_details`: the driver's own lap number,
outgoing and incoming compounds, completed laps on the outgoing set, condition,
rain intensity, surface wetness, race control, and lane/service/queue loss in
seconds. The three cost components sum to the modeled total; they are not exact
arrival timestamps and exclude subsequent on-track traffic. Recording does not
change race decisions or consume random draws. Free red-flag fittings have no
paid-stop record; an opening paid correction has tyre age zero.

The dashboard shows these observations for its selected trial. Statistics and
comparison JSON retain per-trial, per-driver records; the bundle's pit-stops CSV
has one row per paid stop with one-based trial numbers. Legacy JSON uses null for
unknown details and an empty list for a known zero stops; neither produces CSV
stop rows.

New paid-stop records also retain `decision_reason`, captured when the policy
accepts the stop: forced tyre replacement, critical weather mismatch, a weather
reaction, the compound-use requirement, a dry/rain/finite-set forecast, or a
neutralization/planned window. A custom policy that supplies no context leaves
the reason unknown; historical reasons are not reconstructed from tyre choices
or conditions. The dashboard shows this context beside each selected-trial stop,
and JSON and pit-stops CSV preserve it for every trial.

For a forecast decision, `forecast_saving_seconds` is the modeled cost of waiting
minus the cost of stopping, including the planner's remaining strategy and current
Overtake Mode adjustment. It is not a measured gain or a causal comparison of race
results. A slightly negative value can be accepted by the strategy's timing bias.
Compulsory/reactive decisions and nonfinite comparisons have no reported saving.
Missing context is null in JSON, blank in CSV, and “Not recorded” in the dashboard.
Free refits and vetoed pit proposals produce no paid-stop decision record.

The dashboard's Statistics tab and comparison HTML reports also summarize
recorded paid-stop decisions for each driver across trials. Statistics,
scenario JSON exports and dashboard downloads retain the same
`pit_decision_statistics` summary. Reasons describe the policy that accepted a
stop; they do not measure whether the stop improved the result. Unknown reasons
and incomplete records remain visible as missing evidence, including in older
exports. Retirements are included, while free tyre changes are excluded.
Reason shares use stops with recognized reasons, not requested simulations or
all paid stops. The report shows the number of races with complete detail lists
and the missing-reason count alongside these shares. A known zero-stop race
contributes to race coverage but never to the reason-share denominator.
Forecast savings are not summed or presented as realized race-time gains.

Aggregate `pit_loss_statistics` reports each driver's mean total, lane, service
and queue loss per race with complete details, plus queued-stop counts and the
share of those races containing a queue delay. The denominator includes known
zero-stop races and retirements. Unknown, incomplete or inconsistent detail rows
are excluded entirely and counted separately. With no complete observations,
means and queue share are null. Dashboard and comparison reports retain this
observation count; these descriptive costs do not establish a strategy's causal
effect on race results and exclude later on-track traffic and free fittings.

Fresh weather tyre selection shares the slick-mismatch crossover: above 0.2
surface wetness or 0.4 rain intensity, a stop fits intermediates; above 0.7
surface wetness, it fits full wets. These values are normalized model parameters,
not measured millimetres of water. Automatic openings use these thresholds;
free red-flag refits compare usable sets over the remaining projected weather
as described below. Below those fresh-selection thresholds, automatic openings compare
intermediates with slick policies whenever intermediates are not critically
mismatched (surface water at least 0.08 or rainfall at least 0.15), using numeric
conditions consistently across labels. Without driver/car projection context,
the fallback retains the precautionary intermediate choice.
Already-fitted rain tyres have wider drying windows before they trigger another
stop, which avoids repeatedly switching sets near the crossover.

Qualifying compares all fresh compounds using its one-lap model with random
variation and mistakes disabled, then runs normal sampled attempts on the
fastest set. Equal projected times preserve compound enumeration order. This
avoids paying a rain-tyre penalty on a still-dry surface solely because rainfall
has begun. Race stops retain their precautionary rainfall thresholds because
the next racing laps evolve surface wetness. Qualifying's lap model applies
the same driver/car weather multiplier and additive
tyre-weather mismatch penalty as race laps, while retaining qualifying's own
base pace, fresh-tyre grip and push-level variation. A driver with stronger wet
skill can therefore improve a wet qualifying lap, and slicks no longer escape
their mismatch penalty on a wet surface. Each attempt assumes a fresh set;
weather remains fixed across Q1, Q2 and Q3. Qualifying does not model tyre
inventory, track evolution, traffic or a changing-weather session strategy.

During a red-flag suspension, tyre selection compares usable fresh sets over
the remaining race, including any later paid stops that the stop budget permits.
Clearly dry projections retain the ordinary dry planner. Otherwise, each
currently noncritical compound is evaluated through the same fixed-rainfall,
evolving-surface projection as ordinary transition planning. A free intermediate
can therefore avoid a later paid stop as the track wets; a usable slick can
avoid fitting intermediates just before the surface dries. Future paid fits
still obey the ordinary weather-selection rules and stop allowances. The free set
must run at least one lap before another stop; future service and pit-lane time
are priced as green running. A suspension after lap N leaves `total_laps - N`
racing laps, starting with lap N+1. Critically mismatched free candidates are excluded.
The race applies its usual between-lap weather update before selecting the free
set, so that choice uses the conditions in which racing resumes. This avoids
fitting a set for the completed lap's weather and then paying to replace it on
the restart. It adds no extra weather update during the modeled waiting period.
The chronological engine freezes every car's restart horizon and weather
clock before fitting or releasing the first car. This includes a car whose
paid service finished while the pit exit was closed; old service forecasts and
release order cannot change the free-tyre projection.

`python examples/check_weather_restart_choices.py` compares these choices with
separately executed alternatives in both engines, under increasing rain,
drying, steady damp, worsening rain, steady wet and dry restart conditions.
The synthetic runs use mean lap pace, expected service and no incidents after
the suspension. They test consistency with the model, not real-race gains or
the optimality of every possible future stop schedule.

The free change does not consume a paid stop or pit-plan slot. A new distinct
slick can satisfy the compound-use requirement; repeating a slick remains an
option when the projected schedule includes a legal later correction. The
existing mandatory-correction stop exception remains available if needed.
Fresh tyres resolve the modeled puncture's pending stop, while incident time
already lost stays on the clock. Retired cars are not serviced. A suspension
after the final lap adds no tyre stint or compound-use credit.

Changing wheels and tyres during a suspension is permitted by B5.14.4(a)(vii) of
the [FIA 2026 Sporting Regulations, Issue 08](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_08_-_2026-08-05_7.pdf).
Without a finite race pool, the model assumes a free fresh set is available.
Finite pools restrict this selection to actual reusable sets and can retain
the current set. The full suspension work procedure is not modeled; elapsed
waiting follows the shared restart described below.

### Red-flag suspension timing

The standard engine commits each completed crossing before collecting the
field. For a red flag after lap N, the restart clock is the latest surviving
car's lap-N crossing plus a fixed pause. Every surviving car starts lap N+1
from that common clock in the retained physical order. Earlier crossings,
incident losses and paid pit service remain unchanged; a retired car does not
delay collection. This replaces the previous instantaneous gap reset, which
could move a trailing car's cumulative time backwards.

The default pause is 600 seconds, the same model assumption used by the
chronological engine. Direct Python experiments can set
`RaceSimulator(red_flag_pause_seconds=0)` or another finite nonnegative duration.
This is not a prediction of incident clearance or weather recovery. The
simulator's `suspensions` trace records each signal clock, common restart clock
and surviving order, and resets for each race.

In a controlled three-lap race with constant 90- and 110-second cars and a red
flag after lap one, the original crossings remain 90 and 110 seconds. With the
default pause, both cars restart at 710 seconds and finish at 890 and 930
seconds. Their fastest laps remain 90 and 110 seconds.

Results expose `race_suspension_seconds`: the sum of completed race-wide
suspension intervals, including field collection and the restart pause. In
the example above it is 620 seconds, repeated as context on every driver row.
It is not each driver's stationary time: the trailing car still completes
its lap during collection, and a retired driver's last crossing may precede
the suspension. Subtracting it from `total_time` does not give driving pace.
The recorded duration is uncapped; only the finish-deadline extension is capped.

CSV exports leave unknown suspension values blank; JSON uses `null`. Native
engine results record zero when no suspension completes, including a red flag
at the actual finish. Older or manually constructed results remain unknown.
Scenario `suspension_statistics` reports `recorded_races`,
`races_with_recorded_suspension`, and `mean_completed_suspension_seconds`.
The mean includes known zero-duration races and counts each race once, using
only races whose driver rows all carry the same valid duration. Missing,
invalid or inconsistent rows do not enter that denominator. Reports display
the recorded count so incomplete historical data cannot imply zero stoppage.

Collection and pause extend the existing two-hour finish threshold, with at
most one hour of accumulated extension. An already announced final lap stays
latched. Restart tyre selection and subsequent pit planning use the common
resume clock and extended deadline; physical fuel still follows the original
scheduled distance. A red flag at the actual scheduled or timed finish records
the event but starts no suspension, fits no tyres and evolves no extra weather.

Forecasts distinguish the last completed crossing from the next lap's release
clock. If a long suspension exhausts the extension cap, merely resuming after
the deadline does not count as a completed lap or a final-lap announcement.
Both engines retain the crossing that triggers the announcement and the lap
that follows it, subject to the scheduled distance and any existing signal.

Waiting between completed laps consumes elapsed race time without consuming
tyre laps, fuel-model laps, pit service or random draws. The next lap's recorded
duration starts at release, so suspension waiting is not recurring running
pace. A retirement on that lap retains the preceding completed crossing,
including its original time and distance. Paid stops on the restart use the
shared release clocks for the ordinary team service queue.

The standard engine still resolves collection once per shared lap. It does
not model partially completed sectors, cars held at a closed pit exit during
collection, detailed restart formation, abandonment or results countback.
The chronological engine schedules those individual crossings and pit exits,
while retaining its own documented lap-resolution limitations.

A conservative driver outside the top six can switch to the balanced profile
after 40% of the scheduled race when following in modeled dirty air. The default
traffic window is below two seconds; `conservative_switch_gap` can narrow that
window and acts as a maximum gap, not a minimum. The switch persists for later
decisions and compound choices. A large gap to the car ahead does not trigger it.

In the reactive fallback strategy, a non-conservative driver can return to the
earlier-stop plan when following closely after the actual race midpoint, rather
than after a fixed lap number. Wet-condition fallback selection retains priority.
Neither traffic-based profile nor fallback-plan switches are triggered during
safety-car, virtual-safety-car, red-flag or restart laps: their compressed gaps
do not by themselves establish a reason for a lasting strategy change. These
are model heuristics, not forecasts of how long a driver will remain blocked.

For an ordinary stop on a clearly dry track, the optimizer compares pitting now
with driving at least one more lap on the current set. It searches remaining
stint lengths and eligible slick compounds through the finish, allowing unused
stops to be skipped. Terminal schedules must satisfy the distinct-compound rule.
An earlier stop may repeat a compound when enough laps and stops remain to run
a different one later. This matters when a current SC/VSC changes the best
order of stints; the requirement applies to the completed race, not each stop.
The projection uses current tyre age and the shared lap model's pace and wear,
including driver management, circuit stress and the car's degradation factor.
The selected compound is carried into the actual stop.

Projected service time includes the execution model's minimum service duration
and slow-stop probability. Current safety-car or virtual-safety-car discounts
reduce pit-lane loss. The current lap's tyre costs also use the race timing
model's active running multiplier, for both staying out and every eligible
fresh compound. Future running and stops are projected as green; this does not
predict how long a neutralization will last. A large gap behind
does not by itself make a stop worthwhile. From lap two onward, the dry optimizer
can choose an early stop, including during an early SC/VSC, when its projected
benefit justifies the cost. Elective dry stops before the first racing lap are
suppressed because stops occur at the start of a modeled lap; urgent weather
changes and mandatory safeguards retain priority. Mixed-weather slick decisions
also start on lap two and can act outside the former five-lap guards. Legacy
fallback heuristics retain those guards. Beneficial dry stops can also occur in the
final five laps. Team style can shift
a near tie by at most 0.1 seconds per decision. This tolerance is a model
assumption, not an empirical fit.

Teammates share one modeled pit box. For stops on the same lap, service follows
arrival order and a car waits only while that constructor's box is occupied.
The wait is added once to pit loss; safety-car discounts affect pit-lane loss,
not stationary service or queue time. Different constructors have independent
boxes. This represents the consecutive service described in F1's
[double-stack glossary entry](https://www.formula1.com/en/latest/article/f1-glossary-a-e.1MFONigMlQSbSQtpP7YCy2).

Lap-start race clocks approximate relative pit-box arrivals. Before committing
to a dry stop, a driver is charged the expected queue from an earlier-arriving
teammate already committed on that lap. Planning uses expected service; actual
execution uses sampled service, so the decision does not know a future slow-stop
outcome. In the standard engine, reservations reset each lap and race.
The chronological engine carries reservations across individual car lap starts
and resets them for each race; cars on different laps can share the box.
Its planning reservations and projections of pitting rivals use expected
service separately from the sampled execution queue. Once service has visibly
completed, a stale expected reservation cannot keep the box occupied. While a
service is visibly still in progress, the chronological forecast uses its
conditional expected remaining duration given the service time already
observed. It does not read the sampled future completion time. A queued stop
whose service has not started retains the stationary expected service.
Neither engine models pit-lane congestion, crew setup time or unsafe releases.

The recorded time for a stop lap includes the actual pit-lane, stationary and
queue losses, so its clean running pace alone cannot earn a fastest lap. Those
losses enter total race time only once. SC/VSC modifiers slow the running portion
of the lap, with full-SC catch-up bounded as described below; stationary service
and queue time are not multiplied by them. The
model attributes the entire stop loss to the lap on which service occurs, rather
than splitting pit entry and exit across sector timing lines.

After the pit batch, dirty-air pace uses one frozen view of the field with
actual pit losses applied. A car can emerge into traffic, and a following car
can gain clean air when the car ahead stops. Strategy decisions and Overtake
Mode detection retain their pre-stop information. Final position changes still
use the completed lap clocks, including actual service and on-track running.

During green running, the dry, rain-refit and rain-transition optimizers also
compare the first lap's dirty-air cost when stopping with that when staying out.
They retain the current gap and the projected rejoin gap after expected lane,
service and queue loss. The standard engine includes stops already selected
earlier in its frozen arrival-order batch, using expected lane, service and
team-box delay. It projects both staying out and joining those pitters through
the same physical-order merge used in execution. Undecided rivals are still
assumed to stay out. The chronological engine accounts for rivals already in
the pit lane through their expected exits. The shared lap model contributes
between zero and 0.5
seconds before weather scaling, depending on a gap below two seconds; this can
move a close timing decision in either direction. It does not predict a queue of slower cars over multiple laps,
passing opportunities, or rivals' stop decisions. The correction is disabled
under neutralisation. Expected gaps approximate a nonlinear cost at the mean
service time; they are not an average over every possible service outcome.

For example, when two tied cars from different teams both stop behind a
third car, equal expected service leaves the second pitter behind the first.
Its decision cannot claim that the full pit loss will become clean air merely
because the first pitter's original crossing clock was earlier. Conversely,
a car that stays out can gain clean air when its predecessor has already
committed to stop. Reservations use expected service only; actual service is
sampled after the batch's decisions. The observed pre-stop gaps still govern
reactive style changes. Projected traffic does not mutate race clocks, reserve
tyre sets, revisit earlier decisions or predict later rivals' choices.

Each retained or freshly fitted candidate receives its respective gap inside
the current lap calculation, before the minimum lap-time floor and current
control multiplier. A fast candidate whose clean and dirty laps both hit the
floor therefore receives no fictitious traffic penalty or escape benefit.
At the clipping boundary, different compounds can receive different effective
traffic costs and are compared again before selecting the fitted set. Queue
delay remains an independent pit cost. Future laps retain the existing
green, clean-air assumptions and cached cost tables.

Native current-lap stay-out projections also account for an available Overtake Mode
burst. Eligibility uses the observed gap, remaining energy and the engine's
current race-control and weather conditions. The gain passes through the shared
lap physics, including its minimum time floor, rather than being subtracted as
an unconditional bonus. A paid-stop candidate receives no deployment benefit,
matching race execution. Future laps assume no deployment because future gaps
and energy use are unknown. Evaluating a strategy consumes neither energy nor
random draws; actual running still owns deployment and recharge.
The chronological finish-distance protection includes the same eligible first
retained lap. Its optimistic paid-stop bound and later laps keep their existing
assumptions. Direct standalone planner calls have no live energy or deployment
snapshot and retain their no-deployment baseline.

For direct Python calls, `current_traffic_gaps=(stay_gap, rejoin_gap)` supplies
these first-lap observations to the three planners; a gap of `None` means clear
air. Omitting the option preserves the existing clean-air calculation and
additive `additional_current_stop_cost` contract. Native engine snapshots
carry both gaps. Explicit older `StrategyTrafficSnapshot` instances containing
only a scalar rejoin cost retain their additive behavior. The remaining reactive
fallback uses the separate optimistic cost veto described below; it does not
claim to optimize the observed rejoin gap.

### Safety-car queues and elapsed time

Both engines close full-safety-car gaps through subsequent running. Deploying
an SC preserves the lap just completed, including incident penalties and paid
pit losses. A final-lap deployment therefore preserves the actual finishing
gap. VSC keeps the existing individual running-time multiplier without the
SC catch-up model.

In the standard engine, the frozen field after pit service determines the
queue. Its first car sets the nominal pace using the existing 1.4 SC multiplier.
Followers approach a one-second gap, bounded below by their own free-running
lap time. Each follower uses its predecessor's projected crossing, so recovery
propagates through several cars on the same lap. A car too slow to join the
queue keeps losing ground; it can also hold up cars behind it. Compact gaps
need not be widened, and physical order breaks equal-time ties. The one-second
target and lap-resolution pace bounds are modeling assumptions, shared with
the experimental engine, rather than a fitted SC speed profile.

For example, two 90-second cars starting an SC lap 80 seconds apart take four
laps to settle at the target gap. With the leader running 126-second laps,
the follower runs 90, 90, 119 and 126 seconds; the gap becomes 44, 8, 1 and
1 seconds. Every completed lap contributes its full recorded time. A pit stop
can create additional gap that is subsequently recovered on track, while its
lane, service and queue losses remain in the stop ledger and lap accounting.
The lap on which an SC countdown ends still uses its starting restrictions.

This does not make the standard loop chronological: it still advances every
survivor once per leading lap. Pit optimizers retain their nominal
current SC multiplier and do not predict the field's full catch-up sequence or
future SC duration. The experimental engine instead schedules individual
crossings and pit exits. The engines share catch-up bounds, but their physical
queue and pit-arrival approximations can produce different results.

Clean air's relevance to an undercut is described in Formula 1's
[pit-strategy analysis](https://www.formula1.com/en/latest/article/jolyon-palmers-analysis-singapore-and-the-art-of-undercutting.1NgVyVsZnHTDEA9wi0s5lW).
The simulator's two-second range and half-second maximum are existing model
parameters, not values calibrated from that source. Because pit service occurs
before lap pace in this engine, the post-stop gap represents the running lap;
there is no separate in-lap/out-lap sector simulation.

Passing probability also uses the tyre pace difference between the cars in
dry, damp and wet conditions. The shared lap model includes compound,
current tyre age, driver management, circuit stress and car degradation. Tyre
pace receives the same driver/car weather multiplier as actual lap running,
plus the flat penalty for a compound that mismatches surface conditions. This
lets suitable rain tyres help an attack against slicks on a wet surface, and
lets worn or overheating rain tyres weaken an attack. Because
the maneuver is resolved after running the lap, this comparison uses the
current end-of-lap tyre ages. A fresh set fitted for that lap has age one.

The defender's tyre cost minus the attacker's cost is converted to a base-pace
equivalent using the car model's 3%-of-reference-lap scale, then bounded to one
pace unit in either direction before entering the existing passing curve.
Equal tyre contributions preserve the old probability. This is a bounded
heuristic, not a fitted relationship between lap-time advantage and passing
success. Circuit difficulty, proximity, driver skill and Overtake Mode still
affect the maneuver, and a tyre advantage cannot bypass the gap restriction.
Green-running opportunities use this probability curve on every circuit;
there is no separate difficulty cutoff that disables attempts. At the maximum
difficulty of one, the existing unboosted success probability is zero, while
contact and mode/restart effects still follow their normal rules. These are
model limits rather than calibrated circuit-specific passing rates.
The wet passing difficulty multiplier and wet Overtake Mode restriction remain
in effect. This extends the existing lap-time heuristic to rain tyres; it does
not model aquaplaning, a racing line that dries separately, or measured wet-grip
passing probabilities.

Restart passing uses a two-second attempt window, compared with 1.5 seconds
in normal running. The proximity factor decreases across the corresponding
window, so an eligible restart attempt between 1.5 and two seconds can succeed.
The outer gate still rejects larger gaps. These windows are model parameters;
the wider passing opportunity does not override Overtake Mode eligibility.

Battles are resolved from the front toward the back of the physical queue.
After a successful pass, the next attacker faces its new immediate neighbour;
it cannot skip a car by using the pre-pass order. Passing changes positions,
while the existing clock reconciliation charges blocked running without
removing elapsed race time.

Critical weather and damage stops retain priority. A noncritical weather
mismatch does not by itself justify a stop. Rain sets with completed wet-tyre
use and, from lap two, slicks in damp conditions follow the cost planners
described below. For the remaining reactive paths,
before the existing reaction draw,
the simulator estimates whether tyre gains over the remaining race can cover
expected pit-lane, stationary and queue loss. Reactive wet/damp pit-window
proposals, including SC/VSC opportunities, pass this same cost veto after the
window proposes stopping. A large gap behind alone does not make a stop free.
The check also prevents a newly fitted rain set from being replaced solely to
meet a planned lap. Legacy callers without weather retain their existing
behavior because there is no surface state to project. Surface wetness follows the same
deterministic rainfall and drying response used in race evolution, assuming
current rainfall persists. The projection consumes no random draws and does
not predict changes in weather condition or future race interruptions.

The comparison deliberately favours stopping, but every projected refit now pays
for lane travel and expected service. The first replacement follows the actual
fresh-weather crossover: a stop that would fit wets cannot claim the pace of
fresh intermediates. Later refits may use any noncritical compound, with no
inventory, stop-budget or compound-use restriction. Each set must run at least
one lap, ages normally, and cannot continue into a critical mismatch. This
expanded set of future options gives an optimistic cost for stopping now.

The alternative retains the existing set while it is noncritical. If it would
become critical before the finish, this waiting policy pays full lane and
expected service loss to fit an appropriate fresh set before that lap, repeating
only when another change becomes necessary. At a slick transition it chooses
the cheapest such safe-retention policy among the three slick compounds. This
provides a feasible waiting alternative without claiming optimal stop timing.
If rivals remain, the waiting policy receives maximum dirty air throughout,
while the optimistic stop-now plan receives clear air; a lone car receives no
fictional traffic relief. The calculation uses noise-free lap times,
including the pace floor. Only this lap receives the current SC/VSC running and
lane factors; queue delay applies only to the current stop. Future laps and
stops assume green running. Results are cached with bounded capacity.

If even the optimistic paid-refit plan is slower than this waiting policy,
the car stays out and re-evaluates next lap. A currently critical set and
unresolved compound-use requirements retain priority. A forecast need to change
tyres later no longer bypasses today's cost comparison. Passing the filter can
still permit a losing stop because its future options and traffic relief may be
unavailable. This remains a conditional cost filter, not an optimal wet-race
schedule or a guarantee about unpredictable weather.

When a clearly dry stop is already committed but has no optimizer proposal,
the simulator compares eligible fresh slicks over the entire remaining race.
This covers punctures, mandatory stops and returns from rain tyres. The current
paid stop consumes a budget slot before later stops are considered. Its lane
and service loss are common to all compound choices; future paid stops are
included in the comparison. The new set runs the current lap, including any
SC/VSC running multiplier, before a later stop is allowed. A repeated compound
is eligible only if the remaining projected schedule can still satisfy the
dry-use rule. The choice does not sample randomness or alter the race state.

For a damp fallback stop whose compound has not already been selected, the
simulator compares noise-free lap costs over the next stint for each eligible
fresh slick. The projection shares the actual lap model's compound pace, wear,
driver tyre management, circuit stress, car performance and degradation factor,
including its lap-time floor. Fuel uses the original scheduled race distance
even when the planning horizon is shorter. Surface conditions evolve from the
current weather without forecasting random weather changes; current race-control
and aero restrictions apply to the first lap, with future laps assuming green
running. When a new distinct slick is required, the comparison is
restricted to unused compounds. An archetype's preferred compound can override
the fastest only within 0.05 seconds per projected lap. This tolerance represents
a bounded strategy preference; it is a model assumption, not an empirical fit.

In fallback strategy, the current planned stop is consumed before determining
the next stint's target.
The horizon ends at the next remaining future plan entry that the ordinary stop
budget allows, or at the finish when none remains. Pit service occurs before that
lap's pace calculation, so a final stint includes the lap on which the stop
happens. Weather stops still consume stop budgets and fallback plan slots. On
returning to clearly dry slick running, the optimizer reassesses the remaining
race with the actual tyre age, compound history and stops remaining.

The tyre model separates accumulated wear from a bounded grip indicator.
Grip has an absolute floor of 0.5, capped at the set's initial grip, so a custom
set starting below 0.5 cannot gain grip. At tyre-management rating 0.8, the
default compounds reach this numerical grip bound at these completed tyre ages:

| Compound | Age at grip floor | Configured cliff age |
|---|---:|---:|
| Soft | 19 | 20 |
| Medium | 28 | 30 |
| Hard | 46 | 45 |
| Intermediate | 17 | 35 |
| Wet | 20 | 40 |

The floor does not cap the pace penalty. Wear continues at the configured
degradation rate, increasing by the cliff multiplier beyond the configured
threshold. The previous model derived pace from bounded grip and consequently
stopped charging for additional age, often before the cliff. The corrected
model retains the existing coefficients and pre-floor relationship while
removing that plateau. These coefficients remain model assumptions;
see [the wear model and executed strategy checks](tyre-wear.md).

When the current rain compound remains the fresh-set choice throughout the
projected remaining surface conditions, rain strategy compares stopping now
with waiting at least one lap and making optimal later same-compound stops.
The projection uses noise-free lap physics, current tyre age, circuit stress,
car degradation and the remaining paid-stop budget. Fresh sets run on their
fitting lap. Only the current stop receives known queue/rejoin costs and the
current SC/VSC lane discount; future stops assume green running. Only the first
running lap receives current control and Active Aero restrictions. The original
fuel distance remains separate from a shortened planning horizon.

This planner can choose a worthwhile stop outside calendar windows, or wait
when a later stop is cheaper. Ties favor staying out. It assumes current rainfall
persists and replans after each lap; it does not forecast random weather changes,
future incidents, future traffic, or tyre inventory. It compares clean-air pace
for both actions, with only the immediate rejoin adjustment. This is an optimum
within the same-compound projection and budget, not a claim of globally optimal
wet-race strategy.

If the projected fresh compound changes, a car that has actually completed
running on rain tyres compares compound-changing schedules instead. From lap
two, slicks on a surface that is not clearly dry use this same search. The search
can retain its current set while noncritical, fit the appropriate fresh rain
compound, or choose among soft, medium and hard when the surface allows a fresh
slick. It compares stopping now with running at least one more lap before any
stop, including later paid refits and their tyre ageing. It therefore can wait
for slicks instead of buying a short intermediate stint, or move to slicks
before the retained rain set becomes critical. Elective opening-lap stops remain
disabled; fitting an unused rain set does not itself earn a dry-use exemption.

Each projected path carries its actual compound-use history. A finish requires
two distinct slicks unless a rain compound has run in this race. The current
set's prior opening wear is kept separate from race use; future fittings gain
credit only when they run. This permits an elective same-compound refit followed
by a required distinct-compound stop, and prevents an anticipated but unused
rain set from making an otherwise illegal all-slick finish appear cheap.

Direct `plan_rain_transition` calls starting on slicks must pass
`used_compounds`, including an empty set when none has run. Omitting it for a
retained rain compound preserves the older caller's assumed wet exemption.
Native engines always pass their recorded actual use. Histories form part of
the cached cost keys so a legal plan cannot be reused for an incompatible history.

The four-stop rain allowance remains available while the car runs rain tyres,
including on a drying surface below wetness 0.3. It no longer drops to the
ordinary short-race allowance before an intermediate-to-slick stop. Slick-start
projections also reserve this allowance when the projected surface calls for
rain tyres, and reserve the three-stop dry allowance when drying will enable it.
With an externally timed weather clock, the caller keeps those dry and rain
envelopes available even when a no-stop forecast remains damp; the current damp
allowance still limits elective fits before the delayed transition.
Previous paid stops count toward it; after fitting slicks, the applicable dry or damp
policy determines subsequent decisions. The transition search counts all paid
fits against the allowance and restricts later stops on slicks to the applicable
dry or damp limit, except on surfaces above wetness 0.3 where native race policy
retains the rain allowance. A permitted fourth rain-to-slick stop cannot claim another
elective slick refit afterward. Mandatory
replacements of critically unsuitable sets and required compound-use corrections
remain available after that allowance
is exhausted, and their lane and service costs still count. Each new set runs
on its fitting lap before another stop is possible. Only today's stop receives
the known queue/rejoin adjustment and SC/VSC lane discount; only today's running
lap receives current control and aero restrictions. Fuel follows the original
scheduled distance even with a shorter planning horizon. The selected slick is
carried into paid execution, with weather eligibility checked again; a newer
rain requirement overrides it.

Both engines re-evaluate this plan using their observed weather and remaining
distance. Future green, clean-air running and persistent rainfall remain
assumptions. The plan does not anticipate random weather changes, pit queues,
later strategy-style changes, or the actual time of a lapped finish.
Its optimum is bounded by these assumptions and fresh-set eligibility. Slicks
return to the dry optimizer when the surface is clearly dry. A forecast is
recomputed at every decision; changes in weather or the remaining race distance
can change the selected plan.

In the remaining reactive fallback, wet/damp planned windows are consumed once.
After the selected plan is exhausted,
it cannot fall through to another generic late-race window. When no explicit
plan exists, the generic schedule offers at most two stops (or one when the
ordinary budget is one). The higher wet stop allowance still permits reactive
weather changes and SC/VSC opportunities; it does not repeat the second window.
This prevents fresh intermediates being replaced again on consecutive laps
solely because the car remains inside the same calendar window. The remaining
fallback timing is a heuristic.

The fallback compound comparison assumes the stop has already been chosen;
the dry optimizer additionally includes pit loss. The dry optimizer omits common
fuel and car pace terms only when all projected laps stay above the lap-time
floor. If clipping is possible, it compares full noise-free lap times using the
same physics as race execution, including the 95%-of-reference-lap floor.
This prevents crediting a fresh set with pace gains that execution would clip
away. Such clipping is possible with custom high-aero configurations; this is
a consistency correction, not a calibration to a real circuit.

The full-lap projection retains the original scheduled fuel distance even when
the planning horizon is shortened by the race clock or a car being lapped.
Current aero eligibility and SC/VSC running factors apply only to the current
lap; future laps assume green running. Cached costs in this path are absolute
lap times, whereas the ordinary fast path reports tyre-relative costs. Costs
from those two bases should not be compared across different model inputs.
The unlimited-set planners described above do not constrain physical inventory.
Drivers with an explicit pool use the separate [finite-set policy](tyre-inventory.md)
for opening selection and later decisions. Neither policy forecasts random
future weather changes or incidents, prices traffic beyond the immediate rejoin
lap, or jointly schedules both teammates' future stops. Cost tables are bounded
in-memory calculations;
they do not persist provider data or consume simulation random draws.
Static circuit profiles and car/circuit pace terms also use bounded, process-local
caches keyed by their numerical inputs. Editing sector weights, passing
opportunities, car ratings or reference pace produces a new calculation; no
mutable model objects or sampled lap outcomes are retained by these caches.
Weather pace scaling likewise reuses bounded, process-local scalar results.
Each call still evaluates the weather model's multiplier and wet severity;
those values, the driver's wet skill and the car's wet performance form the
cache key. Changed inputs and custom weather methods therefore remain visible.
This avoids repeating identical weather arithmetic across tyre ages and search
branches without changing the search, floating-point formulas or random draws.
Timed same-compound rain searches also reuse the native clock's absolute update
schedule for each paid-stop count and first-stop choice within one planning
call. Stints use the corresponding relative suffix, preserving repeated and
skipped weather updates. The schedules are discarded with that call; they do
not shorten the horizon, limit the search or cache mutable weather models.
Custom clock subclasses retain their ordinary query path.

Run `python examples/check_stint_choices.py` for a deterministic synthetic
comparison of fallback stint choices against actual lap calculations over short,
medium and long stints. The diagnostic makes no network requests and compares
fresh compounds at the same stop, without traffic or incidents.
It also checks damp fallback choices on a custom high-aero circuit, including a
shortened planning horizon with the original fuel distance, against actual
noise-free lap totals and the allowed team-style tolerance.

Run `python examples/check_pit_timing.py` to compare the chosen strategy with
every permitted one-stop lap and unused compound in controlled synthetic
30-lap full races. This also checks pit execution and tyre ageing, not just
the optimizer's own cost calculation.

Run `python examples/check_weather_transitions.py` to compare the adaptive rain
policy with bounded schedules executed by both engines (`--engine standard` or
`--engine chronological` selects one). The synthetic cases cover intermediates
on a drying surface at two lane costs, wet-to-intermediate-to-slick running,
and increasing rainfall. Every safe eligible schedule within a two-stop
allowance is executed, including compulsory replacements after it is exhausted.
The cases use mean lap pace, expected service, evolving surface wetness, and no
incidents or traffic. JSON output records the inputs, checked schedule count,
chosen and best executed results, and their time difference. The eight-lap
drying case uses a 350-second reference lap and a 22-second lane loss. Changing
on lap three saves about 24.46 model seconds against waiting until lap eight,
without an additional stop. In the 24-lap wet-to-dry case, an eight-second lane
loss makes wet-to-intermediate-to-soft changes on laps nine and nineteen about
4.70 seconds faster than a direct wet-to-soft change on lap nineteen. Both
engines match the best executed schedules in these controlled cases.
This is a regression example under synthetic physics, not a real-race estimate.

Run `python examples/check_dry_pit_schedules.py` for a broader bounded dry
comparison in both engines (`--engine standard` or `--engine chronological`
selects one). Each synthetic eight-lap race starts on medium and exhaustively
executes all 1,092 legal schedules of one to three stops after lap one, including
soft/medium/hard replacement sequences and repeated fresh sets of the same
compound. The initial medium set must be used and at least one different slick
must be fitted. Stop counts have 14, 168 and 910 alternatives respectively.

The balanced policy is compared with the fastest executed schedule using mean
pace, expected service, fixed dry weather and no incidents or traffic. Normal
lane loss, cheap stops with high wear, and a deliberately long 600-second lap
reference exercise one-, two- and three-stop optima. Both engines currently
match these references. The JSON output includes inputs, completed distance,
paid laps, tyre sequence, total time and the gap to the best bounded alternative.
The diagnostic makes no network requests or default file writes. Its synthetic
inputs are not calibrated venues, and it does not search beyond three stops,
different opening sets or future weather changes.

Run `python examples/check_restart_choices.py` to compare selected free restart
sets with forced soft, medium and hard alternatives in controlled 60-lap races.
The diagnostic covers a five-lap sprint, a long final stint, and high wear with
a later paid stop available. Each alternative uses the actual race engine and
remaining pit strategy with mean pace and expected stationary service.

Sampling ranges in the dashboard measure Monte Carlo noise under these
assumptions. They do not validate the strategy model against real race outcomes.

Driver points projections divide total awarded points by that driver's observed
race count, including retirements, and rank the resulting means. Requested trial
counts never dilute a partial result. Constructor projections sum the listed
drivers' individual observed means; this is a lineup projection, not an average
over a reconstructed union of team race records. Drivers with no observations
and teams containing an unobserved listed driver are omitted. A recorded zero
remains zero. Normal runs with every entrant observed in every trial are unchanged.

Event rates use the event ledger's recorded trial count when available. Legacy
aggregates without that count retain their nominal requested count; exports and
dashboard responses expose the denominator as `event_rate_trials`. Console and
dashboard event summaries show it alongside the rates. Mechanical breakdowns
report simulated counts and shares. Automatic reports have no configured
reference component shares, so they provide no calibration delta, tuning
suggestions or reliability adjustments. Python analysis helpers retain explicit
caller-supplied reference comparisons; with an empty failure sample their
calibration delta is `None` and their advice is empty. An absence of observed
failures cannot establish the relative proportions of failure components. See
[mechanical reliability](mechanical-reliability.md#component-attribution-and-reporting).

The dashboard statistics and HTML reports include mean winning distance,
lapped finishers, time-limited races and races without a winner. Statistics JSON,
scenario comparison JSON and dashboard scenario responses expose the same
`race_distance_statistics` object. Rates are fractions from zero to one.
Race counts use recorded raw races, not the requested simulation count. Mean
winning distance and finish outcomes also appear together in offline comparison
reports, with their own denominators, so scenarios with shortened races or lapped
finishers can be interpreted alongside points and paid-stop counts. Each driver's
comparison also lists actual tyre sequences, with finished and retired counts,
shares among recorded sequences, and missing records shown separately. Free
fittings remain part of sequences without becoming paid stops. These frequencies
describe model behavior and do not establish the best strategy. Mean
winning distance includes only winners with a positive recorded lap count;
lapped-finisher rates include only finished cars whose own distance and winner's
distance are both known. Classified retirements do not count as finishers, and
the actual finished P1 supplies winner distance even if a retired car completed
more laps. Empty distance denominators return null and display as “Not recorded”.

Run `python examples/check_rain_pit_timing.py` for exhaustive short wet-race
comparisons in both Standard and Lap-aware engines. Each selected strategy is
compared with every schedule of up to four paid stops after the opening lap:
99 schedules for eight laps and 562 for twelve laps. The synthetic cases cover
intermediates, wets, and no-stop, one-stop and two-stop optima using actual tyre
ageing, fuel and pit execution. They use a single car, fixed rainfall/surface,
noise-free laps and expected service, with incidents disabled. Long reference
lap times and cheap lanes deliberately exercise worthwhile fresh-tyre stops;
these are model checks, not calibrated venues or observed race comparisons.

Run `python examples/check_mixed_weather_pit_schedules.py` to check complete
six-lap races starting on slicks in both engines. It enumerates every survivable,
compound-compliant schedule under the balanced policy's changing stop allowances.
The four cases cover a used soft set in steady damp conditions, a drying track,
a wetting track, and a drying track that enables later dry stops. They check
30, 186, 30 and 186 schedules per engine respectively. Every alternative uses
actual lap, ageing, surface and paid-stop execution with noise and incidents
disabled. Original opening tyre wear does not count toward race compound use.

The used-set case previously waited until lap six; cost planning chooses a fresh
soft on lap two and the required medium on lap six, saving 18.64 modeled seconds.
The last case uses a deliberately long 600-second reference lap and cheap pit
lane to expose later dry-stop choices. It checks that today's damp allowance
does not incorrectly restrict the forecast's later dry phase. These synthetic
results validate decisions within the model; they do not calibrate degradation
or establish real-world strategy gains. Grip bounds are separate from the
accumulated wear costs described above.

## Planner performance benchmark

`examples/benchmark_strategy_planning.py` measures complete synthetic races,
including qualifying and adaptive pit decisions, without fetching provider data
or writing files. The default uses 22 drivers, 53 laps and three consecutive
seeds beginning at 42. Every driver starts on an explicit compound, with prior
wear cycling through zero, six and twelve laps. Driver and car performance
vary across the grid. Rainfall and weather condition remain fixed while surface
wetness evolves; ordinary race incidents remain enabled.

```powershell
python examples/benchmark_strategy_planning.py --engine chronological --scenario steady_damp
python examples/benchmark_strategy_planning.py --engine standard --scenario drying
python examples/benchmark_strategy_planning.py --scenario wetting --trials 3
python examples/benchmark_strategy_planning.py --scenario rain_transition --trials 3
```

Use `--drivers`, `--laps`, `--trials` and `--seed` to change workload size and
the seed range. With explicit openings, scenarios start on soft tyres except
`rain_transition`, which starts on intermediates. `dry` begins with no rain or
surface water. These synthetic inputs exercise the planner;
they are not calibrated circuit or weather forecasts.

`--inventory finite` gives each driver three reusable physical sets: soft aged
five laps, fresh hard and intermediate aged four laps. Explicit starting ages
then match the selected physical set. `--opening automatic` removes the opening
override and includes native starting-set selection in the workload, for either
finite or unlimited pools. These modes are recorded in benchmark version 2 JSON
alongside the initial pools and overrides. See the
[finite-pool search and benchmark notes](tyre-inventory.md#search-and-benchmark).

JSON output includes individual trial times, the first trial, the mean of later
trials, and their total. In a fresh command-line process, the first trial starts
with cold strategy caches; later trials can reuse them. Output serialization
and hashing are outside the timed interval. `outcome_sha256` covers race rows,
qualifying rows, weather histories and event statistics across all trials,
excluding timing and runtime provenance. Matching hashes establish exact output
agreement for that workload, not equivalence for every possible input.

Compare revisions with the same script, arguments, Python and dependencies on
the same machine, using a fresh process for each run. Check the outcome hashes
before interpreting speed changes. Run timing comparisons without competing
CPU-heavy work, and repeat them to distinguish improvements from timing noise.
Cache reuse, weather, driver inputs and race length can change the benefit;
there is no hardware-independent timing threshold in the test suite.

The transition solver suspends a stint while evaluating a missing future cost
and resumes at that stop choice. This avoids rescanning earlier choices whenever
a dependency is resolved. An explicit stack handles long horizons without
Python recursion. Completed suffix costs retain the shared 8,192-entry bound,
locking and fork reset. Working memory also includes per-plan completed states,
active frames and their running-lap rows. The optimization preserves candidate
order, cost arithmetic and modeled strategy rules.
