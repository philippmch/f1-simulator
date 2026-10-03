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

Without a prescribed rainfall schedule, dry planning scales tyre costs with the
same current weather multiplier as simulated laps, including surface water,
rainfall and the car's wet-performance
contribution. It holds that multiplier constant over the projected stint;
this is not a forecast of future weather. Expected service and pit-lane loss
are not weather-scaled. Without a usable known-duration field, the current
SC/VSC running multiplier applies in addition to weather scaling for this lap
only, with later laps assumed green.

Before a timed finish is announced, in-race strategy estimates its distance
from the leader's elapsed time and most recent running pace. Pit service,
post-fit costs, incident loss and time spent waiting behind another car are
excluded from the recurring pace estimate. An observed integer SC/VSC countdown
affects all its remaining leading intervals, followed by green running. Unknown
or overlapping deployments retain the single-interval forecast. The estimate
includes the lap following clock expiry and never exceeds the scheduled distance. Without a pace
observation the automatic policy retains that schedule. Both engines use this estimate for pit
decisions and free tyre choices; chronological races also map the leader's
estimated finish time to each car's own remaining laps and include completed
suspension extensions.

The actual timed announcement remains distinct from elapsed clock expiry.
A car can cross the leader's already-completed lap after expiry without
announcing the finish. If that car supplies the next strategy forecast while
the previous leader is in service, the estimate still includes the first
expiring leading crossing and the crossing after it.

Chronological forecasts resolve equal completed distances using physical
on-track order, ahead of cars still in the pits, rather than stale crossing
times. A pitting car a full lap ahead retains distance priority. A pending stop
uses its expected exit when planning the remaining race. A committed tyre
fit's optional first-lap cost delays the projected outlap crossing once, after
the current running multiplier. It does not delay pit exit or become recurring
pace. An on-track pending crossing already includes that cost. The actual
finish controller continues to use executed crossings.

Rain and weather-stop projections use this shared leading clock to estimate
surface conditions at each car's future lap starts. A slow car can see multiple
surface updates between its own laps; a faster car can see the same surface
twice. The forecast includes no update at the projected chequered crossing.
It uses observed free pace and expected unfinished leader service through the
known remaining control intervals, with green running afterward. When committed
field entries are usable, a private crossing projection preserves physical
SC order and pending no-passing barriers, including barriers surviving a return
to green. It resolves changes of the projected leader caused by those committed
entries. Neither live control nor the finish ledger is advanced. Unresolved
rival repairs, tyre instructions or observations retain the simpler held-pace
forecast. Without usable pace observations it retains one update per own lap.
By default rainfall and condition
remain fixed, and no future weather draws are consumed. A supplied
[prescribed rainfall schedule](weather-schedule.md) instead provides known
future changes at that same leading cadence, with existing standing water
carried through each change.

When another car supplies the chronological forecast's leading clock, rain,
weather-stop and finite-pool costs also account for the candidate's own planned
paid stops. Retaining a tyre starts running immediately; fitting a replacement
starts after expected lane, service and any observed queue delay. Each later
paid stop adds its expected green-running pit loss to subsequent projected
starts. Traffic-related cost adjustments do not advance this physical clock.
The compound choice uses conditions before service, while its running cost uses
the projected surface at rejoin; the execution still fits just once.

When the upcoming full-safety-car queue is observable, the external weather
clock applies that queue to its held free-pace estimate for the first running
lap. Retaining and stopping use their respective observed entry gaps. Catch-up
can therefore advance subsequent lap starts across fewer leading weather
updates, while a slow observed queue can delay them. This replaces only the
first running interval in ordinary forecasts. With a usable known-duration field,
nominal retained starts instead use the private crossing projection, including
its first committed fitting fee exactly once. Physical service and fitting
delays retain their separate treatment. A missing or unresolved queue keeps
the uniform control estimate. The clock continues to hold observed free pace
across candidate compounds; first-lap tyre costs are priced separately.

Explicit leading update times preserve the slow control prefix and later green
cadence for both retained and paid branches. Expected paid service and fitting
fees can move an entry across several of these events, but the flag never adds
a surface update. This improves entry surfaces and estimated distance. Rain,
weather-stop and transitional finite-pool cost planners still assume green
running after the current lap and green losses for later paid stops. Clearly
dry unlimited-set decisions and finite pools with zero rainfall and standing
water can price the known control prefix as described below. The remaining
weather paths do not price a complete future neutralized queue.

The external leading clock remains fixed under these comparisons. A candidate
that is itself the projected leader, including a single-car race, retains the
own-lap projection so its stationary pit time cannot invent weather updates.
The forecasts do not resolve future rival decisions, tyre-dependent changes
in free pace, battles or interventions. They retain the current planning
distance and are recalculated at the next decision; stop timing can still change
actual finishing distance. Matching the shared weather cadence does not
establish an optimal complete strategy or predict real weather.

The forecast is recalculated as the race develops. It does not change the
actual finish controller, predict future stops or weather, or guarantee a
globally optimal timed strategy. After a timed announcement, the next leading
crossing remains authoritative even if a lapped driver inherits the lead.
Original scheduled fuel distance remains separate from all strategy horizons.

Before committing an elective stop, both engines also check whether a leading
car would sacrifice a completed lap. The chronological engine applies the same
check to followers under their external projected leading finish.
It compares a mean-pace continuation on the fitted tyre with an optimistic stop
continuation: expected lane, service and queue time, a mean outlap, then later
laps at the lap model's minimum time without further pit or traffic losses.
For followers, both paths end at their first crossing at or after the same
projected flag. For a leading candidate, each path supplies its own crossings
to the timed clock: the first leading crossing at expiry announces the finish,
and the next leading crossing supplies the flag. Frozen rival streams can lead
while the candidate is in service. A rival crossing just before expiry can
therefore preserve an extra lap that an isolated-car forecast would miss.
Every path remains capped at the original scheduled distance.
A stop is cancelled only when retaining the tyre remains feasible and completes
more laps than this optimistic stop path.
Equal-distance decisions retain the ordinary strategy planner's choice.

Each rival starts from its committed on-track crossing or expected unfinished
service and optional fitting cost, then holds its observed free pace. Standard
decisions also include earlier stops committed in that lap's reservation order.
Collected chronological restarts freeze the original leading candidate's rival
view after all free tyre fits and before any running or paid service resumes.
Future elective rival stops, battles and interruptions are unknown.

The check projects persistent rainfall or a prescribed schedule at track entry,
including expected pit exit. Standard paths retain one weather update per shared
lap; chronological paths follow leading crossings, including updates during
service and excluding the flag. It consumes no future weather or service draws
and does not fit or reserve physical tyre sets. Finite-pool comparisons preserve
the proposed replacement's identity and wear. Automatic opening comparisons use
the same guard when executing their isolated policy paths.
Forced repairs, critical tyre mismatch, unavailable fitted sets and unresolved
compound-use requirements remain under the existing compulsory-stop rules.
Explicit executable pit instructions retain their requested stop. The check
also covers SC and VSC fields in both engines. The current running modifier
and Active Aero availability apply at track entry. Reduced lane loss,
stationary service and optional fitting cost keep their execution semantics;
the fitting cost is added once after running. Standard paths retain the later
green forecast.

For a standard SC field, staying out and stopping have separate first-lap
queues. Each freezes the complete pit-exit order, mean rival tyre pace and
expected earlier committed stops. Pitting the original leader can change the
car setting the queue's running pace. The forecast applies the actual shared
running rule, then fitting delays and the no-passing barrier, to the entire
field. Each branch supplies its own candidate and rival first crossings to
the timed clock. Subsequent rival laps still hold observed green pace; future
elective stops and interruptions remain unknown. A known but unresolved rival
repair, critical mismatch, due pit instruction or replacement keeps this SC
comparison inactive, as do unavailable mean inputs or custom merge rules.

Chronological neutralized comparisons advance a private copy of the complete
finish ledger and circular on-track order. Committed running keeps its known
readiness; unfinished paid service uses its conditional expected exit and a
single pending fitting cost. A paid candidate rejoins behind the line. Pending
neutralized laps keep their no-passing restriction until they cross, including
after control returns to green. A lapped physical predecessor can therefore
delay an otherwise leading crossing without inventing an extra retained lap.
Each branch supplies its actual conditional leading crossings to the finish
clock and evolves persistent rainfall or the prescribed schedule only at
those crossings, excluding the flag.

The chronological field forecast also honors the observed remaining SC/VSC
interval count, decremented at each leading crossing. This applies to a
single car when more than the current interval remains, and to green decisions
with earlier neutralized running still pending. An outlap entering after a
control or weather update uses the updated conditions; its lane/service loss
stays committed at the decision. After its mean outlap, the optimistic stop
uses the absolute running floor with applicable control and queue constraints.
Retained laps continue mean tyre ageing and entry traffic. Rival recurring
laps hold their latest observed free pace; later green passage is free and
future stops, incidents or control extensions are not predicted. The retained
first lap preserves eligible current Overtake Mode without forecasting later
deployments. Unresolved rival repairs, critical mismatch, a due next-lap pit
instruction, missing pending-event observations or unknown control durations
keep this comparison inactive.

The check remains inactive during a red flag.
It also remains inactive without a usable forecast, including copied inputs
unavailable to a custom physics hook or an unresolved committed rival fit with
differing possible fitting costs.
It protects distance under the native lap model and these conditional mean-pace
assumptions; it does not guarantee the sampled race outcome or solve
the complete timed strategy problem.

Pit-rejoin traffic uses the same projected finish time, even before the timed
finish is announced. Each rival remains traffic until its own projected final
crossing; lapped cars can therefore remain after the leader takes the flag.
For a rival still in service, a pending fitting cost is included once in its
projected outlap crossing and fractional rejoin position, with pit exit
unchanged.
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

The optional [prescribed rainfall schedule](weather-schedule.md) supplies a
known sequence for controlled rain-and-drying experiments. Its changes take
precedence over random atmosphere transitions and are shared by race execution,
automatic opening choices, paid and free tyre changes, and finite-pool planning.
When future changes remain, constant-dry and same-rain shortcuts give way to the
existing general surface-path searches. Schedule laps follow the shared leading
clock, including red-flag restarts and leader handoffs, rather than a lapped car's
own lap number. Rainfall changes before the ordinary surface update and never
resets standing water. This supplies scenario knowledge, while retaining the
conditional clock, bounded stop budget and other forecast limits described above.

Automatic starting-compound selection on a clearly dry track compares isolated
runs of the existing pit policy for each slick, using noise-free lap pace and
expected service time. These runs follow the race clock, retain the original
fuel distance, and include later paid stops and the distinct-compound rule.
They compare completed distance first, then fulfilled custom instructions when
a plan is supplied, then elapsed time. Only actually executed requests before
the projected finish count; skipped, overridden and unreached requests do not.
This avoids choosing an opening tyre for a scheduled distance that the time limit will cut
short, or rewarding a slower policy merely because it completes fewer laps.
Existing strategy weights choose among equivalent best outcomes, with a
one-billionth-of-a-second tolerance for elapsed-time ties after matching both
distance and request fulfillment. A dry direct comparison with prescribed
weather changes uses the same eight private reaction seeds as the general
weather-opening comparison; a dry surface without that schedule retains one.
One-lap races and cases without a legal projected finish retain the original
weighted choice. Explicit starting-tyre overrides retain priority.

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
pools, and require an explicit policy. Custom pit plans use schema 5; enabling
[post-fit cost sensitivity](tyre-wear.md#optional-post-fit-cost-sensitivity)
uses schema 6 and preserves any configured ages, inventory, and plans.
Explicit [qualifying-session weather](qualifying-weather.md) uses schema 7,
preserving the race weather and any of those optional settings.
Schema 4 records the initial inventory.
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

Paired comparisons also aggregate constructor points using the saved team
membership. A constructor observation requires valid points for every modeled
runnable member in both races of a qualifying-matched seed pair. Points are
summed within that trial before computing the paired difference and its SE,
so teammate covariance is retained. Missing teammate observations exclude a
constructor pair rather than becoming zero points. The reported members,
paired count and excluded count make that population explicit. A focused
driver view still includes the modeled teammate in the team total. See
[constructor comparisons](custom-pit-plans.md#paired-constructor-points).

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

With driver and car inputs, wet and transitional openings compare every
currently noncritical compound, including full wets and safe slicks, using
isolated, traffic-free runs of the existing pit policy. A numeric weather
recommendation does not exclude a faster safe alternative. These runs cover
the full race, use mean lap pace and expected service
time, and advance surface wetness after each lap under constant rainfall and
weather condition. They include later paid stops and the compound-use rule.
Each candidate uses the same eight fixed private reaction seeds; the selector
compares their average completed distance first and average total time second,
so a slower, shorter time-limited race cannot win merely by ending sooner.
Ordinary comparisons favour intermediates in a tie. The real race's random
generator and input objects are untouched. Positive fitting warm-up costs
apply to later paid changes; the opening set is already ready to run.

This is an approximate comparison of the current policy under sustained
conditions, not a global wet-strategy optimizer or a forecast of changing rain,
traffic, or incidents. Close choices can depend on the sampled reaction paths.
The isolated runs follow the default automatic later pit policy, or this driver's
supplied [custom pit plan](custom-pit-plans.md). An explicit empty plan suppresses
elective stops; unsafe or unavailable requests and mandatory corrections use the
same decisions as actual execution. The supplied schedule is retained, rather
than searching for a different one. Dry, wet and finite-pool opening comparisons
all account for it. Explicit opening compounds and ages keep priority.
Compulsory replacements and free red-flag choices under a supplied plan use a
separate continuation over the remaining requested services. It follows the
same skip, availability and actual-use rules, with compulsory weather and final
compound corrections but no elective automatic stops. A projected leader's
observed deadline also truncates these candidate continuations at the lap after
expiry, even before the first pace observation. Distance is ranked first,
fulfilled requests second, and time third. A cheap continuation cannot win by
making a safe requested set unavailable when an equally long path can honor it.
The existing planning cap still applies. Followers retain their estimated own
finish horizon and externally observed weather clock. The forecast preserves
original fuel distance, set wear and post-fit costs and assumes later green
running. It does not forecast traffic, random weather or leader changes, or
optimize the user's requested schedule.
Results are cached with a bounded capacity, including the driver, car, circuit,
weather, tyre configuration, strategy settings, supplied pit plan and race time limit. Dry,
precautionary and finite-pool scores share the same input normalization: driver
and team names and previous race state do not create new physics. Changing the
deadline recomputes the completed-distance ranking before a new opening is
selected. Direct selector calls
without driver/car context retain the numeric weather fallback.

Run `python examples/check_weather_openings.py` to compare automatic openings
with every safe explicit opening executed by both engines. Seven synthetic
wet, drying, damp, warm-up and timed cases use all eight private reaction seeds.
The JSON report compares mean completed distance before mean elapsed time;
the command exits with status 1 for a mismatch. In its 12-lap drying case,
starting on intermediates completes the same distance about 3.67 model seconds
sooner than the threshold-recommended full wets. All fourteen engine/case
comparisons match the fastest safe executed opening policy. This checks the
existing later pit policy under fixed rainfall, not every possible schedule or
real-world tyre performance. `--engine` selects either engine or both.

Add `--custom-plans` to check twenty custom-policy cases in both engines:
empty, early, late and repeated-compound plans; unsafe requests; prescribed
weather; finite pools with used and equivalent physical sets; fitting costs;
and instructions beyond a timed finish. The additional finite cases check
reservation of a sole requested wet set, reuse at accumulated wear, equivalent
wet replacements and request fulfillment before a timed finish. All forty
comparisons must match the best safe executed opening, comparing completed
distance, then executed requests, then elapsed time. The report includes
`mean_executed_instructions` and `mean_instruction_gap` alongside distance and
time; all three gaps must be zero.
Finite alternatives preserve both compound and prior wear. In the synthetic
60-lap, high-wear case with no elective stops, starting on hard tyres saves about
107 model seconds against the former automatic medium opening. These checks
validate the conditional implementation, without calibrating real tyre pace
or establishing an optimal custom pit schedule.

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

The numeric weather recommendation prefers intermediates above 0.2 surface
wetness or 0.4 rain intensity, and full wets above 0.7 surface wetness.
These values are normalized model parameters,
not measured millimetres of water. Direct opening calls without driver/car
inputs retain this numeric fallback. Native automatic openings and paid
strategy decisions compare every currently noncritical set using its projected
cost, including safe alternatives to the weather recommendation;
free red-flag refits compare usable sets over the remaining projected weather
as described below. Precautionary intermediates remain available at surface
water of at least 0.08 or rainfall of at least 0.15, using numeric conditions
consistently across labels.
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
their mismatch penalty on a wet surface. Qualifying also uses the configured
Straight Mode gain shared by race laps and strategy forecasts, subtracting it
before the weather multiplier and applying its existing qualifying time floor.
Compound comparisons and sampled attempts both assume that gain is available.
Each attempt assumes a fresh set;
By default, weather remains fixed across Q1, Q2 and Q3 using the initial race
weather. Optional [qualifying-session weather](qualifying-weather.md) overrides
each named session independently, with omitted sessions retaining race weather.
Compound selection and attempts use the same fixed session conditions; race
weather is unchanged. Qualifying does not model tyre inventory, track evolution,
traffic or changing conditions within a session.

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
The chronological engine freezes every car's restart horizon, weather clock
and custom-plan finish context before fitting or releasing the first car.
All free suspension fits finish before any car begins restart running or
starts a new paid service. This includes a car whose
paid service finished while the pit exit was closed; old service forecasts and
release order cannot change the free-tyre projection.
An exhausted pool retires that car before survivors are released. A scheduled
paid stop after the restart cannot change a follower's earlier suspension fit
by promoting it to projected leader during the release loop. Ordinary later
lap starts update the finish forecast from the observed field as before.

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
the [FIA 2026 Sporting Regulations, Issue 09](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf).
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
passing opportunities, or rivals' stop decisions. The green traffic correction
is disabled under neutralisation; full-SC forecasts instead use the observed
queue described below. Expected gaps approximate a nonlinear cost at the mean
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

The standard engine projects native pit traffic with compact rows containing
only driver identity, status, position and elapsed time. It reuses the same
gap and merge helpers, including physical-order tie breaks and position-hole
compaction, without copying complete tyre and strategy state for each branch.
Instrumented or overridden helpers retain full-state copies. This changes
allocation cost, not search limits, strategy decisions or cancellation checks.

A 2026-09-26 check compared this path with revision `abc1c13` in alternating
fresh processes. The synthetic workload used the standard engine, 22 drivers,
58 laps, unlimited tyre inventory, automatic openings and three consecutive
seeds starting at 42. Across three before/after pairs, median total simulation
time for the three trials fell from 2.003 to 0.848 seconds with fixed dry
weather, and from 8.051 to 6.683 seconds with weather-change probability 0.2.
All paired complete race, qualifying, weather and event digests matched.
Another 120 shorter cases retained identical digests across both engines,
five starting weather patterns, finite/unlimited pools, both opening modes,
and automatic/custom pit plans. These are local synthetic measurements, not
a speed guarantee for live inputs or a change to model realism.

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
Both engines recharge once per completed own lap using the control conditions
captured for that lap's running. A green lap followed by a new SC/VSC deployment
receives green recharge; the next neutralized lap receives neutral recharge.
A lap that began neutralized keeps that recharge rate when control ends at
its completion. The normalized store gains 0.04 for green running or 0.12 for
neutralized running, subject to its capacity. These are model assumptions,
not measured harvesting rates. This timing matters near the energy threshold
for a later Overtake Mode deployment.
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

Paid replacement selection retains these observed gaps and the expected queue
delay supplied at commitment, for automatic choices and custom-plan compulsory
replacements. The first paid outlap uses the rejoin gap in full lap physics,
before the lap floor and current control multiplier. A compulsory dry choice
prices this first running lap before any further service; a finite-inventory
survival fallback keeps the physical set's actual wear and pending fitting
cost. Fresh-stint fallback comparisons also use the observed rejoin gap.
A free-fit projection uses the stay gap until a paid request or compulsory
correction occurs on that same lap. Later laps assume clean air.

Expected queue loss enters only the current paid service, including its
weather and conditional finish clocks, and remains independent of actual
sampled service. Common current lane and service losses cancel when comparing
fresh sets over a fixed dry continuation. The standard engine keeps each car's
commitment snapshot through the service batch; the chronological engine passes
its observed traffic and remaining expected box wait directly to replacement
selection. Executable requested compounds remain authoritative, and these
forecasts do not predict future queues. Fallback lap and surface hooks operate
on copied driver, car, track, tyre and weather models so evaluating candidates
cannot change the live race inputs or shared tyre coefficients.

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
the chronological engine, rather than a fitted SC speed profile.

For example, two 90-second cars starting an SC lap 80 seconds apart take four
laps to settle at the target gap. With the leader running 126-second laps,
the follower runs 90, 90, 119 and 126 seconds; the gap becomes 44, 8, 1 and
1 seconds. Every completed lap contributes its full recorded time. A pit stop
can create additional gap that is subsequently recovered on track, while its
lane, service and queue losses remain in the stop ledger and lap accounting.
The lap on which an SC countdown ends still uses its starting restrictions.

Pit forecasts price that upcoming SC lap against separate retained and paid
track-entry snapshots. Expected lane, stationary service and team-box delay
remain paid costs; recovering some of the resulting gap reduces subsequent
running time. A fresh compound's pace gain can also disappear inside an already
formed queue. Dirty air enters free pace before the queue bounds are applied.
Fitting sensitivity is added afterward, with any known predecessor fitting
delay still preventing a neutralized pass. This applies to dry, rain, finite-set
and custom replacement forecasts, including external weather clocks and
selection after a compulsory stop has been committed.

The standard forecast holds undecided rivals on their current sets and includes
earlier committed stops with expected loss and a resolved replacement. Mean
running through that copied field supplies the same queue anchor and predecessor
crossing used in execution. An unresolved replacement or unavailable mean pace
retains the nominal current multiplier. Mean pace uses an isolated strategy
calculator; it does not advance the live lap sampler or race RNG.
The chronological forecast holds already
observed unfinished running through expected pit exit, including visible fitting
delays. It retains the nominal forecast if a rival is still in service or would
cross before or at that exit, because subsequent starts and physical order are
unresolved. It never reads a private future service sample to fill that gap.

These are conditional first-lap forecasts. Future running stays green and clean;
the plan does not predict later queue catch-up, future SC duration, rival stops,
or a race-control/weather change during service. The standard loop still
advances every survivor once per leading lap. The chronological engine schedules
individual crossings and pit exits. Their physical queue and pit-arrival
approximations can therefore produce different results. VSC retains its uniform
running multiplier and does not bunch the forecast field.

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

### Overtaking-counter reporting

The attempt counters record calls that reach the passing model. In the standard
engine, the adjacent-car end-of-lap proximity gate runs first, including the
wider restart window, so rejected larger gaps are not attempts. In the
chronological engine, an attempt is recorded when a car catches its physical
predecessor and the passing model is
used; a compliant blue-flag yield is not an attempt. Success and contact counts
describe those calls only, not every collision in the race.

Reports show pooled success and contact rates per recorded attempt, alongside
counts and coverage over available driver-race rows. A complete recorded zero
is distinct from a missing or inconsistent legacy counter triplet. These rates
describe the simulator, not real-world calibration. Repeated attempts in one
driver's race are correlated, so they are not presented with a naive binomial
confidence interval.

Battles are resolved from the front toward the back of the physical queue.
After a successful pass, the next attacker faces its new immediate neighbour;
it cannot skip a car by using the pre-pass order. Passing changes positions,
while the existing clock reconciliation charges blocked running without
removing elapsed race time.

A failed attack leaves the attacker behind its defender through that crossing,
including when contact gives the defender a larger sampled time loss. Contact
losses enter each driver's elapsed and lap clocks once and are recorded
separately in the event ledger. A faster provisional attacker then spends any
remaining advantage waiting behind the defender; that blocked running also
belongs to its completed lap. Following cars resolve their own battles against
the resulting physical queue. This keeps contact delays consistent with the
recorded passing outcome in both race engines.

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
for lane travel and expected service. The first replacement may use any compound
that is noncritical on the observed commitment surface, matching automatic
fitting eligibility. Later refits may use any noncritical compound, with no
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

With usable observations during SC or VSC, clearly dry unlimited-set stop timing
and the committed compound comparison price all known remaining control
intervals. Finite pools use that field when rainfall and standing water are
both zero, conserving replacement ages and multiplicity through the prefix.
Each candidate uses full mean lap physics at its projected track
entry, with the original scheduled fuel distance and current dry weather pace
scale. Future stops committed during
that prefix pay the observed lane discount and expected service. Fitting costs
follow running and are not multiplied by control pace; another SC lap can
recover the resulting gap. Known current team queue delay applies to the first
stop only. The standard engine advances its shared lap order, including its
completed-clock VSC pit rejoin. The chronological engine advances a private
copy of committed crossings and expected pit exits, preserving physical order
and old no-passing barriers after control ends.

Rivals hold their observed free pace after already committed running or service;
their future tyre choices, stops and incidents remain unknown. Known control
duration decreases at leading crossings in the chronological engine. If the
projected flag occurs within the prefix, plans rank completed own laps before
elapsed time. Once the control and its pending restrictions have ended, the
comparison uses the existing green suffix with a fixed own-lap horizon and
clean-air future running. It therefore does not claim a complete optimal field
strategy or predict later changes of leader, weather or control.

The search covers up to six known intervals, matching the longest native
deployment; a shorter remaining own-lap horizon can also bound the prefix.
Missing observations, unresolved rival instructions and longer custom control
durations keep the existing forecast. Custom pit plans, prescribed weather and
rain or damp policy paths retain their respective cost models. Finite branches
return to their physical-set green search after the prefix; optimistic native
bounds can relax stock for pricing but never invent an executable set. Field
branches and control-prefix caches belong to one decision and
never retain the live engine or consume simulation random draws. Existing
native green cost tables retain their bounded process-local cache.

Chronological branches reuse frozen scalar finish observations and copy their
mutable ledger, clock, queue and pending events. Additional mutable ledger
attributes retain deep copying. Decision-local memoization ignores stale events
and absolute scheduler counters while preserving the priority order of valid
crossings, including exact-time ties. This lets equivalent physical histories
reuse their future cost without shortening the search or changing pit choices.

In a controlled four-lap illustration, a fresh soft runs at 99.55 seconds versus
100 seconds on medium, but pays a two-second fitting fee. After three known
VSC laps and one green lap, the soft finishes 0.16 seconds sooner; under SC it
finishes 0.52 seconds sooner. Pricing just the first controlled lap would choose
medium in both cases. Completed native races in both engines, with unlimited
and finite pools, verify the choice with the same stop lap and finishing
distance. These are synthetic consistency
checks, not calibrated tyre performance.

Automatic paid changes on rain or damp surfaces compare complete remaining
schedules as described below. The lower-level fallback ranker compares
noise-free lap costs over the next stint for each eligible fresh slick when
that fallback is needed. The projection shares the actual lap model's compound pace, wear,
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

The direct `plan_rain_stop` API compares stopping now with waiting at least one
lap and making optimal later same-compound stops. Its caller must ensure that
the current rain compound remains appropriate throughout the projected horizon.
The projection uses noise-free lap physics, current tyre age, circuit stress,
car degradation and the remaining paid-stop budget. Fresh sets run on their
fitting lap. Only the current stop receives known queue/rejoin costs and the
current SC/VSC lane discount; future stops assume green running. Only the first
running lap receives current control and Active Aero restrictions. The original
fuel distance remains separate from a shortened planning horizon.

This planner can choose a worthwhile stop outside calendar windows, or wait
when a later stop is cheaper. Ties favor staying out. Without a prescribed
schedule it assumes current rainfall persists; it does not forecast random weather changes,
future incidents, future traffic, or tyre inventory. It compares clean-air pace
for both actions, with only the immediate rejoin adjustment. This is an optimum
within the same-compound projection and budget, not a claim of globally optimal
wet-race strategy.

Automatic rain-tyre decisions use the compound-changing planner. From lap two,
slicks on a surface that is not clearly dry use this same search. The search can
retain its current set while noncritical or fit any fresh noncritical compound,
including both rain sets and safe slicks. It compares stopping now with running
at least one more lap before any stop, including later paid refits and their tyre
ageing. It therefore can wait
for slicks instead of buying a short intermediate stint, or move to slicks
before the retained rain set becomes critical. Elective opening-lap stops remain
disabled; fitting an unused rain set does not itself earn a dry-use exemption.

The current rain-compound recommendation does not exclude a safe alternative.
For example, an intermediate can cost less than a wet set during projected
drying, despite being suboptimal on the observed surface. With a nonempty
prescribed rainfall schedule, the same search can avoid an extra short stint
before a known rainfall change. Pace is priced at the projected rejoin surface
after expected service and queue delays;
future critical mismatches still end retention. The selected automatic proposal
takes precedence over the current recommendation when the tyres are fitted.
Explicit safe pit instructions and physical inventory selection retain their
existing priority. Chronological finish protection uses the selected compound,
or the same safe candidate set when the choice is still unresolved.

Without a prescribed schedule, these decisions still assume the observed rain
persists and replan after each lap. They do not forecast random atmosphere
changes. Unlimited-stock opening selection compares the complete existing
policies for every safe opening; it does not establish a globally optimal
wet-race policy. Existing saved runs replay with the current
strategy implementation, so wet-race tyre choices can differ after this change.

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
Without an observed dry control prefix, current aero eligibility and SC/VSC
running factors apply only to the current lap; future laps assume green running.
The prefix described above instead applies those conditions at each projected
entry until known control ends. Both floor-clipped and control-prefix costs are
absolute lap times, whereas the ordinary fast path reports tyre-relative costs. Costs
from those two bases should not be compared across different model inputs.
The unlimited-set planners described above do not constrain physical inventory.
Drivers with an explicit pool use the separate [finite-set policy](tyre-inventory.md)
for opening selection and later decisions. Neither policy forecasts random
future weather changes or incidents or jointly schedules both teammates' future
stops. Green forecasts price traffic only on the immediate rejoin lap; the dry
control prefix also preserves its observed queue and entry gaps. Cost tables are bounded
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

Native timed compound-changing searches reuse safe candidates and criticality
for each observed surface within a decision. After actual rain-tyre running
has earned the wet exemption, prior slick identities no longer distinguish
future legality states; paid-stop counts, tyre ages, compounds, allowances,
clock flags and fitting delays remain separate. Clean future laps use the same
guarded deterministic evaluator and scalar lap cache as the direct rain planner.
Current-lap control and traffic still use their ordinary calculation. Patched
helpers, model extensions and custom clocks retain the uncached dispatch path.

Without fitting delays, native searches merge clock branches only when their
complete remaining before-fit and after-fit update paths agree. Every reachable
paid count remains represented, including compulsory fits beyond the elective
allowance. Fresh-fit costs are independent of the removed set's age once its
eligibility is known, so they are calculated once per matching future state.
Bounded shared caches can reuse refit and retained-stint scalar costs when the
entire remaining surface graph, physical driver/car/track, full retained and
fresh tyre parameters, retained age, budgets and actual-use history agree.
Different raw clock times can share only through that
exact future equivalence; current control, traffic and queue prices are computed
separately. Cache eviction changes work performed, not the available schedules.
The refit and retained-stint pools share a 65,536-entry bound, reserving one
eighth for reusable fresh-fit forecasts during retained-stint churn. Within a
decision, immutable native clocks reuse their validated update counts between
graph construction and running branches, and equal graph nodes are interned
once. Custom clocks and fitting-delay paths keep their original dispatch.
No models or evaluator closures enter these shared caches. Long horizons retain
the complete iterative search when recursion capacity is insufficient, and
cancellation releases each decision's local graph.

If every compound stays noncritical across all reachable surface updates,
future fits are bounded by the remaining elective allowance. An incomplete
actual-use history after a running lap can require one extra fit to an unused
slick or rain set. Shared graphs include both bounds and select the applicable
one from actual use; an opening fit before any race use is also accounted for.
This allows equivalent forecasts to share without comparing impossible extra
stops. Any possible future critical mismatch keeps the complete compulsory-fit
graph. Neither path shortens the planning distance or drops an eligible fit.

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

The same diagnostic accepts `--tire-warmup "soft=0.5,medium=0.5,hard=0.5"`
to include optional post-fit costs in both the adaptive and fixed schedules.
Enabled output records the normalized profile and its policy. Omitting the
option retains the original comparison. These are user-specified sensitivity
assumptions, not measured tyre-temperature coefficients.

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
vary across the grid. By default, rainfall and weather condition remain fixed
while surface wetness evolves; ordinary race incidents remain enabled. Set
`--change-probability` to a value from zero through one to allow stochastic
weather changes using the same seeded weather model as a normal race. The
scenario then describes the initial weather, not the whole race.

```powershell
python examples/benchmark_strategy_planning.py --engine chronological --scenario steady_damp
python examples/benchmark_strategy_planning.py --engine standard --scenario drying
python examples/benchmark_strategy_planning.py --scenario wetting --trials 3
python examples/benchmark_strategy_planning.py --scenario rain_transition --trials 3
python examples/benchmark_strategy_planning.py --engine standard --scenario dry --drivers 22 --laps 58 --trials 3 --opening automatic --change-probability 0.2
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
finite or unlimited pools. These modes and the weather change probability are
recorded in benchmark version 3 JSON alongside the initial pools and overrides. See the
[finite-pool search and benchmark notes](tyre-inventory.md#search-and-benchmark).

Finite inventory forecasts prune replacement branches with an optimistic
completion cost that retains the current set's actual wear until its next
service, includes that pit entry, then relaxes later running and service choices.
The same bound supports own-lap and externally timed forecasts; timed branches
also retain their delayed weather path before that first future service.
Downward rounding protects nearly tied alternatives, and the calculation makes
no assumption that older tyres are slower. It changes search work rather than
physical-set availability, stop allowances or the strategy objective.

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

Same-compound rain forecasts on an external weather clock reuse one projected
surface path within each decision. Native lap physics prepares the fixed
driver, car and circuit terms once, then reuses immutable running costs for
each tyre age, fuel lap and exact weather-update count. Paid stops and fitting
fees retain their delayed weather cadence; current traffic, control and aero
adjustments still use the ordinary first-lap calculation. Every allowed future
stop schedule remains in the search.

Native future green-lap costs also share a process-local 65,536-entry LRU across
decisions. Its keys contain the complete normalized driver, car and circuit
snapshots, original physical fuel distance, complete tyre and projected weather
values, lap number and tyre age. Values are immutable floats; the cache retains
no live models or evaluators. Current traffic, control, aero and fitting costs
remain outside this lookup. Cancellation is checked before cached laps too.
Access is synchronized, and a fork starts with independent empty storage and
a new lock. Custom evaluators or changed native physics helpers bypass this
shared cache, so cached native results cannot hide their behavior.

Cached green stint rows also prepare those fixed physics terms once when a row
is first evaluated. Their existing cache keys, size bound, surface isolation and
arithmetic remain unchanged. Custom physics retains its public evaluator, and
custom weather clocks keep the existing row-projection path. Forecasting does
not sample randomness in either path.

The same native eligibility contract applies to dry pace/floor tables, rain
stint and transition results, projected surfaces, weather-stop bounds and opening
policy scores. Model methods and field descriptors are compared with references
recorded when their defining modules finish loading; lap and forecast helpers
also retain their defining-module originals. Replacements installed before or
after a consumer import bypass shared caches. A callable identity alone cannot
make an external-state-dependent extension safe to cache.
The outermost decision checks definitions once; nested forecasts reuse its
class verdicts while checking instance hooks and nested track objects. Static
class/MRO dictionary comparisons never execute descriptors. Later decisions
check definitions afresh. Projection calibration, expected-service helpers and
opening candidate/seed constants also participate in native eligibility.

Copy methods and Pydantic state descriptors also participate in native
eligibility. Replacement behavior is checked afresh at later decisions, so
previously cached projections cannot hide a changed copy method.

Each non-native decision owns deep copies of its actual model objects and uses
public deterministic physics. Its private snapshot references distinguish
objects with identical serialized values and preserve subclasses and callable
instance hooks without serializing those hooks. These references are discarded
on return, cancellation or an exception. Local search memoization remains valid
within that decision. Untimed surface projection continues to normalize inputs
to the base Weather schema; clocked fallback laps retain their observed actual
surface objects. Equilibrium and dry-floor cadence shortcuts require unchanged
native surface physics.

Stop eligibility is also computed once per compound and remaining stop-budget
combination within a transition plan, then reused across its future-lap search.
These rows stay local to the plan's forecast. Critical-tyre exceptions, dry and
damp allowances, and the distinction between intermediates and wets are retained.

A local comparison against revision `600bfe6` used the standard engine, 22
drivers, 58 laps, automatic openings, unlimited tyres and seeds 42–44. Three
alternating fresh-process pairs per weather setting reduced median simulation
time from 5.463 to 4.995 seconds (8.6%) when starting dry with weather change
probability 0.2. Fixed dry weather was effectively unchanged (0.819 versus 0.814
seconds). These timings exclude process setup and provider loading; they are
synthetic workload measurements, not a general speed guarantee. Outcome hashes
matched for every timed pair and 120 additional short-race cases spanning both
engines, five weather patterns, finite and unlimited inventories, explicit and
automatic openings, evolving weather, and custom pit plans.

## Active Aero modeling limits

Race execution, deterministic strategy forecasts and qualifying use one scalar
gain for configured Active Aero zones, with the same circuit and car
effectiveness calculation. Green wet and dry laps use that gain before the shared
weather multiplier; SC, VSC and red flags disable it in racing. Qualifying assumes
configured Straight Mode is available throughout its fixed-weather sessions.
Direct callers can disable the qualifying gain with
`calculate_qualifying_lap(..., active_aero_enabled=False)` for controlled comparisons.
Live venue profiles provide zone counts and assumed gains, without official
activation-zone maps or a separate qualifying calibration.

[FIA 2026 sporting regulations B1.5.11 and B7.1](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf)
allow only front-wing activation in designated low-grip zones after race control
declares Low Grip Conditions. The simulator does not represent that declaration,
partial wing activation or those separate zones. Weather-dependent aero gains
would need explicit zone data and defensible effect estimates; the configured
scalar gains do not establish those values.
