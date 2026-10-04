# Finite race tyre pools

An optional `tire_inventory` constrains each listed driver to named physical
sets throughout the race. The pool includes the opening set. Unlisted drivers
keep the existing unlimited-set policy. Qualifying uses independent sets;
the simulator does not infer real weekend allocations or qualifying usage.

## Inputs

The dashboard's **Edit tyre setup** editor lets you choose each driver's opening
compound and prior laps, then optionally list the available race sets. Each row
represents one physical set; identical compounds and ages can appear more than
once. Automatic opening selection chooses from the supplied pool. An explicit
opening must match a listed compound and age. Apply the draft to update the run
settings, or cancel to keep the previous settings.

You can also use the dashboard's **Race tyre sets (optional)** field or the CLI:

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
Only `id`, `compound`, `age` and optional `remaining_laps` are accepted in input records.
Unknown drivers
are rejected. An explicit `starting_tires`/`starting_tire_ages` choice must have
an exact compound-and-age match in that driver's pool. No replacement set is
fabricated to satisfy an override or a later strategy decision.

## Optional physical-set usage limits

Supply `remaining_laps` to limit the completed race laps a physical set can
still cover. Omission or `null` leaves that set unrestricted. The allowance is
independent of prior wear: `{"compound": "hard", "age": 5, "remaining_laps": 20}`
starts with five laps of wear and permits twenty more race laps. The CLI and
dashboard shorthand is `hard@5/20`; `hard/25` permits twenty-five laps on a
fresh set. The editor's **Race laps left** field accepts the same allowance;
leave it blank for unrestricted usage. Allowances must be integers 0–1000.
Zero makes a spare ineligible, and an opening selection must have at least
one permitted lap. A pool containing only exhausted sets is rejected.

This supports scenarios such as Pirelli's
[2025 Qatar cumulative per-set restriction](https://press.pirelli.com/a-maximum-of-25-laps-per-tyre-set-in-qatar/).
That prescription counted previous weekend running and SC/VSC laps, and teams
received each set's remaining allowance before the race. Enter the remaining
race allowance yourself; the simulator does not reconstruct practice, sprint
or qualifying tyre usage, or automatically apply a historical limit to another
event or season.

Both engines consume one allowance lap for each accepted own-lap crossing,
including SC/VSC running. Failed retirement crossings, pit service and red-flag
pauses consume none. Removing, retaining during a suspension, or refitting a
used set never resets its allowance. These are the simulator's completed-lap
and pre-lap pit-entry conventions, not a model of partial-lap regulatory
counting, formation laps or individual tyres mixed between sets.

Expiry requires a replacement before the next own lap, even with zero elective
stops left or an empty custom plan. A compulsory expiry stop is recorded as
`tyre_usage_limit`; it remains a paid visit and reduces remaining elective stop
budgets. Free red-flag changes still require usable stock. If no suitable set
remains, the existing exhausted-pool DNF applies before service or running.
Dry, wet, changing-weather and observed SC/VSC forecasts preserve each set's
allowance through exchanges and handoffs. Sets with equal compound and wear
but different allowances are distinct planning choices. Forecast pruning may
relax lifetime constraints optimistically, but executable paths cannot exceed
them. A retained-distance finish veto is bypassed when its assumption of no
further compulsory service cannot safely cover a usage-limited set.

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
On dry, damp or wet surfaces, an observed SC/VSC field also prices up to six
known control intervals, entry gaps, discounted lane loss,
expected service and current queue. Fitting fees remain unscaled. Each branch
returns removed sets at their accumulated wear and earns compound credit only
after running. A chronological flag during that prefix makes completed laps
take priority over elapsed time. Once control and pending barriers clear, the
same physical-pool search prices the fixed remaining own-lap green horizon,
rebased to its projected surface and leading weather updates. Surface drainage,
fixed rainfall and prescribed rain changes apply during the prefix. Missing
observations, longer custom controls, custom plans and free red-flag fits retain
their existing forecasts.

For zero-rain, zero-wetness decisions, native search bounds can temporarily
relax replacement stock to fresh sets
and ignore compound obligations, with an extra correction stop allowed when
needed. They also cap the maximum pit-lane saving under control. These are
optimistic costs used only to discard slower full-distance continuations;
every executable choice still comes from the supplied pool with its actual
age, availability and allowances. Custom model hooks retain isolated actual
models and bypass these native bounds and field-state memoization.

Weather-control decisions reuse at most 65,536 native scalar lap costs within
one decision. Their green suffixes can share an optimistic original-stock wear
bound that ignores earlier use, stop losses and weather chronology. Neither
optimization supplies executable sets or changes actual weather and wear;
custom model hooks bypass both. Comparing branch-dependent entry weather adds
work, especially during an early long control period.

Finite-pool planning adds work for each distinct driver, set age and forecast.
Automatic opening selection also compares complete policy paths. Start with a
small trial count when checking a new pool before running a large ensemble.

With a custom plan, equal-distance opening paths prefer more requests actually
executed before comparing time. A currently fitted set cannot satisfy a later
paid request for that same physical set; another set of the requested compound
must be available. Automatic openings can therefore reserve the sole requested
wet set by starting on intermediates. Two equivalent wet sets can instead allow
a wet opening followed by the requested wet refit. Explicit openings remain
authoritative, including plans whose later requests must be skipped.

A forecasted legal finish takes priority over an incomplete opening policy.
When every eligible opening predicts retirement, the selector compares the
accepted race laps before retirement, then elapsed time at the last crossing.
It no longer treats all those choices as an input-order tie. Failed paths keep
infinite completion cost and earn no custom-request ranking credit; the finite
retirement time is separate comparison evidence, not a finishing time.
For example, an eight-lap drying scenario with normalized surface water `0.2`,
no rain, four permitted laps on intermediates and one on wets reaches five laps
by using the wet set first, versus four by starting on intermediates. Both
policies still retire when no safe permitted set remains.

Run `python examples/check_weather_openings.py --incomplete` to compare each
automatic opening with separately executed alternatives in both engines.
The diagnostic includes that drying case and an equal-distance dry retirement
where the fresher soft set is faster, with automatic and empty custom plans.
It records finite last-crossing times and DNF status. These conditional checks
use mean physics, expected service and no incidents; they do not establish a
globally optimal survival schedule or forecast an actual retirement.

Automatic finite-pool decisions retain partial continuations throughout the
elective search as well. A legal projected finish takes priority; if every
continuation retires, the planner prefers more accepted laps, then less elapsed
time at the last crossing. This can require an early paid switch to use a set
while the weather still permits it. Removed sets retain their actual wear and
remaining allowance, and ordinary stop budgets still govern elective visits.
An illegal final dry crossing earns no distance. Failed completion costs remain
infinite, with partial elapsed time stored separately; a survival decision does
not produce a claimed finishing-time saving.

The transition from observed SC/VSC running to a green forecast preserves
every crossing already accepted in the controlled prefix. If there is no safe,
usable or legal next action, retirement starts at that boundary; an unavailable
green action cannot erase the last completed lap. For example, a one-lap hard
opening followed by five controlled laps across finite medium and intermediate
sets still records all six laps when the stock runs out at the restart.
Its finishing cost remains infinite, while its last-crossing time stays available
for comparing incomplete strategies.

Run `python examples/check_inventory_continuations.py --elective` for an
eight-lap drying race starting at surface water `0.24`, with four permitted
laps on an explicit intermediate opening and one on wets. Switching to wets
before lap two, then refitting the removed intermediates before lap three,
completes five laps. Deferring the wet window retires after four. Both engines
compare independently committed alternatives using mean physics, expected
service and no incidents. These are conditional forecast and execution checks;
later weather, traffic and control observations still trigger replanning.
An empty custom plan intentionally keeps the four-lap path, while an executable
wet request at lap two permits five laps.

When no usable automatic continuation is available, required stops and free
red-flag fits compare compulsory continuations through retirement. Legal
projected finishes take priority; failed paths compare accepted distance, then elapsed time at the
last crossing, without custom-request ranking credit. For automatic strategy,
this fallback inserts only required weather, usage-limit and compound
corrections. A custom plan keeps its remaining executable requests. Neither
fallback searches future profitable elective stops. Later decisions still
replan from the observed race, so this is conditional survival evidence rather
than a claim of globally optimal distance.

The continuation preserves physical identities, wear, usage allowances and
pending fitting costs. Paid starts include expected lane, service and queue
delay in their weather clock; free fits add no paid delay. Eligibility uses
the observed commitment surface. First-lap costs retain current control and
the supplied rejoin/stay traffic gap, with green clean-air running afterward.
Fuel uses the original scheduled distance. Failed completion costs remain
infinite, separate from finite retirement times. If no candidate has a usable
positive-distance forecast, execution keeps the existing immediate safety
choice or retires before service when no eligible legal replacement remains.
Candidate physics uses copied models and consumes no live RNG or service draw.
Native compulsory continuations evaluate one representative of equal compound,
wear and expiry at each decision, retaining every physical copy for later use
and input order for ties. Custom physics keeps separate alternatives. This
avoids enumerating permutations of interchangeable stock in an exhausted pool.

Run `python examples/check_inventory_continuations.py` for a controlled
eight-lap drying race starting at surface water `0.24`: one permitted lap on
hard tyres, four on intermediates and one on wets. After the hard set expires,
using wets first preserves six accepted laps in total; fitting intermediates
first retires after five. The diagnostic independently commits each eligible
replacement, repeats with a free red-flag fit and custom plans in both engines,
and checks distance and time against actual later execution. All these paths
still retire; the extra lap does not turn a DNF into a finish.

Red-flag fittings are free changes and can retain the current physical set.
They consume no paid stop. The physical-set ledger includes opening and later
fittings that never complete a lap; these do not earn compound-use credit.

With a [custom pit plan](custom-pit-plans.md), compulsory paid replacements and
free refits follow that remaining schedule instead of pricing future elective
automatic stops. The continuation keeps every physical identity and its wear,
so a fresh set can be reserved for a later requested compound and a removed
set can be reused. Skipped requests, unavailable sets and required corrections
remain part of the comparison. A leader's conditional timed finish can leave
later requests unrun, while followers retain their existing estimated horizon.
Keeping the fitted set preserves both its age and any pending first-lap cost.

## Search and benchmark

Future planning groups interchangeable physical IDs by compound, age and usage
expiry while retaining their number. Each set accumulates wear when it runs,
including after removal and reuse. Search skips a replacement branch with an optimistic
remaining-time bound only after finding a legal completion that the branch
cannot improve. A bound on full-distance time cannot discard a longer partial
continuation while all explored paths still retire.
Native prescribed-weather searches carry the best known finishing time through descendant branches.
They keep exact continuations separately from bounds on excluded continuations:
an excluded suffix must be searched again when another prefix leaves it more time.
Bounds round downward, and incoming cutoffs round upward to preserve near ties.
Admitted candidates preserve usable stock, its wear and physical multiplicity.

Native forecasts also share otherwise identical continuations after the active
set exhausts its usage allowance. That set cannot run or return to the future
pool, so its last compound and wear no longer distinguish the suffix. Usable
stock, stop allowances and elapsed fitting costs remain part of the decision.
Equivalent satisfied compound credit shares a suffix, while each unsatisfied
single-slick obligation remains distinct. At the final boundary only compound
compliance remains relevant. The race ledger preserves actual set identities.
The completion bound charges the next compulsory service instead of pricing
further running on an exhausted set. Custom physics retains its actual model
state and bypasses this reduction.

Native green forecasts also omit stock that is critically mismatched at every
remaining possible entry. A currently unsafe compound stays in the pool if
later weather can permit it. Externally timed forecasts check every entry and
service surface admitted by their clock relaxation before removing stock.
They also share weather-clock states after all remaining complete projected
weather snapshots become identical, including after a prescribed schedule.
Paid service and fitting costs still enter elapsed time. Future weather changes,
custom clock subclasses and custom physics retain their distinct branches.
Observed SC/VSC field state still distinguishes controlled continuations.

`python examples/benchmark_inventory_retirement.py --sets 18 --trials 1`
measures a conditional decision with distinct used intermediate sets, one
permitted lap on each, constant surface water and rainfall `0.3`, and a longer
scheduled distance. Both own-lap and external clocks must preserve 18 accepted
laps followed by retirement. The JSON reports times, proposed replacement and
decision, plus an outcome digest that excludes timings. This is a synthetic
search workload, not a prediction of race retirement or an executed race.

The conserved-wear bound starts with the cheapest reachable running cost on each future lap.
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

A second bound keeps the fitted set's actual age until its first future service.
It compares retaining that set to the finish, when compliant, with paying a green
pit entry and then using the optimistic running-cost bound above. This avoids
expanding branches whose apparent gain would require an unpaid replacement.
It ignores replacement availability, eligibility and subsequent service or
warm-up charges, so it remains a lower bound rather than an executable plan.
Own-lap forecasts reuse these costs only within the same decision, for the same
compound, age trajectory and compound-rule completion status. The full search
still enforces physical sets, stop allowances and actual compound use.

For prescribed forecasts, native own-lap searches also use a
relaxation with unlimited fresh replacements. Each idealized stint still ages,
pays its pit visit and any configured fitting fee. Externally timed searches
use this relaxation when fitting fees are disabled, retaining the delayed weather path.
Native accumulated
wear and the fading soft-tyre pace bonus cannot improve an older set over a
fresh copy. Ignoring finite availability, stop allowances and compound obligations
therefore yields another completion bound. It prices later service visits that
the conserved-wear bound omits; neither relaxation supplies an executable plan.

Within one native SC/VSC field decision, own-lap and externally timed green continuations
of up to 100 own laps can share completed fresh-service costs across different
physical-pool histories. Reuse requires the same native physics, fuel laps,
complete starting weather, prescribed forecast and reachable paid-stop weather
observations. The retained set's age and usage expiry stay in its own bound;
the exact physical search still enforces stock, allowances and compound rules.
At most 32 forecast tables remain in the decision's LRU, which resets on return
or cancellation. Externally timed fitting fees, current traffic or control adjustments,
changed dispatch and longer horizons keep independent completion costs.
Own-lap tables retain their exact cumulative weather cadence and prescribed
schedule, with distinct fitting profiles kept separate. A ready retained set
pays no fitting fee, including at age zero. Native running costs also share identical compound, age, fuel lap,
complete surface, traffic and aero inputs within the decision's bounded lap
memo. Actual custom weather objects and instance hooks bypass that memo,
including when their serialized fields match a previously evaluated surface.

For externally timed chronological weather, each paid stop also advances the
candidate's weather clock. The running-cost bound considers the reachable
delayed surfaces, including compulsory stops after the elective allowance is
exhausted. The first-service bound additionally retains the current set's exact
delayed surface timeline until that stop. Taking the stronger bound avoids expanding
unnecessary full-distance stint combinations without approximating tyre ages,
weather timing or the selected strategy. Free restart fits add no elapsed pit
time; future paid stops still advance their weather forecast. With native clocks
and at most 100 own laps, fitting fees widen each entry's possible surfaces only
by the fees that previous own laps could have accumulated. This includes pending
and free initial fits; the current entry receives no new fitting fee. Every
intermediate weather count remains available to the bound. Custom clocks and
longer fee-bearing forecasts retain the full surface envelope. Native callers
reserve the dry and rain stop envelopes for those delayed branches while the
planner continues to enforce the damp or dry allowance at the surface where
each stop occurs.

The timed-weather search retains each action's completion bound when sorting
candidates and reuses that exact value for pruning. Bounds remain local to the
same forecast; the ordering key and downward-rounded pruning comparison are
unchanged.
Tyre-safety checks are also reused for each compound and projected weather
update within that forecast. The cache is discarded with the planning call,
so later race conditions are evaluated afresh.

Finite-pool forecasts prepare the deterministic lap evaluator once for the
fixed driver, car, track and physical fuel distance, with either an external
leading weather clock or the supplied own-lap surface intervals. The prepared
path keeps tyre age, projected weather, fuel lap, traffic gap and active-aero
availability as explicit inputs, and shares the final arithmetic composition
with ordinary lap calculation. Custom simulator methods, overridden car pace or
track active-aero calculations (including hooks replaced before importing the
lap simulator), and driver, car or track subclasses keep the public evaluator.
Custom tyre or projected weather
objects also use its fallback; surface projection retains its existing input
normalization. The shortcut preserves extension dispatch and seeded race behavior.

Fixed field descriptors, custom attribute dispatch and nested sector/active-aero
zone behavior are checked without executing custom getters. Their references
come from the model definitions, so pre-import replacements also retain public
lap dispatch. Unchanged native tyre and weather leaf methods remain dynamically
invoked by the prepared evaluator. Shared forecast caches require native model
and helper behavior; custom decisions keep isolated actual models and only local
memoization. Equal schema values do not merge distinct custom tyre instances.
Untimed projected surfaces intentionally use the base Weather schema, including
when the caller supplies a Weather subclass.

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

A 2026-09-21 check isolated deferred replacement-state construction against
the planner at `f2e83d9`. With `--engine chronological --scenario wetting
--inventory finite --opening automatic --drivers 2 --laps 53 --trials 1
--seed 42`, three alternating fresh-process pairs took 29.43/24.64,
29.35/24.67 and 29.59/25.42 seconds before/after. Median time fell by about
16%; all six outcome hashes matched (`7df7020aca25`). Twenty shorter cases
covering both engines, five weather patterns and explicit/automatic openings
also retained identical hashes. This measures one allocation optimization on
these workloads, not a general runtime guarantee or a change in strategy.

The same day's timed-weather bound-reuse check used four drivers and explicit
openings with the other settings above. Holding deferred state construction
enabled in both versions, three fresh-process pairs took 8.39/8.10,
7.89/7.70 and 8.06/7.97 seconds. All six hashes matched (`5ce6e5126dc8`).
The measured gain was smaller (about 1–3% per pair); profiling confirmed that
completion-bound calls fell from 1,058,663 to 549,081 while the number of
expanded action lists stayed at 183,027. The search itself is unchanged.

A 2026-09-22 check compared prepared lap evaluation against `00e4f27` in
alternating fresh processes. With chronological wetting weather, finite pools,
four drivers, explicit openings, 53 laps and seed 42, three before/after pairs
took 6.72/5.97, 6.67/6.05 and 6.66/5.97 seconds. Median runtime fell by about
10.5%; all six complete outcome hashes matched (`5ce6e5126dc8`). A single
two-driver automatic-opening pair took 22.07/21.32 seconds with matching
outcomes. Another 24 shorter cases covered both engines, all five benchmark
weather patterns, both opening modes and additional seeds without an outcome
change. These are local workload measurements, not a general speed guarantee;
the optimization changes neither search allowances nor cancellation checks.

A 2026-10-02 check compared untimed preparation with the inventory planner from
`72fccc8`, using shared native lap physics with the custom-hook guards above.
Three alternating fresh-process pairs used chronological wetting weather,
finite pools, automatic openings, two drivers and 53 laps, running seeds 42
then 43 in each process. Median first-trial time fell from 19.828 to 18.263
seconds (about 8%); the second-trial median fell from 3.388 to 3.173 seconds
(about 6%), with variation across pairs. Every complete outcome digest matched
(`30397bd0a312`). A smaller Standard finite-pool case also retained its exact
outputs. These measurements describe that synthetic workload; they do not
establish a full-grid runtime bound or empirical strategy accuracy.

A 2026-10-03 check compared the own-lap first-service bound against `eccf38f`.
Three alternating fresh-process pairs used chronological wetting weather,
finite pools, automatic openings, two drivers, 53 laps and seed 42. Median
simulation time fell from 17.560 to 5.511 seconds (about 69%); all six complete
outcome hashes matched (`96dd3580b5f3`). The benchmark includes own-lap forecasts
while comparing opening sets. These are local synthetic-workload measurements,
not a full-grid speed guarantee or evidence of real-world strategy accuracy.

## Saved inputs and audit records

The input pool is saved separately from the final pool, in request/scenario
provenance and replay inputs. A nonempty finite pool uses snapshot schema 4,
schema 5 with custom pit plans, or schema 6 with post-fit cost sensitivity;
older supported snapshots retain their existing unlimited-set interpretation.
Unsupported schemas are rejected instead of silently dropping the constraint.
Any explicit usage allowance selects schema 9 and
`tire_usage_policy="completed_race_laps_v1"`, retaining other configured weather,
qualifying, warmup and custom-plan inputs. Replay and paired comparisons reject
usage limits in older schemas and unknown or missing usage policies.

Finite race results include `tire_set_history` records with `lap`, `kind`
(`start`, `pit`, `red_flag`), `set_id`, `compound`, `age_at_fit`, `age_at_end`
and `laps_used`. The result's `tire_inventory` is the final snapshot, containing
`id`, `compound`, `age`, `current`, `available` and `unavailable`. Do not pass
that final snapshot directly as an input pool: its extra state flags are not
input fields. Unlimited or legacy results have no finite ledger.
Wear and `laps_used` count completed laps; an interrupted retirement lap is
not credited as a full lap. A paid fitting on that lap remains in the ledger.
Limited sets additionally record `remaining_laps` in the final pool and
`remaining_laps_at_fit`/`remaining_laps_at_end` in their fitting history.
An exhausted undamaged set has zero allowance and `available=false`, while
`unavailable` continues to identify damage. Reports show unrestricted sets as
**Unlimited** when they share a table with limited sets.

Dashboard and HTML reports display both fittings and final pools. JSON and
race CSV exports preserve the records, while paid-stop details additionally
identify outgoing/incoming sets and incoming wear. Compound strategy sequences
and paid-stop totals remain separate views: a free fitting is not a paid stop.
