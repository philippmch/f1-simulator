# Custom pit plans

A custom plan lets you test deliberate paid stops against the automatic
strategy. It controls when a listed driver stops and which compound is
requested. It does not predict future incidents or weather, and it does not
claim that the chosen schedule is optimal.

## Entering a plan

The dashboard's custom-plan field and the race CLI use the same shorthand:

```text
VER=18:medium,36:hard;NOR=24:hard
```

Each number is the driver's **own lap at pit entry**. A stop at `18` takes
place before the driver runs lap 18, after 17 completed laps. In a chronological
race this can differ from the leader's lap. Stops must be ordered, have distinct
integer lap numbers from 2 through the original scheduled race distance, and
use canonical compounds: `soft`, `medium`, `hard`, `intermediate`, or `wet`.
Each driver may have at most 20 instructions.

Use `NOR=none` to request no elective stops. Omit a driver to retain the
automatic policy. A blank field leaves everyone automatic. These are different
choices: a driver with an explicit empty plan still makes compulsory repairs
or corrections, but does not make an elective automatic stop.

```powershell
python examples/simulate_race.py --race 1 --starting-tyres "VER=soft" --pit-plans "VER=18:medium,36:hard" --export
```

The Python runner and HTTP `/api/run` accept the equivalent `pit_plans` mapping:

```python
pit_plans = {
    "VER": [
        {"lap": 18, "compound": "medium"},
        {"lap": 36, "compound": "hard"},
    ],
    "NOR": [],
}
```

Fixed instructions accept `lap` and `compound`. A control window also requires
`earliest_lap` and `trigger`, together. Boolean, fractional and string
lap values are rejected. Driver IDs are case-sensitive and must belong to the
loaded roster. A requested compound must exist in that driver's finite input
pool when one is supplied. Validation does not guarantee that the set will
remain available after incidents or earlier use.

## Safety-car and VSC windows

Use a window to take a requested stop under observed neutralization, while
retaining a deadline if that opportunity does not arise:

```text
VER=18-25@sc:hard,40:medium
NOR=20-28@vsc:hard
PIA=18-25@neutralized:hard
```

The first request tries to fit hards at the first safe safety-car pit entry
on VER's own laps 18 through 24. Otherwise it attempts the stop at lap 25 under
any control. `@vsc` responds only to VSC; `@neutralized` responds to either a
safety car or VSC. An already active deployment can trigger a stop at the start
of the window. Green running and a different control type do not trigger an
early stop. Red-flag refits remain separate free fits.

The equivalent JSON instruction is:

```json
{"lap": 25, "compound": "hard", "earliest_lap": 18, "trigger": "safety_car"}
```

JSON triggers are `safety_car`, `vsc` and `neutralized`. The earliest lap must
be at least 2 and strictly precede the deadline. Each following instruction or
window must start after the previous deadline, even if a request actually
executes earlier. The deadline cannot exceed the original scheduled distance.

An unsafe or unavailable early requested fit leaves the window pending. A
compulsory repair can proceed without consuming that request when it cannot
use the requested compound. At the deadline, the ordinary custom-stop rules
apply: the request can execute, be overridden, or be skipped. A retirement or
shortened finish can still leave it not reached. Executing an early request
consumes it once; it does not cause another stop at the deadline.

Opening and replacement forecasts use the window deadline for unobserved
future control. They do not sample or predict future deployments. A paid
service already fulfilling the current early request removes its deadline
from that continuation. The observed control still prices the current paid
service using the existing pit-lane and queue model.

Window histories retain `earliest_lap` and `trigger`, plus `actual_lap` for a
committed service. That lap is null when no service occurred. Aggregate
`service_laps` lists the recorded lap and count of each paid service fulfilling
or overriding the window, using complete valid histories only. Reports show
these laps alongside status counts; the counts do not measure strategy quality.

Saved window inputs use schema 10 with
`pit_plan_policy="neutralized_window_deadline_v1"`. Replay and comparisons
reject window fields in older schemas and reject unsupported policies. Existing
fixed plans retain their earlier input schemas and history shape. Windows work
with the standard and chronological engines, finite physical sets, warmup,
qualifying weather and prescribed race rainfall.
An explicit [SC/VSC scenario schedule](control-schedules.md) uses schema 11,
retaining this window policy. It can provide repeatable observed opportunities;
the tyre forecasts do not know future control announcements.

## Opening tyres and forecasts

Opening tyres remain a separate input. Without an opening override, the
automatic selector compares opening choices while executing this driver's
custom plan in its isolated projections. An empty plan suppresses elective
stops in those comparisons too. Compulsory corrections, actual compound use,
finite-set wear and the timed race finish remain active. The selector compares
completed distance first, then the number of requests actually executed before
the finish, then elapsed time, using mean pace and expected service. Skipped,
overridden and unreached instructions earn no execution credit. This can reserve
a sole physical wet set for a later paid request by starting on intermediates,
even if starting on wets and skipping that request would be quicker.
The selector does not search for a different pit schedule or predict traffic
and incidents. A longer legal projected race still takes priority over more
executed requests at a shorter distance.
Unlisted drivers retain the automatic later policy. Supply an opening
explicitly when testing a complete tyre sequence; its compound and age remain
authoritative.

## Requested stops and actual execution

An executable custom stop is not cancelled merely because the automatic
planner predicts a faster alternative or a better finishing distance. Custom
plans can also request more stops than the automatic planner's bounded search
allowance. This makes deliberately poor strategies testable.

Existing compulsory behavior still applies: puncture repair, an unavailable
fitted set, a critical tyre/weather mismatch, and mandatory compound
correction. When a coincident instruction can satisfy that requirement, its
requested compound is used. Otherwise the compulsory policy chooses the
replacement and records the instruction as overridden.
The compound requirement counts tyres actually used on track. A free red-flag
refit that is replaced before running a lap does not satisfy it.
If keeping that fitted set satisfies the requirement, the rule can override
the request without another paid stop; the history then has no actual service
compound or set ID.

With a finite pool, an ordinary requested stop chooses the least-worn eligible
replacement of the requested compound, breaking equal-age ties in input order.
It cannot fabricate a set or select the currently fitted physical set. Without
a compulsory stop, an unavailable or critically mismatched requested compound
causes a fixed or deadline instruction to be skipped. Early window opportunities
remain pending until another eligible opportunity or their deadline.

A free red-flag tyre change does not consume a scheduled paid-stop instruction.
Earlier repairs and weather stops likewise do not consume future instructions.
Automatic compulsory replacements and free restart fits are priced against the
remaining custom instructions, including an empty plan. The forecast inserts
only critical-weather repairs, physical-set usage-limit replacements and
required compound corrections. It preserves
physical-set wear and availability, including sets removed and reused later,
and accounts for skipped requests and optional fitting costs. It does not add
profitable elective stops or change an executable requested compound.

The forecast uses expected service and mean pace, with current control on the
next lap and green running thereafter. Compulsory paid replacements keep the
observed rejoin gap and expected pit-box wait from the stop decision. That gap
enters the first lap's physics before minimum-time clipping and control scaling;
later laps assume clean air. A directly supplied free-fit forecast uses the
stay gap until a paid service is committed on that same lap. Suspension fits
without a traffic observation retain the clean-air assumption. Expected box
waits are separate from sampled service and actual queue loss, so a long random
service cannot change the forecast after the stop decision.
A projected leader's observed clock
can shorten each candidate at the lap following expiry; completed distance
is ranked first, then fulfilled instructions, then time. This prevents a cheap
replacement winning by making a later safe request unavailable when an equally
long continuation can honor it. This also works before an initial pace observation,
so a future request beyond that finish need not reserve a fresh physical set.
Other cars retain the engine's estimated own-lap finish horizon and shared
weather cadence. Original scheduled fuel distance remains unchanged. Future
incidents, traffic, leader changes and unprescribed weather remain unknown.

When every eligible finite-pool opening predicts retirement, opening selection
compares accepted distance and last-crossing time without giving failed paths
custom-request ranking credit. A forecasted legal finish retains priority.
See the [incomplete-opening diagnostic](tyre-inventory.md) for controlled
comparisons with actual execution.

Compulsory replacements and free fits also retain accepted distance and
last-crossing time when all remaining custom continuations retire. They keep
infinite completion cost and receive no request-ranking credit. A legal
projected finish, including a shorter timed finish, wins over a failed path.
The continuation still executes safe requests at their specified laps; a
requested fit is not replaced merely to preserve more distance. The
[`check_inventory_continuations.py` diagnostic](../examples/check_inventory_continuations.py)
checks paid and free fits against separately committed physical alternatives.

At a chronological red-flag restart, free fits use the collected field's
frozen finish horizons, weather clocks and projected-leader identity. Every
survivor completes that fit before any restart lap is released or new paid
service begins. A car entering the pits immediately after the restart cannot
change a later car's suspension-time tyre choice. Cars still held at a closed
pit exit follow the same fitting step without a second paid service. Subsequent
ordinary lap starts refresh the finish context from the live field.

Actual finish boundaries always apply: a retirement, shortened race, or lapped
finish can leave later instructions unrun. No pit service is created on a lap
that the driver never starts.

Results retain `pit_plan_history`, with one record per instruction containing
its requested `lap` and `compound`, `status`, `reason`, `actual_compound`, and
`actual_set_id`. The statuses are `executed`, `overridden`, `skipped`, and
`not_reached`. Actual tyre fields are null when no stop was committed; unlimited
pools have no physical set ID. An empty history means an explicit no-elective-
stop plan, while null means no custom plan was recorded for that driver.
Paid-stop totals and physical tyre ledgers remain separate records of what
actually happened.

The dashboard and exported reports also summarize instruction outcomes across
recorded trials. Each saved driver instruction has counts for executed,
overridden, skipped and not reached. Coverage shows valid, missing and invalid
histories separately; missing evidence never counts as an instruction that was
not reached. A duplicate driver row or incomplete, duplicated or mismatched
instruction history makes that driver-trial invalid rather than multiplying
its counts. Only complete valid histories contribute instruction outcomes.

The denominator is the number of recorded trials, not the requested simulation
count. Explicit empty plans retain their own history coverage and mean no
elective instructions, not no compulsory stops. Legacy results without saved
plan context remain unknown. JSON exports include these counts under
`pit_plan_statistics`; detailed per-trial histories remain available. These
frequencies describe how the model executed the requests, not strategy quality.

Large standalone HTML histories show one selected trial at a time. Use the
trial selector to inspect the remaining records; every history row is retained in
the offline report. The aggregate summary still covers all recorded trials,
and JSON and CSV exports retain their complete histories. This limits the
number of table rows rendered at once without truncating the audit trail.

## Comparing with automatic strategy in the dashboard

After entering custom plans, enable **Compare with automatic strategy** to run
both alternatives for each selected weather scenario. The main result views
continue to show the submitted plans. The reference removes every custom-plan
override, including explicit no-elective-stop plans, while retaining the same
drivers, cars, weather inputs, opening tyres, physical set pools and seed range.
Unlisted drivers keep their automatic policy in both alternatives; their race
outcomes can still change through traffic and interactions with other drivers.

Comparisons accept 10–500 trials per alternative, keeping total work within
the ordinary limit of 1,000 races per weather scenario. The paired results show
changes relative to automatic strategy, with recorded-pair counts and sampling
uncertainty. Distance and pit-cost comparisons use their own valid-data subsets.
Equal seeds do not freeze later incidents, battles or pit service.

**Match random draws** defaults to **Weather (default)**. Choosing **Weather and
mechanical checks** also gives each driver stable mechanical draws for each own
lap across the alternatives. Both variants use the selected policy and record
it in their saved inputs. Weather draws align by weather-update interval, not
elapsed seconds. Changed heat, risk or laps driven can still change failures;
other incidents, battles and pit service can also differ. This option does not
guarantee lower sampling uncertainty or isolate every effect of a strategy.

Choosing **Driver and purpose streams** additionally separates native
qualifying, lap variation, pit service, stochastic choices and passing draws
by driver and purpose, with a separate field event stream. An extra service
call cannot shift an unrelated car's pace noise. Draws still advance within
each purpose, so changed sampling opportunities and physical interactions
can change outcomes. Offline comparisons can select the same policy with
`--rng-policy isolated_race_v1`. See [random streams](random-streams.md).

Download the automatic reference separately to replay it using the existing
saved-input workflow. Both alternatives retain their own input snapshots; edits
to the form after a run do not change the results or downloaded inputs.

Use **Download strategy report** to save an HTML comparison of the custom plans
and automatic reference for the selected weather scenario. It includes paired
outcomes, pit costs, coverage and uncertainty from the saved run. Switching
weather selects that scenario's report; changing form inputs does not rewrite
it. The ordinary comparison report continues to compare weather scenarios.
JSON downloads retain replay inputs and statistics while omitting HTML reports.

## Comparing plans offline

Create a JSON file of named alternatives:

```json
{
  "automatic": null,
  "earlier": [{"lap": 18, "compound": "hard"}],
  "later": [{"lap": 25, "compound": "hard"}],
  "no-elective-stops": []
}
```

```powershell
python examples/compare_pit_plans.py output/saved_statistics.json --driver VER --plans plans.json --reference automatic --simulations 100 --export
```

The null alternative restores automatic strategy for the selected driver;
other drivers' saved plans remain in place. Every alternative retains the
saved roster, cars, track, weather, openings, tyre pools and seed range. All
plans are validated before trials begin, and the source file is unchanged.
The command accepts at most 10 named alternatives and writes output only with
`--export`. Exports include replayable inputs, requested-versus-executed plans,
and the existing paired points, retirement, distance and pit-cost comparisons.

Equal seeds do not freeze later incidents, battles or service times. Paired
changes describe modeled outcomes and sampling uncertainty, not isolated causal
effects or a proven real-race strategy. Replay uses the installed model code.
Snapshots with custom plans use schema 5, or schema 6 when
[post-fit cost sensitivity](tyre-wear.md#optional-post-fit-cost-sensitivity)
is enabled; older supported snapshots retain
their original automatic-policy interpretation.

### Comparing constructor plans

To compare coordinated alternatives for a whole constructor, use
`--constructor` instead of `--driver`. For a saved constructor `TEAM` with
modeled drivers `A` and `B`, a plans file can contain:

```json
{
  "automatic": null,
  "double-stack": {
    "A": [{"lap": 15, "compound": "hard"}],
    "B": [{"lap": 15, "compound": "hard"}]
  },
  "staggered": {
    "A": [{"lap": 14, "compound": "hard"}],
    "B": [{"lap": 16, "compound": "hard"}]
  }
}
```

```powershell
python examples/compare_pit_plans.py output/saved_statistics.json --constructor TEAM --plans team_plans.json --reference automatic --simulations 100 --export
```

Replace `TEAM`, `A` and `B` with exact IDs from the saved inputs. Each non-null
alternative must name every modeled member of that constructor and no rival
drivers. This prevents an omitted teammate from silently retaining a different
saved plan. A top-level null restores automatic strategy for all selected team
members; inside a member mapping, null restores that driver's automatic policy
and an empty list disables only that driver's elective stops. Rival constructors'
saved plans remain in place. A saved one-driver constructor is supported without
inventing an absent teammate.

All alternatives are validated before simulation starts, including lap bounds
and finite-pool requirements. The source file, openings, tyre pools, warm-up
assumptions and other saved physics remain unchanged. Results retain the common
seed range and expose both driver and constructor comparisons. Constructor-mode
exports include all modeled drivers, so effects on rivals remain visible.

### Paired constructor points

When choosing among multiple plans, the separate
[selection and validation workflow](strategy-selection.md) freezes the
training winner before comparing it with a fixed reference on fresh seeds.
It supports either a driver or a complete constructor as the points objective.

The dashboard, console and comparison export also show the change in total
points for each modeled constructor. This matters when a stop helps one driver
but costs a teammate time in the shared pit box. A four-point gain for one
driver and a five-point loss for the other is a one-point loss for the team.

For each matching seed, the comparison sums the points of every modeled runnable
driver in the constructor before computing the variant-minus-reference change.
The standard error uses those team differences directly, preserving correlation
between teammates. It is not the sum of the individual standard errors.

A team pair needs a valid result for every member in both runs and matching
qualifying. Missing, duplicate or invalid teammate results exclude the entire
team pair without removing valid individual-driver comparisons. Retirements
remain usable, including their classified points. Each team reports its own
paired and excluded counts; its mean need not equal a sum of driver means
computed from different subsets.

Member IDs identify the modeled team population. A simulation with one modeled
driver for a constructor reports that one driver's team contribution, without
inventing an absent teammate. Focusing a comparison on one driver retains the
full modeled team in the constructor total. Older reports without these
statistics do not imply zero team impact. Positive changes mean more points;
the results describe the chosen model and sampled trials, not a causal or
real-race guarantee.

### Paired elapsed race time

Comparisons also report elapsed-time changes in seconds, using only seed pairs
where the driver finished both races with the same positive completed-lap count.
Equal-distance lapped finishes are eligible; DNFs, unequal distances, and missing
or invalid times are excluded. This subset has its own paired and excluded counts,
mean reference and variant times, and standard error of the paired difference.
Negative variant-minus-reference differences mean faster. Fewer than two pairs
leave the standard error undefined; zero standard error does not prove certainty.

This is a conditional comparison among matching finishes, not an overall strategy
ranking: a risky plan can look fast in its surviving races. Check the paired points,
retirements, and completed distance alongside time. Adaptive race events can differ
even with matched seeds, so these differences do not isolate causal pit-plan savings
or establish real-race accuracy. Missing time in older records is not zero.
