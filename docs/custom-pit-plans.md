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

Instructions accept only `lap` and `compound`. Boolean, fractional and string
lap values are rejected. Driver IDs are case-sensitive and must belong to the
loaded roster. A requested compound must exist in that driver's finite input
pool when one is supplied. Validation does not guarantee that the set will
remain available after incidents or earlier use.

Opening tyres remain a separate input. Without an opening override, the
existing automatic opening policy is used; it does not optimize against the
custom future schedule. Supply an opening explicitly when testing a complete
tyre sequence.

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
causes that instruction to be skipped. It is not silently deferred.

A free red-flag tyre change does not consume a scheduled paid-stop instruction.
Earlier repairs and weather stops likewise do not consume future instructions.
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
Snapshots with custom plans use schema 5; older supported snapshots retain
their original automatic-policy interpretation.
