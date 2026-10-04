# Selecting a pit plan across weather and rival strategies

The rival-strategy workflow selects one target pit plan across a set of
predeclared race-weather and opponent-plan scenarios, then evaluates that frozen choice on a
disjoint seed range. Choose the candidate plans, rival scenarios, and scenario
weights before looking at validation results. The supplied weights are explicit
analysis assumptions; the simulator does not infer that they are real-world
probabilities.

This measures outcomes under the saved simulator inputs. It does not establish
that a plan is optimal, calibrated to real races, or better under unmodeled
competitor strategies, weather, or model assumptions.

## Use the dashboard

Enable pit-plan candidate selection, choose a driver or constructor and the
selection objective, and enter
the candidate plans and training/validation trial counts. Open **Weather and rival
assumptions** and enable weighted scenarios. Add named scenarios with
positive relative weights, then add any rival-driver overrides. Drivers omitted
from a scenario inherit their saved plan; the controls also offer automatic,
no elective stops, and custom own-lap stops. Target members cannot be rivals.

Each case can inherit the source race weather or choose dry, cloudy, light rain
(rainfall and surface water 0.3), or heavy rain (both 0.8). The run's rainfall
mode applies to the initial weather. Enable **Override known rainfall steps**
to enter `LAP=RAIN[:CONDITION]` lines for that case, or leave its override empty
to clear the source schedule. An unchecked override inherits the source schedule.
Qualifying stays fixed to the source experiment across all cases. A single
target plan is selected across the cases for each source weather entry; selecting
several source weather entries still creates separate selection experiments.

Each weather has a maximum workload of 1,000 races, including source trials,
every candidate in every rival scenario for training, and both reference and
winner in every rival scenario for validation. The editor shows this worst-case
budget even when training later chooses the reference itself.

Completed results show the frozen rival assumptions and weights, the weighted
training choice, and fresh validation both in aggregate and by rival scenario.
Editing the form does not change completed evidence. Training and validation
JSON downloads use the existing replay format: each saved scenario is named by
a JSON pair of rival-scenario name and candidate label. The file's
`rival_scenario_index` lists these keys for the replay command's `--scenario`
option. Each entry retains its simulation inputs, engine and cohort metadata.
The full dashboard JSON also preserves the results grouped by rival scenario.
The HTML report includes the weighted and per-scenario
evidence. Existing race and statistics charts continue to describe the source
run, rather than a synthetic weighted race.

## Run a selection

Start with saved statistics or a saved comparison JSON containing simulation
inputs. Provide two to ten target candidate plans (including the reference) and
one to ten named rival scenarios:

```powershell
python examples/validate_rival_pit_plan_selection.py output/saved_statistics.json `
  --driver VER --plans plans.json --reference automatic `
  --rival-scenarios rival-scenarios.json `
  --training-simulations 100 --validation-simulations 100 --export
```

For a constructor target, replace `--driver VER` with `--constructor TEAM` and
provide a complete target-member mapping in every non-null candidate plan, as
described in the [pit-plan selection guide](strategy-selection.md). Rival
scenario plans are individual rival-driver overrides and must not name any
target member. Rival drivers omitted from a scenario keep their saved plans.

The rival-scenarios JSON maps scenario names to a positive finite weight and a
map of rival driver IDs to plans. For example:

```json
{
  "rival conservative": {
    "weight": 1,
    "pit_plans": {"RIV": [{"lap": 18, "compound": "hard"}]}
  },
  "rival no elective stop": {
    "weight": 2,
    "pit_plans": {"RIV": []}
  }
}
```

The scenario weights are normalized to sum to one. A missing rival ID inherits
its saved plan; `null` removes that saved plan and restores the automatic
strategy; `[]` keeps elective stops disabled while required repairs and weather
corrections remain active. Requested lap numbers are each driver's own lap at
pit entry. Scenario labels are report labels only.

The selection procedure does not pass the scenario name or rival plan as an
extra input to the automatic target strategy. Scenarios set fixed rival plans
in the saved-input runner; the automatic target strategy continues to operate
on the ordinary evolving race state. The selected target candidate is one
global choice shared across scenarios and is frozen before validation.

Use `--scenario NAME` to choose a source scenario if the saved input contains
several; it does not select a rival scenario. Optional `--parallel
--max-workers 4` enables process workers. Each phase accepts 1–1,000 trials.
All target candidates, rival scenarios, plan overrides, and seed ranges are
preflighted before simulation starts. Saved simulator models, engine, opening
tyres, finite tyre pools, warm-up assumptions, and RNG policy are retained unless
`--rng-policy` is supplied.

## Joint weather and rival assumptions

The same JSON format accepts optional `weather` and `weather_schedule` fields.
For example, choose one target plan across a dry race and a race with known
rainfall beginning on leading lap 12:

```json
{
  "dry": {
    "weight": 3,
    "pit_plans": {},
    "weather": {"condition": "dry", "rain_intensity": 0, "track_wetness": 0},
    "weather_schedule": []
  },
  "rain at lap 12, rival stops at 13": {
    "weight": 1,
    "pit_plans": {"RIV": [{"lap": 13, "compound": "intermediate"}]},
    "weather": {"condition": "dry", "rain_intensity": 0, "track_wetness": 0},
    "weather_schedule": [
      {"lap": 12, "rain_intensity": 0.8, "condition": "heavy_rain"}
    ]
  }
}
```

`weather` is a partial Weather object. Omitted fields inherit the source,
including temperatures, humidity, and `change_probability`. Omitted, `null`, or
empty weather objects all inherit the source. Accepted fields are `condition`,
`track_temperature`, `air_temperature`, `humidity`, `rain_intensity`,
`track_wetness`, and `change_probability`, with the usual strict numeric bounds.
Conditions are `dry`, `cloudy`, `light_rain`, and `heavy_rain`. A condition change
alone does not replace inherited rainfall or surface water; supply those fields
when you want them to change. Set `change_probability` to zero for fixed rainfall.

Omitted or `null` `weather_schedule` inherits the source schedule. An explicit
`[]` clears it. Nonempty schedules require ascending, unique leading laps from
2 through the scheduled race distance and rainfall in 0–1; `condition` is optional.
As with ordinary scheduled-weather runs, the strategy knows the schedule,
atmospheric randomness is disabled, and surface water continues to evolve.

Race-weather variation does not rerun qualifying under the changed weather.
The effective Q1/Q2/Q3 weather from the source is frozen across every candidate,
case, and phase. The selection metadata records `frozen_qualifying_weather`
and each case's complete effective `weather` and `weather_schedule`; replay
exports retain those same inputs. Native tests compare qualifying outcomes
across cases and serial/process replay outcomes for both race engines.
All other source controls, including finite tyre pools, opening tyre ages,
fitting fees, and RNG policy, remain shared. Weighted cases express joint
weather and rival-plan assumptions; their weights are not learned forecast
probabilities. Rival-only requests retain their existing behavior and metadata.

## Scoring and validation

Every target candidate runs in every rival scenario on the same training seed
cohort. Each candidate's objective scores are first combined across scenarios
within each seed using the normalized weights. The resulting per-seed weighted
scores are then averaged to select one candidate. For a constructor, member
points are summed within each scenario and seed for the default `points`
objective. Use `--objective win` or `--objective podium` for classified race-win
or podium probability. A constructor then succeeds once per scenario and seed
when at least one member qualifies; two podiums never count as two successes.
The [objective definitions and score fields](strategy-selection.md#run-a-selection)
apply to both workflows. Points remain available as context even when probability
determines the winner. Freeze the objective with the plans and rival weights.
Complete target coverage is required in every scenario; an incomplete team
outcome does not get replaced with zero or omitted from just one scenario.

Only the fixed reference and training winner run in validation, in every rival
scenario and on the same fresh validation seeds. The weighted selected-minus-
reference difference is formed within each seed, then its mean and standard
error are calculated across seeds. This retains the covariance between rival
scenarios; scenario-level standard errors are not treated as independent. The
report also gives the per-scenario comparisons. Exact training ties prefer the
reference, then use candidate order from the plans file.

Weighted training decisions and validation aggregates retain exact arithmetic
on the normalized floating-point weights: each stored normalized weight is
treated as its exact binary value, multiplied by the objective score, and summed
before comparing candidates. Scores are converted to ordinary JSON numbers
only for reporting. Two displayed means can therefore look equal while one
candidate has a real, very small advantage; the exact-tie policy applies only
when the retained weighted scores are equal.

Normalization scales supplied weights by their maximum and sums the scaled
values with `math.fsum`. Reordering the same rival scenarios therefore preserves
the normalized weights and exact selection decision; their supplied order is
still used for display. The accepted weights remain floating-point values,
including their binary representation, rather than reinterpreted decimal ratios.
Weights that become zero during normalization are rejected.

Weighted aggregate `training_score_table` rows include `mean_score_behind_selected`
and `tied_for_best`. The objective shortfall is the exact best mean minus that candidate's
exact mean, converted to a JSON number only after subtraction. The boolean
records exact equality with the best objective mean. `mean_points_behind_selected`
remains selected minus candidate mean points; it can be negative when the
selected probability winner has fewer points. Per-rival tables retain their
points and add `total_score`/`mean_score`. Aggregate and per-rival validation
include paired objective-score metrics as well as points. Probability changes
and their SEs are rendered in percentage points, using the same covariance
calculation within each seed. The report and CLI show the actual selection reason: a unique
highest weighted mean, reference preference on an exact tie, or candidate order
on an exact tie. They never infer a tie from equal displayed means.

A positive shortfall smaller than the minimum reportable float can appear as
JSON zero with `tied_for_best: false`; this means below numeric reporting
precision, not an exact tie. Missing legacy evidence is shown as not recorded.
Nonzero report and CLI quantities use scientific notation when three decimal
places would otherwise hide them, including weights, validation differences,
standard errors, and conditional gains or losses. Training shortfalls describe
the cohort used to choose the plan; they are not fresh validation estimates or
calibrated advantages. Only the separate held-out cohort provides the reported
validation comparison.

Validation metrics also include paired points outcome profiles. Each
per-scenario profile summarizes the more/equal/fewer outcomes and conditional
gain and loss magnitudes within that scenario. The weighted profile is computed
after rival points are combined within each seed: its counts are seed outcomes,
not separate scenario outcomes or probabilities. Empty gain or loss categories
have null conditional means. If the reference is selected, profiles are null
because no independent alternative was evaluated. These descriptive simulator
outcomes are not calibrated win probabilities, real-world causal effects, or
confidence bounds.

Equal seeds align the random-stream inputs under the selected RNG policy; they
do not freeze later weather reactions, incidents, or other race events across
different plans. The paired differences describe this simulator experiment and
are not proof of a causal real-world advantage. Repeating a command with the
same inputs reuses validation seeds, and choosing among repeated validation
runs or editing candidates after viewing them turns validation into another
selection step.

## Exports and API

With `--export`, the command writes a standalone weighted selection HTML report,
a selection manifest, and separate scenario-specific training and validation
comparison JSON/HTML files. The summary report shows the frozen selected and
reference target plans, supplied and normalized rival weights, training scores,
held-out paired changes and outcome profiles, and the actual disjoint seed cohorts. It links to each
local comparison export and the JSON manifest. Each detailed comparison
contains its own saved inputs and seeds for offline replay. The weighted
aggregate is metadata computed from seed-level target points; it is not
fabricated as a pooled `SimulationResults` object. Standard errors describe
paired sampling error, not confidence intervals. For one validation trial the
standard error is unavailable; if selection returns the reference itself, the
zero difference is identity by definition and has no independent alternative
estimate. The held-out result does not feed back into selection. Generated
filenames use scenario indexes, so user-supplied scenario labels cannot select
or overwrite filesystem paths. The source export is never modified.

The Python entry point is
`f1sim.analysis.rival_strategy_selection.evaluate_saved_rival_pit_plan_selection`.
It returns selection metadata and nested training/validation results keyed by
rival scenario and target candidate label.
