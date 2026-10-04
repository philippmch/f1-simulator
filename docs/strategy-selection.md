# Selecting a pit plan with fresh-seed validation

Comparing several plans and choosing the highest simulated points total can
reward a lucky sample. This offline workflow separates that choice from its
evaluation: it selects a plan on one set of seeds, freezes the choice, then
compares it with a fixed reference on a different set of seeds.

It evaluates expected points, race-win probability, or podium probability under
the saved simulator inputs. It does not
establish that a plan is optimal, calibrated to a real race, or better under
different weather, competitors, tyre pools or model assumptions.

## Use candidate selection in the dashboard

In the dashboard setup controls, turn on **Choose among candidate plans, then
validate the frozen choice**. The editor opens in Race Results. Choose a driver
or constructor from the selected race's current roster. Choose **Selection
objective** before running: expected points (the default), race win probability,
or podium probability. Add two to ten named candidates (up to 80 characters
each), mark one as the fixed reference, and
choose each plan before running. Names are labels; they do not change scoring.

Each candidate can use the automatic strategy, or custom stops written with
the same lap and compound shorthand as **Custom pit plans**. For example,
`18:medium,36:hard` schedules two elective stops for the target member and
`none` explicitly suppresses elective stops. Automatic maps to `null`; `none`
maps to an empty instruction list. A custom constructor plan must include every
current target member.

Training and validation each default to 50 trials. The dashboard shows the
worst-case trial budget for every selected weather, including the ordinary
source simulation count. Without optional rival scenarios, the budget is:

```text
source simulations + candidate count × training trials + 2 × validation trials ≤ 1,000
```

To choose one plan across possible race weather and opponent plans, open **Weather and rival
assumptions**. The [weighted scenario selection guide](rival-strategy-selection.md#use-the-dashboard)
explains the inputs and results. Each rival scenario adds its own candidate
training and reference/winner validation runs, so the budget becomes source
simulations plus rival-scenario count times the training and validation work
above. Source simulations are still counted once.

The source count keeps its existing meaning. Source simulations still produce
the ordinary race charts and statistics; candidate simulations are additional
work. Candidate selection and **Compare with automatic strategy** cannot be
used in the same run. The dashboard checks these inputs and the budget before
sending `/api/run`.

After the run, open **Scenario Lab** to see the separate selection evidence.
The training table identifies the winner and its reference; a later section
shows the frozen selected plan and fixed reference, their held-out paired point
means, mean difference and Monte Carlo standard error. The more/equal/fewer
counts and conditional gains/losses describe those validation trials. If the
reference wins training, the dashboard reports an identity choice and states
that no independent alternative estimate exists. Missing measurements are
shown as unavailable rather than as zero.

The selected candidate never changes in response to validation results.
Changing the form after a run does not rewrite the saved result, and the
source race charts continue to describe the source simulation. The selection
section follows the focused weather entry in that response. Its downloads
include selection evidence JSON, separate training and validation replay JSON,
and an optional validation HTML report. JSON downloads omit generated report
HTML.

## Run a selection

Use an exported statistics file, or select a scenario from a comparison export.
The plans JSON has the same format as the [pit-plan comparison](custom-pit-plans.md)
workflow. Supply between two and ten named candidates, including the fixed
reference. Choose the candidates and reference before inspecting validation
results.

For a driver:

```powershell
python examples/validate_pit_plan_selection.py output/saved_statistics.json --driver VER --plans plans.json --reference automatic --training-simulations 100 --validation-simulations 100 --export
```

Example driver plans:

```json
{
  "automatic": null,
  "earlier": [{"lap": 14, "compound": "hard"}],
  "later": [{"lap": 18, "compound": "hard"}]
}
```

For coordinated teammate plans, replace `--driver VER` with
`--constructor TEAM` and provide the complete saved constructor membership in
every non-null alternative:

```json
{
  "automatic": null,
  "staggered": {
    "A": [{"lap": 14, "compound": "hard"}],
    "B": [{"lap": 16, "compound": "hard"}]
  }
}
```

IDs must exactly match the saved inputs. The default objective maximizes that
driver's points or the sum of all modeled constructor members' points per race.
It supports a saved one-driver constructor.
Rival plans remain unchanged. The usual distinction between `null` (automatic
strategy) and `[]` (no elective stops) applies.

Use `--scenario NAME` when the source has multiple scenarios. Optional
`--parallel --max-workers 4` enables process workers. Each phase accepts
1–1,000 trials in the CLI. All candidates are validated before any simulation
starts. Saved physics, engine, opening tyres, physical pools and warm-up
assumptions are retained; the saved RNG policy is retained unless explicitly
overridden with `--rng-policy`.

Use `--objective win` or `--objective podium` to maximize the corresponding
simulator probability. The Python selectors and dashboard API accept
`objective="points"`, `"win"`, or `"podium"`; the dashboard field belongs inside
`pit_plan_selection`. Unsupported values are rejected before loading inputs or
starting trials. Omission retains expected points.

For a driver, win means a classified P1 and podium means a classified P1–P3.
For a constructor, the objective succeeds once per race if **at least one**
member meets that condition. Two team podiums still count as one successful
race. Classification follows the ordinary published race statistics: a late
retirement that remains classified can count, while an unclassified result
cannot. Legacy records without a classification flag use finished status.
Every member must still have valid outcome evidence, even when another member
already satisfies the objective. Points remain reported for all objectives.

Selection metadata records `objective`, its description, and `score_unit`.
Training rows retain `total_points` and `mean_points`, and add `total_score`,
`mean_score`, `mean_score_behind_selected`, and `tied_for_best` for the chosen
objective. Validation retains point metrics and adds `reference_mean_score`,
`selected_mean_score`, `mean_score_difference`, and
`score_difference_standard_error`. Probability scores use 0–1; the CLI,
dashboard, and HTML render probabilities as percentages and differences/SEs
as **percentage points**. The objective is frozen before training, and neither
it nor the selected plan is changed in response to held-out results. An
identity comparison has no independent alternative SE; one paired trial also
cannot estimate a sample SE. Older metadata without an objective means points.

## Selection and validation rules

Training starts immediately after the seed range recorded in the source file.
Validation starts immediately after training. For example, a source recording
seed 42 and ten trials used seeds 42–51. With 100 training and 100 validation
trials, training uses 52–151 and validation uses 152–251. The ranges never
overlap or wrap; requests extending beyond seed `2**32 - 1` are rejected.

Every training candidate uses the same complete seed range and matching
qualifying results. Every target driver must have a valid points outcome in
every trial. A constructor needs every modeled teammate in every trial.
Retirements remain valid points observations. Missing, duplicate or malformed
observations reject the selection rather than giving candidates different
denominators.

The candidate with the highest training mean objective score is selected. An exact tie
prefers the reference; a remaining tie uses the order in the plans file.
Training scores describe the sample used to make that choice. They are not an
independent estimate of the selected plan's advantage.

Only the selected plan and reference run in validation. The selected label
cannot change in response to validation outcomes. If the reference itself
wins training, it runs once in validation and the result explicitly records
that no change was selected.

For a selected alternative, the held-out comparison reports the mean paired
points change and its Monte Carlo standard error. Constructor points are
summed within each seed before computing paired differences, retaining
teammate covariance. Validation also requires complete target observations
and qualifying matches. Equal seeds do not freeze every later incident or
race event across different plans.

The held-out metrics also include a descriptive points-outcome profile: the
number of seeds with more, equal, or fewer target points, the mean gain among
seeds with a gain, and the mean loss magnitude among seeds with a loss. For a
constructor, teammate points are summed inside each seed before this profile
is calculated. An empty gain or loss category has a null conditional mean. If
the reference is selected, the profile is null because no independent
alternative was evaluated. These are paired simulator outcomes, not
calibrated win probabilities, real-world causal effects, or confidence bounds.

## Reading and replaying the result

The console separates training scores from validation evidence. A training
winner can lose to the reference in validation; that result is retained without
selecting another candidate. Small differences or a small number of trials
are weak evidence. A zero sample standard error does not prove that the
underlying outcome has no uncertainty.

With `--export`, the command writes selection metadata and separate training
and validation comparison files. Their saved inputs and seed ranges support
the existing offline replay workflow. The source file is never rewritten.
Runtime provenance is reported because replay uses the installed model and
dependencies.

Fresh seeds here means disjoint from the source file and the current training
phase. Repeating the command with the same inputs reuses the same validation
seeds. Changing candidates after inspecting those results, or choosing among
many validation runs, turns validation into another selection exercise.
This workflow does not track experiments across files or correct for that
additional selection.

The Python entry point is
`f1sim.analysis.strategy_selection.evaluate_saved_pit_plan_selection`.
It returns selection metadata alongside the training and validation
`SimulationResults` mappings for further analysis and export.
