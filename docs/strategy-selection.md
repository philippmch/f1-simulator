# Selecting a pit plan with fresh-seed validation

Comparing several plans and choosing the highest simulated points total can
reward a lucky sample. This offline workflow separates that choice from its
evaluation: it selects a plan on one set of seeds, freezes the choice, then
compares it with a fixed reference on a different set of seeds.

It evaluates expected points under the saved simulator inputs. It does not
establish that a plan is optimal, calibrated to a real race, or better under
different weather, competitors, tyre pools or model assumptions.

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

IDs must exactly match the saved inputs. Driver selection maximizes that
driver's points; constructor selection maximizes the sum of all modeled
members' points in each race. It supports a saved one-driver constructor.
Rival plans remain unchanged. The usual distinction between `null` (automatic
strategy) and `[]` (no elective stops) applies.

Use `--scenario NAME` when the source has multiple scenarios. Optional
`--parallel --max-workers 4` enables process workers. Each phase accepts
1–1,000 trials in the CLI. All candidates are validated before any simulation
starts. Saved physics, engine, opening tyres, physical pools and warm-up
assumptions are retained; the saved RNG policy is retained unless explicitly
overridden with `--rng-policy`.

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

The candidate with the highest training mean points is selected. An exact tie
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
