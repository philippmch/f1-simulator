# Selecting a pit plan across rival strategies

The rival-strategy workflow selects one target pit plan across a set of
predeclared opponent-plan scenarios, then evaluates that frozen choice on a
disjoint seed range. Choose the candidate plans, rival scenarios, and scenario
weights before looking at validation results. The supplied weights are explicit
analysis assumptions; the simulator does not infer that they are real-world
probabilities.

This measures outcomes under the saved simulator inputs. It does not establish
that a plan is optimal, calibrated to real races, or better under unmodeled
competitor strategies, weather, or model assumptions.

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

## Scoring and validation

Every target candidate runs in every rival scenario on the same training seed
cohort. Each candidate's target points are first combined across scenarios
within each seed using the normalized weights. The resulting per-seed weighted
scores are then averaged to select one candidate. For a constructor, member
points are summed within each scenario and seed before scenario weighting.
Complete target coverage is required in every scenario; an incomplete team
outcome does not get replaced with zero or omitted from just one scenario.

Only the fixed reference and training winner run in validation, in every rival
scenario and on the same fresh validation seeds. The weighted selected-minus-
reference difference is formed within each seed, then its mean and standard
error are calculated across seeds. This retains the covariance between rival
scenarios; scenario-level standard errors are not treated as independent. The
report also gives the per-scenario comparisons. Exact training ties prefer the
reference, then use candidate order from the plans file.

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
