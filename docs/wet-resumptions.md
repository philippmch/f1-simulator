# Compulsory full-wet resumption scenarios

Use `resume_wet` to supply a Race Director instruction to resume a suspended
race on full-wet tyres. This is an experimental input, not a prediction from
rain intensity. Both engines execute the existing one counted safety-car
resumption lap, requiring full wets until the safety car returns.

```powershell
python examples/simulate_race.py --scenarios dry --no-parallel --control-schedule 12:red:resume_wet
python examples/check_wet_resumptions.py
```

In the dashboard, select **Assumed announcements** and enter
`12:red:resume_wet`. Python and `/api/run` accept:

```json
{"control_schedule": [{"lap": 12, "control": "red_flag", "action": "resume_wet"}]}
```

An announcement after crossing 12 normally means the counted resumption lap
is lap 13. Chronological cars may have different own-lap numbers. Strategies
learn the instruction when it is announced; future red decisions are not
available to their forecasts. Ordinary `resume` retains free compound choice,
and `abandon` retains historical countback without resumption or a free fit.

The supplied instruction follows the director decision in B5.15.2(a) and the
full-wet requirement in B6.3.7 of the
[FIA 2026 Sporting Regulations, Issue 09](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf).
The model restricts legal tyre choices. It does not simulate deliberate
violations or the official penalty procedure.

## Stock, plans and actual use

The suspension fit is free and adds no paid stop. Unlimited-stock scenarios
can fit fresh full wets. Finite pools must provide a usable full-wet physical
set, including a retained or previously used set. Fitting does not reset its
wear or remaining-lap allowance. The actual counted SC lap consumes one
allowance lap; waiting and fitting consume none. A car with no usable full-wet
set retires at its previous completed crossing with the reason
`No usable full-wet tyre set for compulsory resumption`. No stock or paid
service is fabricated to make that car continue.

Dry-weather mismatch cannot trigger a change away from full wets during the
mandate. A puncture or expired set still requires a usable full-wet replacement.
Once the safety car returns, ordinary weather selection and compulsory
corrections apply again. Full-wet use counts only after actual running; an
unrun fitting does not grant a dry-compound exemption.

A fixed non-wet request due during the mandated lap is recorded as overridden
with `mandatory_wet_tires`, without a paid fit to another compound. A conditional
window cannot make an early non-wet change, but keeps its later deadline if
that deadline has not arrived. The planners retain later custom instructions,
weather forecasts, finite stock and ordinary paid-stop costs when selecting
among eligible full-wet sets.

## Evidence and replay

Schedules containing `resume_wet` use snapshot schema 13 and
`control_schedule_policy="observed_control_schedule_v3"`. They preserve tyre
inventories, usage limits, pit windows, warmup, qualifying weather, prescribed
rainfall and random-stream policies. Older schemas cannot claim this action.
Schedules without it retain their existing schema and policy.

Reports explain the compulsory full-wet assumption; CSV and JSON retain the
canonical `resume_wet` action. Physical-set histories distinguish free fitting,
prior wear and completed use. Native tests cover both engines, process workers,
replay, unavailable stock and repairs. Independent schedule enumeration checks
finite-stock planning costs under the supplied constraint. Those checks prove
consistency with the model, not calibrated real-race tyre performance.

## Limits

This adds a resumption instruction to the existing lap-resolution procedure.
It does not model compulsory full-wet formation laps at the initial start,
standing grids, unlapping, pit-lane Control Line geometry, sector-level SC
lights-out timing or extra director-ordered circulations. A request at an
already finished crossing is suppressed without fitting tyres. The modeled
pause and one counted resumption circulation remain scenario conventions;
see [suspension timing](strategy-model.md#red-flag-suspension-timing).
