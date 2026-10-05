# Reproducible SC/VSC scenarios

Use an assumed race-control schedule to compare a pit window or a fixed plan
under a specific sequence of Safety Car and Virtual Safety Car announcements.
The schedule is an experimental assumption, not a prediction of marshal decisions.
Both race engines execute it using their existing observed control and countdowns.

## Inputs

Omit `control_schedule`, or pass `null`/Python `None`, for the usual automatic
SC/VSC model. An explicit `[]` disables random SC/VSC deployments. Individual
incidents, mechanical failures and red flags remain active in either mode.

For example, this JSON source requests SC after leading crossing 12, followed
by VSC after leading crossing 26:

```json
[
  {"lap": 12, "control": "safety_car", "duration_laps": 4},
  {"lap": 26, "control": "vsc", "duration_laps": 2}
]
```

Each entry requires exactly `lap`, `control` and `duration_laps`. Laps must be
JSON integers from 1 through 1000 within the original scheduled distance;
duration must be an integer from 1 through 6. Only `safety_car` and `vsc` are
accepted. At most 20 entries are allowed, in increasing order. A later lap
must exceed the previous lap plus its duration, leaving a completed green
interval after clearance. A duration may extend past the race finish.

The CLI uses `--control-schedule 12:sc:4,26:vsc:2`; `--control-schedule none`
means an explicit empty schedule. Omit the flag for automatic race control.
The dashboard's **SC/VSC scenario** selector has the same distinction: choose
**Assumed announcements** and enter one `LAP:sc/vsc:DURATION` per line, or
leave the editor blank for no SC/VSC announcements. A hidden editor under
**Automatic race control** does not submit its stored text.

The API accepts the JSON list as `control_schedule` on `/api/run`. Python
callers can pass it to `MonteCarloRunner(...)` or `RaceSimulator(...)`.
Structural validation occurs before live loading or simulation; distance is
checked after the track is loaded and before trials begin. Inputs are copied.
The existing direct-Python forced SC hook cannot be combined with this source,
including an empty schedule. Forced red flags retain priority.

## Observation and strategy

An announcement is processed after the chosen leading crossing. It does not
retroactively slow the completed lap or remove its green-lap credit for points.
It does invalidate green credit for other unfinished own laps exposed to the
procedure. Continuing cars observe the active
SC/VSC and countdown at their next decisions; already sampled running remains
committed. The standard engine advances together, while chronological cars
can be on different own laps or in service when an announcement occurs.

For example, an announcement after two green leading laps preserves the
required consecutive pair even if every remaining lap runs under SC/VSC.
An intervention sampled during the second lap instead breaks that pair.
Actual awards still depend on completed distance and classification.

In the chronological engine, the shared race-control count keeps increasing
if a lapped survivor inherits the lead. It does not replay the retired leader's
old control intervals. This is the existing simplified control cadence, rather
than sector or pit-lane Control Line geometry.

Opening and in-race forecasts receive current observed control only. They
cannot see a future SC/VSC request. This differs from a
[prescribed rainfall schedule](weather-schedule.md), which is explicitly known
to the tyre policy. A [conditional pit window](custom-pit-plans.md#safety-car-and-vsc-windows)
can respond early to a matching observed deployment, or use its fixed deadline:

```powershell
python examples/simulate_race.py --scenarios dry --no-parallel `
  --control-schedule 12:sc:4,26:vsc:2 `
  --pit-plans "VER=13-20@sc:hard"
```

The two lap inputs have different meanings: control announcements use the
shared race-control clock, and paid pit instructions use each driver's own lap.
Queuing, inventory availability, tyre usage limits, weather suitability and
free red-flag refits retain their existing behavior. No physics coefficients
are changed by selecting the source.

## Outcomes and coverage

`SimulationResults.control_schedule_histories` stores one global history per
trial, rather than duplicating it on every driver. Each requested entry has
one of these outcomes:

| Status | Meaning |
| --- | --- |
| `applied` | The announcement was deployed (`scheduled_announcement`). |
| `suppressed` | The due request lost priority to `red_flag`, `no_survivors`, or an existing neutralization. |
| `not_reached` | The race ended before the request was processed (`race_ended_before_request`). |

Applied records do **not** prove that all requested intervals ran. A later
suspension or the finish can interrupt the duration. Requested duration remains
an input assumption. A shortened race can leave later valid requests unreached.

`get_control_schedule_statistics()` aggregates only complete matching histories.
A malformed or partial trial contributes no request counts. Coverage records
complete, missing, invalid and unrecorded trials, plus unexpected extra histories.
An explicit empty schedule has a valid empty history, while a missing history
remains unknown. Reports show these coverage counts with each request's outcomes.

Statistics and comparison JSON include the raw histories and aggregate coverage.
`export_control_schedule_history_csv()` and dashboard **Download SC/VSC history CSV**
write global outcomes once per trial, with explicit missing/invalid rows. Empty
scenarios receive a complete `no_requests` row. The dashboard also retains the
selected sample's history, and JSON downloads retain all recorded histories.
Automatic legacy inputs have no fabricated schedule history.

## Replay, comparisons and frozen scenarios

Any explicit list, including `[]`, uses saved input schema 11 and
`control_schedule_policy="observed_control_schedule_v1"`. Schema 11 retains
pit windows, tyre usage limits, warmup costs, qualifying weather and prescribed
rainfall with their respective policy markers. Older schemas cannot claim
control fields, and replay rejects missing or unsupported schedule policy.

Saved engine, opening-tyre, driver-plan and constructor-plan variants inherit
the source, preserving empty lists. Ordinary paired comparisons require the
same canonical control assumptions. Matching seeds do not hold every later
race event constant across different plans.

[Weighted rival scenarios](rival-strategy-selection.md) can override
`control_schedule` alongside race weather and rival plans. Omission or `null`
inherits the source; `[]` selects no SC/VSC announcements. Scenario schedules
are frozen and checked against the original track before training starts.
Qualifying conditions remain shared, and selected plans are assessed on the
existing disjoint validation seed range. Each scenario records its effective
source and actual per-trial outcomes separately. Weights remain supplied
assumptions; these experiments do not establish real-world calibration or
a globally optimal strategy.
