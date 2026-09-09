# Live F1 Monte Carlo Simulator

A Formula 1 race simulator whose live runs use **only the current UTC season**. The backend fetches the active calendar, full-time driver lineup, standings, race results, and qualifying results when it runs. Older-season fetching, bundled grids, cached FastF1 sessions, and stale fallback calendars are intentionally unavailable. Explicitly exported simulation inputs can also be replayed offline for reproducibility.

## What it does

- Fetches the current active calendar from Jolpica, including mid-season cancellations and replacement venues.
- Fetches the official 22-seat lineup from Formula1.com so reserve, former, and FP-only drivers do not enter the grid.
- Calibrates driver form, team pace, and reliability from completed races in the current season only.
- Simulates qualifying for every run; a completed event's real grid or result is never replayed.
- Runs completed and future current-season venues. Future races use the live event identity plus circuit physics configuration and current-season form.
- Keeps fetched F1 data in short-lived memory only. It does not create a data cache or silently fall back to an older season.
- Returns source and fetch-time provenance with calendar, ratings, and simulation responses.
- Shows individual 95% Monte Carlo sampling ranges for win, podium, and DNF probabilities. These describe sampling noise under the chosen model, not confidence in the real race outcome.

The race engine models circuit-dependent car performance, tyre stress and degradation, wet-weather car/driver performance, race-level safety-car risk, 2026 Active Aero, proximity-gated and energy-limited Overtake Mode, incidents with time/strategy consequences, current-season compound form, reliability, pit strategy, and Monte Carlo uncertainty. Active Aero is available to the field on configured straights rather than being a following aid; Overtake Mode is handled separately and is disabled during neutralisations, wet running, and restart laps. Monaco's 2026 Active Aero exception is represented with no configured zones.

Pit strategy compares remaining tyre and pit costs, including projected transitions
from rain tyres to slicks as the surface dries; see [the strategy model and its limits](docs/strategy-model.md)
for its assumptions and remaining limitations.

Retirements retain completed distance and can still qualify for points under the rounded 90% classification threshold; see [classification conventions and limits](docs/race-classification.md).

Both engines apply driver consistency and wet skill to relative random-incident
risk. Lap-aware execution allocates the active field's risk to each car's own
lap; see [incident exposure and model assumptions](docs/weather-calibration.md).

The terminology and operating model follow Formula 1's [official 2026 regulations explainer](https://corp.formula1.com/f1-2026-regulations-terminology-update/) and the FIA's [2026 technical overview](https://www.fia.com/news/f1s-new-era-everything-you-need-know-about-how-fia-making-formula-1-more-competitive-more).

## Requirements

- Python 3.11+
- Internet access while loading the calendar or starting a live-data run

## Install and run

PowerShell on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[web]"
python -m f1sim.web.server
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[web]'
python -m f1sim.web.server
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). The server serves the dashboard itself; a second static-file server is not required.

## Command line

The CLI also accepts current-season races only:

```powershell
python examples/simulate_race.py --race "Italian Grand Prix" --simulations 500
python examples/simulate_race.py --race 14 --scenarios dry,light_rain,heavy_rain
```

The standard race model remains the default. To try individual car crossings,
lapped finishes and shared red-flag restarts, choose **Lap-aware (experimental)**
in the dashboard or pass `--race-engine chronological` to the CLI. The API and
Python `MonteCarloRunner` accept `race_engine="chronological"` (or `"standard"`).
Results record the selected model; see [its assumptions and limits](docs/chronological-race-design.md).

To test an opening tyre choice, fill **Starting tyres (optional)** in the dashboard
with driver-code pairs such as `VER=hard, NOR=soft`, or use the CLI:

```powershell
python examples/simulate_race.py --race 1 --starting-tyres "VER=hard,NOR=soft" --seed 42 --export
```

Use codes from the loaded roster. Allowed compounds are `soft`, `medium`, `hard`,
`intermediate`, and `wet`. Unlisted drivers remain automatic; subsequent stops
still follow the normal policy, including immediate correction for unsuitable
weather tyres. Python `MonteCarloRunner` and `/api/run` accept
`starting_tires={"VER": "hard"}`. Overrides apply to every selected weather
scenario and are saved with the inputs for replay.

The same inputs and seed reproduce an overridden run, including across worker
counts. Changing a tyre choice can change later random draws and race events;
equal seeds do not hold those events fixed across different strategies. Live
runs also reload ratings, so compare saved inputs when checking which inputs changed.

New runs draw evolving weather independently of race decisions. With matching
weather inputs and seeds, tyre choices and pit-service draws no longer change
the weather sequence. This alignment is by shared weather-update interval,
not elapsed seconds; shorter races can record fewer intervals. Incidents,
traffic and adaptive race decisions can still differ between strategies.

Use `--export` only when you explicitly want files for the newly simulated run:

```powershell
python examples/simulate_race.py --race Monza --export --output-dir output
```

Exported CSV, JSON and HTML files use UTF-8 so international driver, team and
circuit names are preserved across operating systems. Select UTF-8 when importing
CSV into a tool that asks for a text encoding.

Each `export_all` bundle receives a unique filename identifier, so exporting the
same race/scenario again preserves earlier CSV, statistics and report files.
The run-history index shows each bundle's race model and links to its own files.
Individual export methods still use their explicitly supplied filenames.

New statistics exports, scenario comparisons and dashboard JSON downloads include
the driver, car, circuit and initial weather models used for each scenario. These
are derived simulation inputs, not cached provider responses. Replay a selected
simulation without fetching live data:

```powershell
python examples/replay_simulation.py output/saved_statistics.json --simulation 2
python examples/replay_simulation.py output/dashboard_run.json --scenario dry --simulation 2 --export
```

Simulation numbers are one-based, matching the CSV. A multi-scenario file needs
`--scenario`; a single-scenario file selects its only scenario automatically.
The command reconstructs qualifying and the race with the saved model inputs,
race engine, starting-tyre overrides and effective seed. Optional exports use a new unique bundle in
`output/replays` (override with `--output-dir`). Track lap, pit-lane and sector
times must be finite positive values; overflowing numbers are rejected before
replay starts. Older exports without inputs
cannot be reconstructed this way.

Snapshots record Python, NumPy, Pydantic and simulator versions plus a digest of
the simulation source files. Replay uses the installed code; identical results
require matching model code, dependencies and runtime behavior. Snapshots do not
archive executable code or runtime monkeypatches, and replay does not claim to
reproduce a real race. Ordinary live runs still fetch current-season inputs.

New snapshots use schema version 2 and require `rng_policy`; new runs use
`isolated_weather_v1`. Earlier installations reject this new schema instead
of silently using the wrong random streams.
Older snapshots without this field replay with `shared_v1`, which preserves
the former shared weather/race draw sequence. Python callers can select
either policy with `MonteCarloRunner(..., rng_policy=...)`.

Compare one driver's opening choices against the same saved inputs, offline:

```powershell
python examples/compare_starting_tyres.py output/saved_statistics.json --driver VER --simulations 100 --export
python examples/compare_starting_tyres.py output/dashboard_run.json --scenario dry --driver VER --compounds automatic,soft,medium,hard
```

Each choice uses the saved race engine and base seed, with the requested number
of trials per choice. All other drivers' starting overrides remain in place;
`automatic` removes only the selected driver's override. Later pit decisions
remain automatic. The table reports wins with 95% sampling intervals, podiums,
retirements and points per race. Equal seeds do not freeze subsequent random
events, and these estimates do not establish the best strategy for a real race.

Saved-input comparisons also show changes against a reference choice. The first
selected choice is the default; use `--reference hard`, for example, to change it.
For each driver, the console and HTML report show the mean points change, its
estimated standard error (SE), how many paired trials earned more/equal/fewer
points, and the retirement-rate change. Positive points changes mean more points;
positive retirement changes mean more DNFs. These use shared recorded trial
seeds and matching qualifying, with their own paired and excluded counts.
SE describes sampling error in the mean difference; it is not a confidence
interval, and zero observed variation does not establish equivalent strategies.
One paired observation has no estimable SE.

Combined exports retain these summaries in `paired_comparisons`. Pairing requires
matching saved models, runtime provenance and random-stream policy; starting
tyres and engine may differ. Missing, duplicate or invalid driver records and
unmatched qualifying are excluded. Different weather inputs are not paired.
Python exporters accept `reference_scenario="hard"`; ordinary scenario exports
omit paired summaries unless a reference is requested.

Nothing is written unless `--export` is supplied. Exported bundles, combined JSON
and an offline HTML comparison report go to `output/strategy-comparisons` (or
`--output-dir`), with unique names. The report opens the selected driver and shows
win, podium and retirement rates with 95% sampling intervals, mean points and
paid stops for every choice. Recorded tyre sequences show how the race policy
actually responded, including free fittings and retirement runs. Race-distance
outcomes make shortened races and lapped finishes visible alongside strategy
results; each metric shows its own recorded counts. Replay a trial from the comparison JSON using
`--scenario hard`, for example.
The source file is unchanged. Comparisons use installed simulator code and share
the replay limitations above. Use `--parallel --max-workers 4` for process workers.

Both saved-input comparison commands retain the source's random-stream policy
by default. Add `--independent-weather` to compare older saved inputs under
independent weather draws. The policy applies to every variant and is recorded
in each exported snapshot; the source file is unchanged. Python comparison
functions accept `rng_policy="isolated_weather_v1"` for the same override.
The console and HTML report identify whether weather draws are independent
of race decisions or shared with race events.

You can also compare the Standard and experimental Lap-aware engines against
the same saved models, weather behavior, starting tyres and base seed:

```bash
python examples/compare_race_engines.py output/saved_statistics.json --simulations 100 --export
```

For a multi-scenario input, select it with `--scenario dry`. Results appear in
`standard,chronological` order by default; `--engines chronological,standard`
reverses that order. Nothing is written without `--export`. Unique bundles and
paired comparison JSON/HTML go to `output/engine-comparisons` or `--output-dir`.
The console includes winning distance, timed races, lapping and driver outcomes;
the HTML also shows actual tyre sequences. Replay an exported trial with
`python examples/replay_simulation.py output/engine-comparisons/race_engines_ID.json --scenario chronological --simulation 2`.
Paired changes use the first selected engine as reference unless overridden with
`--reference chronological` or `--reference standard`.
These comparisons measure sensitivity to the execution model. Equal seeds do
not align all later events, and a different result does not establish that one
engine is more accurate. Qualifying inputs and qualifying seed ranges are shared.
Use `--parallel --max-workers 4` to run each engine's trials in process workers.

Multi-scenario live CLI exports also include a combined HTML report and JSON with
matching unique filenames. Comparison JSON preserves observed driver counts,
points per observed race, sampling intervals and pit-stop statistics. The HTML
report needs no network connection or JavaScript; expand a driver to compare
scenarios. Individual intervals describe each rate, not the difference between
two choices or the accuracy of the model against real races.

After a dashboard run, open the Scenarios tab and choose **Download comparison
report** for the same offline report of all returned scenarios and drivers.
It uses the completed run, regardless of current display filters or edited
controls. JSON downloads retain the replay inputs without embedding the HTML.
Report context includes the initial condition, rain, surface wetness and modeled
per-lap weather-change chance; these describe simulation settings, not a forecast.

Expand **Weather during this trial** in the Race tab to inspect the selected
trial's recorded rainfall and surface wetness. Full export bundles include a
weather CSV, linked from the run-history index. Statistics and comparison JSON
include `weather_histories` in trial order; dashboard JSON includes only the
selected trial's `sample_weather_history`. Values are observations, not forecasts.
Standard entries correspond to shared race-lap starts. Lap-aware entries are
shared weather-update intervals, which can differ from a driver's own lap count
and do not identify each driver's exact pit-stop conditions. Legacy output
without a trace remains explicitly unrecorded.

Expand **Paid pit stops during this trial** to inspect each driver's tyre age,
compound change, weather and modeled lane, service and queue loss. Export bundles
include a pit-stops CSV; statistics and comparison JSON retain `pit_stop_details`
for every trial, while dashboard JSON retains the selected trial's details.
These are observations of the simulation, excluding free tyre changes and later
on-track traffic. Missing legacy details remain distinct from a recorded zero stops.

In Statistics, expand **Paid-stop costs and queue delays** to compare mean lane,
service and queue loss per recorded race, and the share of races with a queue.
Only complete stop histories contribute; recorded zero-stop races and retirements
are included. Statistics and comparison JSON retain `pit_loss_statistics`, and
the comparison report shows mean loss with its own observation count.

The dashboard's scenario chart and driver matrix show the same individual 95%
sampling ranges and observed trial counts. Expand a driver in the chart to view
its scenarios. Highlights identify the highest estimate, not a proven strategy
advantage. Missing observations display as not recorded and export as blank CSV
cells; an observed zero remains 0% with its sampling range.

There is no `--year` option. The current season comes from the backend's UTC date and requests for any other year are rejected. CLI runs use the same 1,000-simulation, 16-worker, and 32-bit-seed safety bounds as the dashboard.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Service status, current season, and live-only data policy |
| `/api/calendar` | GET | Fresh active current-season calendar and provenance |
| `/api/ratings?race=Monza` | GET | Current lineup plus current-season driver/car ratings |
| `/api/run` | POST | Run one or more weather scenarios |

Example body:

```json
{
  "race": "Italian Grand Prix",
  "simulations": 250,
  "scenarios": "dry,light_rain",
  "seed": 42,
  "qualifying_mode": "simulated",
  "parallel": true,
  "max_workers": 4
}
```

If supplied, `year` must equal the current UTC year; omitting it selects the live season.
`qualifying_mode` accepts only `simulated`.
Dashboard requests accept 10–1,000 simulations, up to four supported weather scenarios, seeds from 0 through 4,294,967,295, and at most 16 worker processes. Inputs are validated before any live-data request is made.

Weather scenario names describe the starting conditions. Weather evolves during each race, so a dry start can develop rain. The [weather calibration notes](docs/weather-calibration.md) explain the observations, model assumptions, and reproducible checks.

For controlled strategy comparisons, select **Fixed rainfall** under Weather
behavior, pass `--weather-mode fixed_rainfall` to the CLI, or set
`"weather_mode": "fixed_rainfall"` in `/api/run`. This keeps each scenario's
initial condition and rainfall unchanged. Surface wetness still responds to rain
and drying, and race incidents and interruptions remain active. The default is
`evolving`; neither mode is a forecast. Saved inputs and replay retain the chosen
behavior, and result labels describe the completed run rather than current controls.

## Server capacity

The server admits one complete simulation request at a time by default, including
live-data loading and every selected weather scenario. Additional requests receive
HTTP 429 with `Retry-After: 5`; the dashboard keeps the Run button available for a
later retry. Calendar and health requests remain available.

Set `F1SIM_MAX_CONCURRENT_RUNS` to an integer from 1 through 4 before starting the
server to change this limit. Each admitted request still has its own maximum of
16 simulation workers, so choose both limits to suit the host.

The limit is shared across threads and server processes through OS file locks.
All processes must use the same limit and `F1SIM_RUN_LOCK_DIR`, which defaults to
`f1sim-run-capacity` in the system temporary directory. Use a common writable
local directory for services with different temporary directories or containers
sharing one host. Coordination files contain no live F1 data; do not delete them
while servers are running. Locks release on completion, failure, or process exit.
This limit covers the API on one host; independent hosts and direct CLI runs are
separate capacity domains.

## Live data policy

- Canonical calendar, circuits, standings, and results: [Jolpica F1 API](https://api.jolpi.ca/docs/)
- Full-time team/driver pairing: [official Formula 1 teams page](https://www.formula1.com/en/teams)
- Only data labelled with the current season is accepted.
- No on-disk API cache is created.
- API responses are marked `no-store`; each endpoint creates an isolated live-data snapshot.
- Live snapshots have a bounded aggregate fetch budget and fail closed on truncated, repeated, seasonless, ambiguous, or conflicting provider data.
- Each HTTP response is limited to 8 MiB before parsing. Body reads also check elapsed time between chunks; connection/header handling and individual socket reads still use socket timeouts.
- Form calibration uses only rounds with near-complete, uniquely identified result and qualifying fields; each row remains attributed to the constructor that entered it.
- Cancelled/non-championship events and non-current drivers are excluded.
- If fresh current-season data is unavailable, the request fails clearly instead of using a stale local list.

The small circuit-profile table in the simulator is physics configuration (lap count, aero/speed balance, Active Aero opportunity, tyre stress, overtaking difficulty, and safety-car baseline), not stored race history. Live calendar identity and host location remain authoritative, so unexpected replacement venues still receive a conservative neutral profile.

## Development

```powershell
python -m pip install -e ".[web,dev]"
pytest -q
ruff check .
```

Run `python examples/check_dry_pit_schedules.py` to compare dry strategy against
every legal schedule of up to three stops in short synthetic races. It checks
both engines, reports the best executed alternative and needs no network.
See [strategy diagnostics](docs/strategy-model.md) for the search bounds and
model assumptions.

For an interactive browser regression check, start the server and run
`node tests/browser_dashboard.cjs` with Playwright installed and its Chromium
browser available. `PLAYWRIGHT_MODULE` can point to an existing Playwright
installation; `BROWSER_CHANNEL=msedge` selects installed Microsoft Edge.
The check uses live data for a small run, then tests responsive layouts,
exports, keyboard navigation, and recovery from simulated connection failures.

Set `F1SIM_OFFLINE=1` to run that browser check without a server or F1 network
access. It generates deterministic synthetic results using the real simulation
and serializer, and intercepts all browser requests. `PYTHON` selects the Python
executable when it is not available as `python`. CI runs this offline check in
Chromium and runs the Python suite on Linux (3.11 and 3.12) and Windows (3.12).
The browser job uses the official Playwright container with preinstalled browser
and OS dependencies. Keep its image version and the job's npm Playwright version
aligned when upgrading. Browser setup therefore does not update unrelated apt
repositories supplied by the hosted runner.

`node tests/browser_html_exports.cjs` also runs offline and is included in CI.
It opens real report and history exports with markup-like names, checks local
filename links and verifies that chart data stays intact without executing
injected markup. Plotly calls are captured by a test stub; this check covers the
generated HTML and data embedding, not Plotly's chart rendering.

Project layout:

```text
src/f1sim/
├── data/          # current-season live gateway and rating calibration
├── models/        # drivers, cars, tracks, tyres, weather
├── simulation/    # qualifying, laps, race events, strategy
├── analysis/      # Monte Carlo aggregation and scenarios
├── output/        # opt-in CLI exports
└── web/           # FastAPI service and dashboard payloads
    └── static/
        └── index.html  # packaged current-season-only dashboard
```
