# Live F1 Monte Carlo Simulator

A Formula 1 race simulator that runs **only for the current UTC season**. The backend fetches the active calendar, full-time driver lineup, standings, race results, and qualifying results when it runs. Older seasons, bundled grids, cached FastF1 sessions, and stale fallback calendars are intentionally unavailable.

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

Pit strategy compares remaining dry-race tyre and pit costs and reacts to changing weather; see [the strategy model and its limits](docs/strategy-model.md) for its assumptions and remaining limitations.

Retirements retain completed distance and can still qualify for points under the rounded 90% classification threshold; see [classification conventions and limits](docs/race-classification.md).

The terminology and operating model follow Formula 1's [official 2026 regulations explainer](https://corp.formula1.com/f1-2026-regulations-terminology-update/) and the FIA's [2026 technical overview](https://www.fia.com/news/f1s-new-era-everything-you-need-know-about-how-fia-making-formula-1-more-competitive-more).

## Requirements

- Python 3.11+
- Internet access while loading the calendar or running a simulation

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

Use `--export` only when you explicitly want files for the newly simulated run:

```powershell
python examples/simulate_race.py --race Monza --export --output-dir output
```

Exported CSV, JSON and HTML files use UTF-8 so international driver, team and
circuit names are preserved across operating systems. Select UTF-8 when importing
CSV into a tool that asks for a text encoding.

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
