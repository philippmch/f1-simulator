"""Minimal web dashboard for running and comparing simulations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f1sim.analysis import MonteCarloRunner, parse_scenario_labels, scenario_weather_from_label
from f1sim.data import HistoricalDataLoader
from f1sim.models import Weather, WeatherCondition
from f1sim.output import Exporter


@dataclass
class DashboardRunRequest:
    """Input payload for dashboard simulation run."""

    year: int = 2025
    race: str = "Bahrain"
    simulations: int = 200
    scenarios: str = "dry,light_rain"
    seed: int = 42
    parallel: bool = True
    max_workers: int | None = None


def _summarize_scenario_results(results_by_name: dict[str, Any]) -> dict[str, Any]:
    """Build compact summary payload for UI responses."""
    summary: dict[str, Any] = {"scenarios": {}}
    for scenario_name, results in results_by_name.items():
        win_probs = results.get_win_probabilities()
        top_3 = list(win_probs.items())[:3]
        summary["scenarios"][scenario_name] = {
            "num_simulations": results.num_simulations,
            "seed": results.seed,
            "top3_win_probabilities": top_3,
            "event_rates": results.get_event_rates(),
            "team_projection": results.get_team_championship_projection(),
        }

    return summary


def _read_run_history(output_dir: str | Path = "output") -> list[dict[str, Any]]:
    """Read exporter run history file if present."""
    history_path = Path(output_dir) / ".run_history.json"
    if not history_path.exists():
        return []

    data = json.loads(history_path.read_text())
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def run_dashboard_simulation(request: DashboardRunRequest) -> dict[str, Any]:
    """Execute one dashboard simulation bundle and return summary."""
    loader = HistoricalDataLoader(cache_dir="data/cache")

    track_stats = loader.get_track_stats(request.year, request.race)
    driver_stats = loader.get_weighted_driver_stats(
        year=request.year,
        target_race=request.race,
        form_races=3,
        track_weight=0.5,
        form_weight=0.3,
        quali_weight=0.2,
    )

    drivers = loader.create_drivers_from_stats(driver_stats)
    cars = loader.create_cars_from_stats(driver_stats)
    track = loader.create_track_from_stats(track_stats)
    historical_grid = loader.get_historical_grid(request.year, request.race)

    base_weather = Weather(
        condition=WeatherCondition.DRY,
        track_temperature=35.0,
        air_temperature=25.0,
        change_probability=track.weather_variability,
    )

    labels = parse_scenario_labels(request.scenarios)
    scenario_results = {}
    exporter = Exporter(output_dir="output")

    for idx, label in enumerate(labels):
        scenario = scenario_weather_from_label(base_weather, label)
        runner = MonteCarloRunner(
            drivers=drivers,
            cars=cars,
            track=track,
            weather=scenario.weather,
            seed=request.seed + idx * 1000,
            historical_grid=historical_grid,
        )
        result = runner.run(
            num_simulations=request.simulations,
            parallel=request.parallel,
            max_workers=request.max_workers,
        )
        scenario_results[scenario.name] = result
        exporter.export_all(result, prefix=f"{request.year}_{track.id}_{scenario.name}")

    payload = _summarize_scenario_results(scenario_results)
    payload["track"] = track.name
    payload["year"] = request.year
    payload["race"] = request.race
    return payload


def build_dashboard_html() -> str:
    """Build minimal dashboard HTML UI."""
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>F1Sim Dashboard</title>
  <style>
    body {
      font-family: Inter, system-ui, sans-serif;
      margin: 24px;
      background: #0f1220;
      color: #e8ebff;
    }
    .card {
      background: #181c30;
      border: 1px solid #2a3156;
      border-radius: 12px;
      padding: 16px;
      max-width: 860px;
    }
    label { display: block; margin-top: 10px; }
    input {
      width: 100%;
      padding: 8px;
      border-radius: 8px;
      border: 1px solid #2a3156;
      background: #10142a;
      color: #e8ebff;
    }
    button {
      margin-top: 14px;
      padding: 10px 14px;
      border-radius: 8px;
      border: 0;
      background: #7aa2ff;
      color: #0b1024;
      font-weight: 700;
      cursor: pointer;
    }
    pre {
      background: #10142a;
      border: 1px solid #2a3156;
      padding: 12px;
      border-radius: 10px;
      overflow: auto;
    }
    .runs-list { border: 1px solid #2a3156; border-radius: 10px; overflow: hidden; }
    .run-item { padding: 10px 12px; border-bottom: 1px solid #2a3156; }
    .run-item:last-child { border-bottom: 0; }
    .run-meta { color: #b6c0ff; font-size: 12px; }
    .scenario-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
    }
    .scenario-card {
      border: 1px solid #2a3156;
      border-radius: 10px;
      padding: 10px;
      background: #10142a;
    }
    .scenario-title { font-weight: 700; margin-bottom: 6px; }
    .chart-wrap {
      border: 1px solid #2a3156;
      border-radius: 10px;
      padding: 10px;
      background: #10142a;
    }
    .chart-row { margin-bottom: 8px; }
    .chart-label { font-size: 12px; color: #b6c0ff; margin-bottom: 4px; }
    .bar-track { background: #242c4f; border-radius: 8px; height: 12px; overflow: hidden; }
    .bar-fill { background: #7aa2ff; height: 100%; }
    .matrix-wrap { border: 1px solid #2a3156; border-radius: 10px; overflow: auto; }
    .matrix-table { width: 100%; border-collapse: collapse; background: #10142a; }
    .matrix-table th, .matrix-table td { border: 1px solid #2a3156; padding: 8px; font-size: 12px; }
    .matrix-table th { background: #181c30; }
    .matrix-best { background: rgba(122, 162, 255, 0.2); font-weight: 700; }
    a { color: #7aa2ff; text-decoration: none; }
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>F1Sim Dashboard</h1>
    <p>Run scenario simulations from the browser.</p>
    <label>Year <input id=\"year\" value=\"2025\" /></label>
    <label>Race <input id=\"race\" value=\"Bahrain\" /></label>
    <label>Simulations <input id=\"simulations\" value=\"200\" /></label>
    <label>Scenarios <input id=\"scenarios\" value=\"dry,light_rain,heavy_rain\" /></label>
    <div style=\"margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;\">
      <button type=\"button\" id=\"presetDry\" style=\"margin-top:0\">Preset: Dry vs Cloudy</button>
      <button type=\"button\" id=\"presetMixed\" style=\"margin-top:0\">Preset: Mixed Race</button>
      <button type=\"button\" id=\"presetWet\" style=\"margin-top:0\">Preset: Wet Heavy</button>
    </div>
    <label>Seed <input id=\"seed\" value=\"42\" /></label>
    <button id=\"runBtn\">Run Simulation</button>
    <button id=\"rerunBtn\" type=\"button\">Re-run Last Config</button>
    <button id=\"clearBtn\" type=\"button\">Clear Saved Config</button>
    <div id=\"quickActions\" style=\"margin-top:10px;\"></div>
    <h2>Scenario Comparison</h2>
    <div id=\"resultCards\">No runs yet</div>
    <h2>Win Probability Chart</h2>
    <div id=\"resultChart\">No chart yet</div>
    <h2>Scenario Trend Strip</h2>
    <label style=\"margin:0\">Trend metric
      <select id=\"trendMetric\">
        <option value=\"sc\">Safety Car race rate</option>
        <option value=\"rf\">Red flag race rate</option>
        <option value=\"inc\">Average incidents</option>
      </select>
    </label>
    <div id=\"resultTrends\">No trend data yet</div>
    <h2>Driver Matrix (by scenario)</h2>
    <div style=\"margin:6px 0; display:flex; gap:8px; flex-wrap:wrap;\">
      <label style=\"margin:0\">Sort
        <select id=\"matrixSort\">
          <option value=\"driver\">Driver</option>
          <option value=\"best\">Best win%</option>
          <option value=\"avg\">Average win%</option>
        </select>
      </label>
      <label style=\"margin:0\">
        <input type=\"checkbox\" id=\"matrixHighlight\" checked />
        Highlight best scenario per driver
      </label>
      <label style=\"margin:0\">Filter driver
        <input id=\"matrixDriverFilter\" value=\"\" placeholder=\"e.g. VER\" />
      </label>
      <label style=\"margin:0\">Filter scenarios
        <input id=\"matrixScenarioFilter\" value=\"\" placeholder=\"e.g. dry\" />
      </label>
    </div>
    <div id=\"resultMatrix\">No matrix yet</div>
    <h2>Raw Result</h2>
    <pre id=\"result\">{\"status\": \"idle\"}</pre>
    <h2>Recent Runs</h2>
    <label>
      Filter runs by track
      <input id=\"runsFilter\" value=\"\" placeholder=\"e.g. Bahrain\" />
    </label>
    <div id=\"runs\">No runs yet</div>
  </div>

  <script>
    const runsEl = document.getElementById('runs');
    const resultEl = document.getElementById('result');
    const cardsEl = document.getElementById('resultCards');
    const chartEl = document.getElementById('resultChart');
    const trendsEl = document.getElementById('resultTrends');
    const trendMetricEl = document.getElementById('trendMetric');
    const matrixEl = document.getElementById('resultMatrix');
    const matrixSortEl = document.getElementById('matrixSort');
    const matrixHighlightEl = document.getElementById('matrixHighlight');
    const matrixDriverFilterEl = document.getElementById('matrixDriverFilter');
    const matrixScenarioFilterEl = document.getElementById('matrixScenarioFilter');
    const actionsEl = document.getElementById('quickActions');
    const scenariosInput = document.getElementById('scenarios');
    const rerunBtn = document.getElementById('rerunBtn');
    const clearBtn = document.getElementById('clearBtn');
    const runsFilterInput = document.getElementById('runsFilter');
    let latestScenarioData = null;

    function renderScenarioCards(data) {
      const scenarios = data.scenarios || {};
      const names = Object.keys(scenarios);
      if (!names.length) {
        cardsEl.textContent = 'No scenario results yet';
        return;
      }

      const cards = names.map((name) => {
        const s = scenarios[name] || {};
        const top = (s.top3_win_probabilities || [])
          .map(([driver, prob]) => `${driver}: ${prob.toFixed(1)}%`)
          .join('<br/>');
        const rates = s.event_rates || {};
        const sc = ((rates.safety_car_race_rate || 0) * 100).toFixed(1);
        const rf = ((rates.red_flag_race_rate || 0) * 100).toFixed(1);
        const teams = s.team_projection || {};
        const topTeam = Object.keys(teams)[0];
        const topTeamPts = topTeam ? Number(teams[topTeam]).toFixed(2) : '-';
        return `
          <div class="scenario-card">
            <div class="scenario-title">${name}</div>
            <div><strong>Top 3 win%</strong><br/>${top || '-'}</div>
            <div style="margin-top:6px">
              <strong>Event rates</strong><br/>
              SC race rate: ${sc}%<br/>
              Red flag rate: ${rf}%
            </div>
            <div style="margin-top:6px">
              <strong>Top team</strong><br/>
              ${topTeam || '-'} (${topTeamPts} pts/race)
            </div>
          </div>
        `;
      });

      cardsEl.innerHTML = `<div class="scenario-grid">${cards.join('')}</div>`;
    }

    function renderWinChart(data) {
      const scenarios = data.scenarios || {};
      const names = Object.keys(scenarios);
      if (!names.length) {
        chartEl.textContent = 'No chart data yet';
        return;
      }

      // Use first scenario's top-3 drivers as chart focus for quick visual compare.
      const seedScenario = scenarios[names[0]] || {};
      const focusDrivers = (seedScenario.top3_win_probabilities || []).map(([d]) => d);

      const rows = focusDrivers.map((driver) => {
        const bars = names
          .map((name) => {
            const top = scenarios[name].top3_win_probabilities || [];
            const item = top.find(([d]) => d === driver);
            const pct = item ? Number(item[1]) : 0;
            const safePct = Math.max(0, Math.min(100, pct));
            return `
              <div class="chart-row">
                <div class="chart-label">${driver} · ${name} · ${pct.toFixed(1)}%</div>
                <div class="bar-track">
                  <div class="bar-fill" style="width:${safePct}%"></div>
                </div>
              </div>
            `;
          })
          .join('');
        return bars;
      });

      chartEl.innerHTML = `<div class="chart-wrap">${rows.join('')}</div>`;
    }

    function renderScenarioTrends(data) {
      const scenarios = data.scenarios || {};
      const names = Object.keys(scenarios);
      if (!names.length) {
        trendsEl.textContent = 'No trend data yet';
        return;
      }

      const metric = trendMetricEl.value;

      const rows = names.map((name) => {
        const rates = scenarios[name].event_rates || {};
        const sc = Number((rates.safety_car_race_rate || 0) * 100);
        const rf = Number((rates.red_flag_race_rate || 0) * 100);
        const incidents = Number(rates.avg_incidents || 0);

        let value = sc;
        let label = `${name} · SC ${sc.toFixed(1)}%`;
        let maxValue = 100;
        if (metric === 'rf') {
          value = rf;
          label = `${name} · RF ${rf.toFixed(1)}%`;
        } else if (metric === 'inc') {
          value = incidents;
          maxValue = 5;
          label = `${name} · Inc ${incidents.toFixed(2)}`;
        }

        const safePct = Math.max(0, Math.min(100, (value / maxValue) * 100));
        return `
          <div class="chart-row">
            <div class="chart-label">${label}</div>
            <div class="bar-track">
              <div class="bar-fill" style="width:${safePct}%"></div>
            </div>
          </div>
        `;
      });

      trendsEl.innerHTML = `<div class="chart-wrap">${rows.join('')}</div>`;
    }

    function renderDriverMatrix(data) {
      latestScenarioData = data;
      const scenarios = data.scenarios || {};
      const allNames = Object.keys(scenarios);
      const scenarioFilter = (matrixScenarioFilterEl.value || '').trim().toLowerCase();
      const names = allNames.filter(
        (name) => !scenarioFilter || name.toLowerCase().includes(scenarioFilter)
      );
      if (!names.length) {
        matrixEl.textContent = 'No scenarios match this filter';
        return;
      }

      const drivers = new Set();
      names.forEach((name) => {
        const top = scenarios[name].top3_win_probabilities || [];
        top.forEach(([driver]) => drivers.add(driver));
      });

      const filterText = (matrixDriverFilterEl.value || '').trim().toLowerCase();
      const driverRows = Array.from(drivers)
        .filter((driver) => !filterText || driver.toLowerCase().includes(filterText))
        .map((driver) => {
          const values = names.map((name) => {
            const top = scenarios[name].top3_win_probabilities || [];
            const item = top.find(([d]) => d === driver);
            return item ? Number(item[1]) : 0;
          });
          return {
            driver,
            values,
            best: Math.max(...values),
            avg: values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1),
          };
        });

      if (!driverRows.length) {
        matrixEl.textContent = 'No drivers match this filter';
        return;
      }

      const sortMode = matrixSortEl.value;
      if (sortMode === 'best') {
        driverRows.sort((a, b) => b.best - a.best);
      } else if (sortMode === 'avg') {
        driverRows.sort((a, b) => b.avg - a.avg);
      } else {
        driverRows.sort((a, b) => a.driver.localeCompare(b.driver));
      }

      const highlight = matrixHighlightEl.checked;
      const header = `<tr><th>Driver</th>${names.map((n) => `<th>${n}</th>`).join('')}</tr>`;
      const rows = driverRows
        .map((row) => {
          const cols = row.values.map((value) => {
            const isBest = highlight && value === row.best && value > 0;
            const cls = isBest ? ' class="matrix-best"' : '';
            return `<td${cls}>${value.toFixed(1)}%</td>`;
          });
          return `<tr><td><strong>${row.driver}</strong></td>${cols.join('')}</tr>`;
        })
        .join('');

      matrixEl.innerHTML = `
        <div class="matrix-wrap">
          <table class="matrix-table">
            <thead>${header}</thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }

    function renderQuickActions(runs) {
      if (!runs.length) {
        actionsEl.innerHTML = '';
        return;
      }
      const latest = runs[0];
      const files = latest.files || {};
      const links = [];
      if (files.report_html) {
        links.push(`<a href="/output/${files.report_html}" target="_blank">Open latest report</a>`);
      }
      if (files.statistics_json) {
        links.push(
          `<a href="/output/${files.statistics_json}" target="_blank">Open latest stats</a>`
        );
      }
      actionsEl.innerHTML = links.join(' · ');
    }

    function renderRunItem(run) {
      const files = run.files || {};
      const links = [];
      if (files.report_html) {
        links.push(`<a href="/output/${files.report_html}" target="_blank">report</a>`);
      }
      if (files.statistics_json) {
        links.push(`<a href="/output/${files.statistics_json}" target="_blank">stats</a>`);
      }
      if (files.race_csv) {
        links.push(`<a href="/output/${files.race_csv}" target="_blank">race csv</a>`);
      }
      const linkHtml = links.length ? links.join(' · ') : '-';
      const summary =
        `<strong>${run.track || '-'} </strong> · ` +
        `sims=${run.num_simulations ?? '-'} · seed=${run.seed ?? '-'}`;

      return `
        <div class="run-item">
          <div>${summary}</div>
          <div class="run-meta">${run.timestamp || '-'} · ${run.prefix || ''}</div>
          <div>${linkHtml}</div>
        </div>
      `;
    }

    async function refreshRuns() {
      try {
        const res = await fetch('/api/runs');
        const data = await res.json();
        const runs = data.runs || [];
        const filter = (runsFilterInput.value || '').trim().toLowerCase();
        const filteredRuns = !filter
          ? runs
          : runs.filter((run) => String(run.track || '').toLowerCase().includes(filter));

        if (!filteredRuns.length) {
          actionsEl.innerHTML = '';
          runsEl.textContent = filter ? 'No runs match this filter' : 'No runs yet';
          return;
        }
        renderQuickActions(filteredRuns);
        runsEl.innerHTML =
          `<div class="runs-list">${filteredRuns.map(renderRunItem).join('')}</div>`;
      } catch (err) {
        runsEl.textContent = JSON.stringify({ status: 'error', detail: String(err) }, null, 2);
      }
    }

    document.getElementById('presetDry').addEventListener('click', () => {
      scenariosInput.value = 'dry,cloudy';
    });
    document.getElementById('presetMixed').addEventListener('click', () => {
      scenariosInput.value = 'dry,light_rain,heavy_rain';
    });
    document.getElementById('presetWet').addEventListener('click', () => {
      scenariosInput.value = 'light_rain,heavy_rain';
    });

    function readPayloadFromInputs() {
      return {
        year: Number(document.getElementById('year').value),
        race: document.getElementById('race').value,
        simulations: Number(document.getElementById('simulations').value),
        scenarios: document.getElementById('scenarios').value,
        seed: Number(document.getElementById('seed').value),
      };
    }

    function writePayloadToInputs(payload) {
      if (!payload) return;
      if (payload.year != null) document.getElementById('year').value = payload.year;
      if (payload.race != null) document.getElementById('race').value = payload.race;
      if (payload.simulations != null) {
        document.getElementById('simulations').value = payload.simulations;
      }
      if (payload.scenarios != null) document.getElementById('scenarios').value = payload.scenarios;
      if (payload.seed != null) document.getElementById('seed').value = payload.seed;
    }

    async function runWithPayload(payload) {
      resultEl.textContent = JSON.stringify({ status: 'running', payload }, null, 2);
      try {
        const res = await fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        resultEl.textContent = JSON.stringify(data, null, 2);
        renderScenarioCards(data);
        renderWinChart(data);
        renderScenarioTrends(data);
        renderDriverMatrix(data);
        localStorage.setItem('f1sim:lastPayload', JSON.stringify(payload));
        await refreshRuns();
      } catch (err) {
        resultEl.textContent = JSON.stringify({ status: 'error', detail: String(err) }, null, 2);
      }
    }

    document.getElementById('runBtn').addEventListener('click', async () => {
      await runWithPayload(readPayloadFromInputs());
    });

    rerunBtn.addEventListener('click', async () => {
      const raw = localStorage.getItem('f1sim:lastPayload');
      if (!raw) {
        resultEl.textContent = JSON.stringify(
          { status: 'error', detail: 'No previous config saved yet' },
          null,
          2
        );
        return;
      }
      const payload = JSON.parse(raw);
      writePayloadToInputs(payload);
      await runWithPayload(payload);
    });

    clearBtn.addEventListener('click', () => {
      localStorage.removeItem('f1sim:lastPayload');
      resultEl.textContent = JSON.stringify({ status: 'saved config cleared' }, null, 2);
    });

    runsFilterInput.addEventListener('input', () => {
      refreshRuns();
    });

    matrixSortEl.addEventListener('change', () => {
      if (latestScenarioData) {
        renderDriverMatrix(latestScenarioData);
      }
    });
    matrixHighlightEl.addEventListener('change', () => {
      if (latestScenarioData) {
        renderDriverMatrix(latestScenarioData);
      }
    });
    matrixDriverFilterEl.addEventListener('input', () => {
      if (latestScenarioData) {
        renderDriverMatrix(latestScenarioData);
      }
    });
    matrixScenarioFilterEl.addEventListener('input', () => {
      if (latestScenarioData) {
        renderDriverMatrix(latestScenarioData);
      }
    });
    trendMetricEl.addEventListener('change', () => {
      if (latestScenarioData) {
        renderScenarioTrends(latestScenarioData);
      }
    });

    const saved = localStorage.getItem('f1sim:lastPayload');
    if (saved) {
      writePayloadToInputs(JSON.parse(saved));
    }

    refreshRuns();
  </script>
</body>
</html>
"""


def build_fastapi_app() -> Any:
    """Build FastAPI dashboard app.

    FastAPI import is lazy so core package remains usable without web deps.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles
    except Exception as exc:  # pragma: no cover
        msg = "FastAPI is not installed. Install with: pip install -e '.[web]'"
        raise RuntimeError(msg) from exc

    app = FastAPI(title="F1Sim Dashboard", version="0.1")
    Path("output").mkdir(parents=True, exist_ok=True)
    app.mount("/output", StaticFiles(directory="output"), name="output")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return build_dashboard_html()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        return {"runs": _read_run_history()[:20]}

    @app.post("/api/run")
    def run(payload: DashboardRunRequest) -> dict[str, Any]:
        try:
            return run_dashboard_simulation(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> int:
    """Run dashboard with uvicorn."""
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        msg = "uvicorn is not installed. Install with: pip install -e '.[web]'"
        raise RuntimeError(msg) from exc

    app = build_fastapi_app()
    uvicorn.run(app, host="127.0.0.1", port=8080)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
