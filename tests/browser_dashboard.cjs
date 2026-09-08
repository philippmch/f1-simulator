// Run against a local server. Requires Playwright and a Chromium browser.
// F1SIM_OFFLINE=1 generates synthetic inputs with PYTHON (default: python),
// intercepts all requests, and needs neither a server nor network access.
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const offline = process.env.F1SIM_OFFLINE === '1';
  const fixture = offline ? JSON.parse(execFileSync(process.env.PYTHON || 'python',
    [path.join(__dirname, 'browser_fixture.py')], {encoding: 'utf8', maxBuffer: 8 * 1024 * 1024,
      timeout: 120000})) : null;
  const browser = await chromium.launch({
    headless: true,
    channel: process.env.BROWSER_CHANNEL || undefined,
  });
  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(120000);
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    const unexpectedRequests = [];
    if (offline) {
      await page.route('**/*', route => {
        const url = new URL(route.request().url());
        if (url.hostname !== 'f1sim.test') {
          // The dashboard uses external fonts; system fallbacks keep this test offline.
          if (!['fonts.googleapis.com', 'fonts.gstatic.com'].includes(url.hostname)) {
            unexpectedRequests.push(url.href);
          }
          return route.abort();
        }
        const body = url.pathname === '/api/run' ? fixture.payload
          : url.pathname === '/api/calendar' ? fixture.calendar
          : url.pathname === '/api/health' ? {status: 'ok', season: 2026} : null;
        if (body) return route.fulfill({json: body});
        if (url.pathname === '/') return route.fulfill({
          contentType: 'text/html', body: fixture.html,
        });
        if (url.pathname !== '/favicon.ico') unexpectedRequests.push(url.href);
        return route.fulfill({status: 404, body: ''});
      });
    }
    await page.goto(offline ? 'http://f1sim.test' :
      process.env.F1SIM_URL || 'http://127.0.0.1:8080');
    await page.waitForFunction(() => !connectionRefreshInProgress);
    assert(await page.locator('#btnRun').isEnabled(), 'Live calendar must be available');
    await page.locator('#simCount').fill('10');
    assert.equal(await page.locator('#raceEngineSelect').inputValue(), 'standard');
    await page.locator('#raceEngineSelect').selectOption('chronological');
    await page.locator('#parallelSelect').selectOption('false');
    await page.locator('#startingTiresInput').fill('S00=soft,S00=hard');
    assert.equal(await page.evaluate(() => buildRunPayload()), null);
    assert((await page.locator('#appStatus').innerText()).includes('use each driver once'));
    await page.locator('#startingTiresInput').fill(offline ? 'S00=hard, S01=soft' : '');
    await page.evaluate(() => setScenarioSelection(['dry', 'light_rain', 'heavy_rain']));
    const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/run'));
    await page.locator('#btnRun').click();
    const response = await responsePromise;
    assert.equal(response.status(), 200);
    assert.equal(response.request().postDataJSON().race_engine, 'chronological');
    assert.deepEqual(response.request().postDataJSON().starting_tires,
      offline ? {S00: 'hard', S01: 'soft'} : {});
    const payload = await response.json();
    assert.equal(payload.request.race_engine, 'chronological');
    // The matrix initially shows aggregate top contenders, which need not
    // include the winner of the representative individual race.
    const driverId = Object.values(payload.scenarios)[0].win_probabilities[0][0];
    await page.waitForFunction(() => !runInProgress);
    assert((await page.locator('#panel-race').textContent()).includes('Lap-aware model (experimental)'));
    if (offline) {
      assert((await page.locator('#panel-race').textContent()).includes('S00=hard, S01=soft'));
      for (const scenario of Object.values(payload.scenarios)) {
        assert.deepEqual(scenario.simulation_inputs.starting_tires, {S00: 'hard', S01: 'soft'});
      }
      assert(payload.scenarios.dry.strategy_statistics.S00.strategies.every(row => row.compounds[0] === 'hard'));
      // Critical weather corrections still replace an unsuitable unrun set.
      assert(payload.scenarios.heavy_rain.strategy_statistics.S00.strategies.every(row => row.compounds[0] === 'wet'));
    }
    await page.locator('#tab-stats').click();
    await page.locator('#probabilityIntervals summary').click();
    const statistics = Object.values(payload.scenarios)[0].driver_statistics;
    const distance = Object.values(payload.scenarios)[0].race_distance_statistics;
    assert.equal(distance.recorded_races, 10);
    assert((await page.locator('#raceDistanceStatistics').innerText()).includes(
      `${distance.mean_winner_laps.toFixed(1)} laps`));
    assert((await page.locator('#raceDistanceStatistics').innerText()).includes(
      `${distance.lapped_finishers} of ${distance.finishers_with_comparable_distance} finishers with known distance`));
    assert.equal(await page.locator('#probabilityIntervals tbody tr').count(),
      Object.keys(statistics).length);
    assert((await page.locator('#probabilityIntervals').innerText()).includes(
      'These ranges do not measure model accuracy'));
    for (const [id, stats] of Object.entries(statistics)) {
      const row = page.locator('#probabilityIntervals tbody tr').filter({
        has: page.locator('th', {hasText: new RegExp(`^${id}$`)}),
      });
      assert.equal(await row.locator('td').first().innerText(), '10');
      const interval = stats.probability_intervals.win;
      assert((await row.innerText()).includes(
        `${interval.lower.toFixed(1)}–${interval.upper.toFixed(1)}%`));
      if (stats.win_rate === 0) assert(interval.upper > 0);
    }
    const pitStatistics = Object.values(payload.scenarios)[0].pit_stop_statistics;
    const strategyStatistics = Object.values(payload.scenarios)[0].strategy_statistics;
    assert.equal(await page.locator('#strategyStatistics details').count(),
      Object.keys(strategyStatistics).length);
    const [sequenceId, sequenceStats] = Object.entries(strategyStatistics).sort(([a], [b]) => a.localeCompare(b))[0];
    const sequenceDetails = page.locator('#strategyStatistics details').first();
    await sequenceDetails.locator('summary').click();
    assert((await sequenceDetails.innerText()).includes(`${sequenceId}: ${sequenceStats.races_with_recorded_strategy} recorded of ${sequenceStats.races}`));
    assert.equal(await sequenceDetails.locator('tbody tr').count(), sequenceStats.strategies.length);
    assert.equal(await sequenceDetails.locator('tbody th').first().innerText(),
      sequenceStats.strategies[0].compounds.join(' → '));
    await page.evaluate(() => {
      window.savedStrategyStatistics = getScenarioEntry().data.strategy_statistics;
      getScenarioEntry().data.strategy_statistics = {'<img src=x>': {
        races: 3, races_with_recorded_strategy: 2, missing_strategy_races: 1,
        strategies: [{compounds: ['soft', '<svg onload=alert(1)>', 'soft'], races: 2,
          share: 1, finished_races: 1, dnf_races: 1}],
      }};
      renderStats();
    });
    await page.locator('#strategyStatistics summary').click();
    assert.equal(await page.locator('#strategyStatistics img, #strategyStatistics svg').count(), 0);
    assert((await page.locator('#strategyStatistics').innerText()).includes('Missing tyre sequences: 1'));
    assert.equal(await page.locator('#strategyStatistics tbody th').innerText(),
      'soft → <svg onload=alert(1)> → soft');
    assert.deepEqual(await page.locator('#strategyStatistics tbody td').allTextContents(),
      ['2 (100.0%)', '1', '1']);
    await page.evaluate(() => { delete getScenarioEntry().data.strategy_statistics; renderStats(); });
    assert((await page.locator('#strategyStatistics').innerText()).includes('No tyre sequences were recorded'));
    await page.evaluate(() => {
      getScenarioEntry().data.strategy_statistics = savedStrategyStatistics;
      delete window.savedStrategyStatistics;
      renderStats();
    });
    assert.equal(await page.locator('#pitStopStatistics tbody tr').count(),
      Object.keys(pitStatistics).length);
    for (const [id, stats] of Object.entries(pitStatistics)) {
      const row = page.locator('#pitStopStatistics tbody tr').filter({
        has: page.locator('th', {hasText: new RegExp(`^${id}$`)}),
      });
      assert.equal(await row.locator('td').nth(0).innerText(), String(stats.races));
      assert.equal(await row.locator('td').nth(1).innerText(), stats.average_stops.toFixed(2));
    }
    // Group high-stop outcomes and preserve a real zero; escape driver labels.
    await page.evaluate(() => {
      window.savedPitStatistics = getScenarioEntry().data.pit_stop_statistics;
      getScenarioEntry().data.pit_stop_statistics = {'<img src=x>': {
        races: 4, average_stops: 2.25, stop_count_distribution: {0: 1, 1: 1, 3: 1, 5: 1},
      }};
      renderStats();
    });
    assert.equal(await page.locator('#pitStopStatistics img').count(), 0);
    assert.equal(await page.locator('#pitStopStatistics tbody th').textContent(), '<img src=x>');
    assert.deepEqual(await page.locator('#pitStopStatistics tbody td').allTextContents(),
      ['4', '2.25', '25.0%', '25.0%', '0.0%', '50.0%']);
    await page.evaluate(() => {
      getScenarioEntry().data.pit_stop_statistics = {};
      renderStats();
    });
    assert.equal(await page.locator('#pitStopStatistics').count(), 0);
    assert((await page.locator('.pit-summary-card').innerText()).includes('No aggregate pit-stop observations'));
    await page.evaluate(() => {
      getScenarioEntry().data.pit_stop_statistics = savedPitStatistics;
      delete window.savedPitStatistics;
      renderStats();
    });
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      window.savedDistance = scenario.race_distance_statistics;
      delete scenario.race_distance_statistics;
      renderStats();
    });
    assert((await page.locator('#raceDistanceStatistics').innerText()).includes('not recorded'));
    await page.evaluate(() => {
      getScenarioEntry().data.race_distance_statistics = savedDistance;
      delete window.savedDistance;
      renderStats();
    });
    // Explicit classified and unclassified retirements must remain distinct.
    await page.locator('#tab-race').click();
    for (const row of Object.values(payload.scenarios)[0].sample_race) {
      assert(Array.isArray(row.pit_laps), 'Current race results must include actual pit laps');
      assert.equal(row.pit_laps.length, row.pit_stops);
    }
    await page.evaluate(() => {
      window.savedClassificationSample = getScenarioEntry().data.sample_race;
      getScenarioEntry().data.sample_race = savedClassificationSample.slice(0, 3).map((row, index) => ({
        ...row, position: index + 1, status: index === 0 ? 'finished' : 'dnf',
        classified: index < 2, laps_completed: [60, 54, 53][index],
        fastest_lap: [90, 89, 91][index], dnf_reason: index ? 'Engine failure' : null,
        pit_stops: [3, 0, 1][index], pit_laps: [[7, 17, 34], [], null][index],
      }));
      renderRace();
    });
    const classifiedRetirement = page.locator('#raceContent .driver-row').nth(1);
    assert.equal(await classifiedRetirement.locator('.pos').innerText(), '2');
    assert((await classifiedRetirement.locator('.status').getAttribute('aria-label'))
      .includes('Retired · Classified · 54 laps · Engine failure'));
    assert((await classifiedRetirement.getAttribute('class')).includes('podium'));
    assert.equal(await classifiedRetirement.locator('.is-fastest').count(), 1);
    const unclassifiedRetirement = page.locator('#raceContent .driver-row').nth(2);
    assert.equal(await unclassifiedRetirement.locator('.pos').innerText(), 'NC');
    assert(!(await unclassifiedRetirement.getAttribute('class')).includes('podium'));
    assert.equal(await page.locator('#raceContent .pit-laps').first().innerText(), 'L7 · L17 · L34');
    assert.equal(await classifiedRetirement.locator('.pits').getAttribute('title'), 'No paid pit stops');
    assert.equal(await unclassifiedRetirement.locator('.pits').getAttribute('title'), 'Pit laps unavailable');
    assert.equal(await page.locator('#raceContent .time-limit-note').count(), 0);
    await page.evaluate(() => {
      getScenarioEntry().data.sample_race.forEach((row, index) => {
        row.race_time_limited = true;
        row.laps_completed = [5, 4, 3][index];
        row.points_awarded = [19, 12, 0][index];
      });
      renderRace();
    });
    assert((await page.locator('#raceContent .time-limit-note').innerText()).includes('two-hour limit'));
    assert((await page.locator('#raceContent .race-header').innerText()).includes('5 of'));
    assert((await page.locator('#raceContent .driver-row .status').first().innerText()).includes('19 pts'));
    assert((await page.locator('#raceContent .driver-row .status').nth(2).innerText()).includes('0 pts'));
    assert.equal(await page.locator('#raceContent .driver-row .gap').nth(1).innerText(), 'DNF');
    await page.evaluate(() => {
      getScenarioEntry().data.sample_race.forEach(row => {
        row.status = 'finished';
        row.classified = true;
        row.dnf_reason = null;
      });
      renderRace();
    });
    assert.equal(await page.locator('#raceContent .driver-row .gap').nth(1).innerText(), '+1 lap');
    assert.equal(await page.locator('#raceContent .driver-row .gap').nth(2).innerText(), '+2 laps');
    assert.equal(await page.evaluate(() => formatGap({
      position: 2, status: 'finished', gap_to_leader: 90,
    }, 5)), '+90.000');
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({width, height: 900});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
        `Retirement classification overflows at ${width}px`);
      assert(await page.locator('#raceContent .pit-laps').first().isVisible(),
        `Pit laps must remain visible at ${width}px`);
      if (process.env.F1SIM_SCREENSHOT_DIR && [390, 1440].includes(width)) {
        await page.screenshot({path: path.join(process.env.F1SIM_SCREENSHOT_DIR,
          `pit-laps-${width}.png`), fullPage: true, animations: 'disabled'});
      }
    }
    await page.evaluate(() => {
      getScenarioEntry().data.sample_race = savedClassificationSample;
      delete window.savedClassificationSample;
      renderRace();
    });
    await page.locator('#tab-qualifying').click();
    await page.evaluate(() => {
      window.savedQualifyingSample = getScenarioEntry().data.sample_qualifying;
      getScenarioEntry().data.sample_qualifying = [...savedQualifyingSample, {
        driver_id: 'MISSING', driver_name: 'Missing car', team: 'Unknown',
        position: savedQualifyingSample.length + 1, best_time: null,
        q1_time: null, q2_time: null, q3_time: null, eliminated_in: 'Q1',
      }];
      renderQualifying();
    });
    const missingQualifying = page.locator('#qualiContent .quali-row').filter({
      has: page.locator('.driver-code', {hasText: /^MISSING$/}),
    });
    assert.deepEqual((await missingQualifying.locator('.quali-time').allTextContents())
      .map(text => text.trim()), ['--', '--', '--']);
    assert((await missingQualifying.getAttribute('class')).includes('eliminated'));
    assert.equal(await missingQualifying.locator('.best').count(), 0);
    await page.evaluate(() => {
      getScenarioEntry().data.sample_qualifying = savedQualifyingSample;
      delete window.savedQualifyingSample;
      renderQualifying();
    });
    await page.locator('#tab-stats').click();
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      for (const tab of ['race', 'qualifying', 'stats', 'scenarios']) {
        await page.locator(`#tab-${tab}`).click();
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `${tab} overflows at ${width}px`);
        if (tab === 'scenarios') {
          assert(await page.locator('#compareChart .bar-track').first().evaluate(
            node => node.getBoundingClientRect().height >= 10), 'Win bars must have visible height');
          assert(await page.locator('#compareTrends .bar-track').first().evaluate(
            node => node.getBoundingClientRect().height >= 10), 'Event bars must have visible height');
          if (process.env.F1SIM_SCREENSHOTS && [390, 1440].includes(width)) {
            await page.locator('#compareChart').screenshot({
              path: path.join(process.env.F1SIM_SCREENSHOTS, `scenario-ranges-${width}.png`),
            });
          }
        }
        if (tab === 'stats' && process.env.F1SIM_SCREENSHOTS && [390, 1440].includes(width)) {
          if (width === 1440) await page.setViewportSize({width, height: 1600});
          await page.locator('#strategyStatistics details').first().evaluate(node => { node.open = true; });
          await page.locator('#strategyStatistics').screenshot({
            path: path.join(process.env.F1SIM_SCREENSHOTS, `tyre-sequences-${width}.png`),
          });
          if (width === 1440) await page.setViewportSize({width, height: 1300});
          await page.locator('.pit-summary-card').evaluate(card => card.scrollIntoView({block: 'center'}));
          await page.locator('.pit-summary-card').screenshot({
            path: path.join(process.env.F1SIM_SCREENSHOTS, `pit-stops-${width}.png`),
          });
          if (width === 1440) await page.setViewportSize({width, height: 900});
        }
        if (tab === 'race' || tab === 'qualifying') {
          assert(await page.locator('.tab-panel.active .team-stripe').first().evaluate(stripe =>
            stripe.getBoundingClientRect().right <= stripe.nextElementSibling.getBoundingClientRect().left),
          `Team stripe overlaps driver code at ${width}px`);
        }
      }
    }
    assert.equal(await page.evaluate(() => formatGap({position: 2, gap_to_leader: 0})), '+0.000');
    assert.equal(await page.evaluate(() => formatGap({position: 1, gap_to_leader: 0})), 'LEADER');
    assert.equal(await page.locator('#scenarioContent pre').count(), 0);
    const firstScenario = Object.keys(payload.scenarios)[0];
    const firstChartDriver = payload.scenarios[firstScenario].win_probabilities[0][0];
    const expectedRange = payload.scenarios[firstScenario].driver_statistics[firstChartDriver].probability_intervals;
    const firstChartRow = page.locator('#compareChart .chart-row').first();
    assert((await firstChartRow.innerText()).includes(`${expectedRange.trials} observed trials`));
    const extent = await firstChartRow.locator('.bar-range').evaluate(node => {
      const outer = node.parentElement.getBoundingClientRect(), inner = node.getBoundingClientRect();
      return [(inner.left-outer.left)/outer.width*100, inner.width/outer.width*100];
    });
    assert(Math.abs(extent[0]-expectedRange.win.lower) < .1);
    assert(Math.abs(extent[1]-(expectedRange.win.upper-expectedRange.win.lower)) < .1);
    await page.evaluate(() => {
      window.originalComparisonScenarios = simResults.scenarios;
      simResults.scenarios = {
        observed: {win_probabilities: [['TEST', 0]], driver_statistics: {TEST: {
          probability_intervals: {trials: 4, confidence: .95, method: 'wilson',
            scope: 'monte_carlo_sampling', win: {lower: 0, upper: 49}},
        }}},
        missing: {win_probabilities: []},
        empty: {win_probabilities: [['TEST', 0]], driver_statistics: {TEST: {
          probability_intervals: {trials: 0},
        }}},
        legacy: {win_probabilities: [['TEST', 50]]},
      };
      renderScenarioWinChart(simResults);
      renderDriverMatrix(simResults);
    });
    assert.equal(await page.locator('#compareChart .bar-range').count(), 1);
    assert.equal(await page.locator('#compareChart .bar-track').count(), 2);
    const matrixCells = await page.locator('#compareMatrix tbody tr').first().locator('td').allTextContents();
    assert(matrixCells[1].includes('0.0%') && matrixCells[1].includes('4 observed trials'));
    assert.deepEqual(matrixCells.slice(2, 4), ['Not recorded', 'Not recorded']);
    assert(matrixCells[4].includes('50.0%') && matrixCells[4].includes('Sampling range unavailable'));
    const missingCsvPromise = page.waitForEvent('download');
    await page.locator('#downloadScenarioMatrixBtn').click();
    const missingCsv = await missingCsvPromise;
    assert(readFileSync(await missingCsv.path(), 'utf8').includes('"TEST","0.0","","","50.0"'));
    await page.evaluate(() => {
      simResults.scenarios = window.originalComparisonScenarios;
      delete window.originalComparisonScenarios;
      updateScenarioViews();
    });
    for (const [id, extension] of [
      ['downloadScenarioJsonBtn', '.json'], ['downloadScenarioMatrixBtn', '.csv'],
      ['downloadScenarioReportBtn', '.html'],
    ]) {
      const downloadPromise = page.waitForEvent('download');
      await page.locator(`#${id}`).click();
      const download = await downloadPromise;
      assert(download.suggestedFilename().endsWith(extension));
      if (extension === '.json') {
        const saved = JSON.parse(readFileSync(await download.path(), 'utf8'));
        assert.equal(saved.comparison_report_html, undefined);
        for (const [name, scenario] of Object.entries(saved.scenarios)) {
          assert.deepEqual(scenario.simulation_inputs, payload.scenarios[name].simulation_inputs);
          assert.deepEqual(scenario.strategy_statistics, payload.scenarios[name].strategy_statistics);
          assert.equal(scenario.simulation_inputs.schema_version, 1);
        }
      } else if (extension === '.html') {
        const report = readFileSync(await download.path(), 'utf8');
        assert.equal(report, payload.comparison_report_html);
        assert(report.includes('Simulation comparison'));
        assert(!report.includes('<script'));
      }
    }
    await page.evaluate(() => {
      window.savedComparisonReport = simResults.comparison_report_html;
      delete simResults.comparison_report_html;
      updateScenarioViews();
    });
    assert(await page.locator('#downloadScenarioReportBtn').isDisabled());
    await page.evaluate(() => {
      simResults.comparison_report_html = window.savedComparisonReport;
      delete window.savedComparisonReport;
      updateScenarioViews();
    });
    assert(await page.locator('#downloadScenarioReportBtn').isEnabled());
    await page.locator('#compareDriverFilter').fill(driverId);
    assert((await page.locator('#compareMatrix').innerText()).includes(driverId));
    await page.locator('#tab-race').focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('[role=tab][aria-selected=true]').getAttribute('id'),
      'tab-qualifying');
    for (const [status, detail] of [
      [503, 'Provider temporarily unavailable'], [429, 'Simulation capacity is busy'],
    ]) {
      await page.route('**/api/run', route => route.fulfill({
        status, contentType: 'application/json', body: JSON.stringify({detail}),
      }));
      await page.locator('#btnRun').click();
      await page.waitForFunction(() => !runInProgress);
      assert((await page.locator('#appStatus').innerText()).includes(detail));
      assert(await page.locator('#btnRun').isEnabled());
    }
    await page.route('**/api/run', route => route.abort());
    await page.locator('#btnRun').click();
    await page.waitForFunction(() => !runInProgress);
    assert(await page.locator('#btnRun').isDisabled());
    await page.locator('#btnRefresh').click();
    await page.waitForFunction(() => !connectionRefreshInProgress);
    assert(await page.locator('#btnRun').isEnabled());
    await page.route('**/api/calendar*', route => route.fulfill({
      status: 503, contentType: 'application/json', body: '{"detail":"Calendar unavailable"}',
    }));
    await page.locator('#btnRefresh').click();
    await page.waitForFunction(() => !connectionRefreshInProgress);
    assert(await page.locator('#btnRun').isDisabled());
    await page.unroute('**/api/calendar*');
    await page.locator('#btnRefresh').click();
    await page.waitForFunction(() => !connectionRefreshInProgress);
    assert(await page.locator('#btnRun').isEnabled());
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpectedRequests, [], 'Offline test encountered unexpected network traffic');
    console.log(`Browser audit passed (${offline ? 'SYNTHETIC offline' : 'live'}): ` +
      'run, responsive views, exports, tabs, filters, recovery.');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
