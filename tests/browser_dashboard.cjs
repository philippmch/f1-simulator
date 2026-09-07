// Run against a local server. Requires Playwright and a Chromium browser.
// F1SIM_OFFLINE=1 generates synthetic inputs with PYTHON (default: python),
// intercepts all requests, and needs neither a server nor network access.
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
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
    await page.locator('#parallelSelect').selectOption('false');
    await page.evaluate(() => setScenarioSelection(['dry', 'light_rain', 'heavy_rain']));
    const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/run'));
    await page.locator('#btnRun').click();
    const response = await responsePromise;
    assert.equal(response.status(), 200);
    const payload = await response.json();
    const driverId = Object.values(payload.scenarios)[0].sample_race[0].driver_id;
    await page.waitForFunction(() => !runInProgress);
    await page.locator('#tab-stats').click();
    await page.locator('#probabilityIntervals summary').click();
    const statistics = Object.values(payload.scenarios)[0].driver_statistics;
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
    await page.locator('#tab-stats').click();
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      for (const tab of ['race', 'qualifying', 'stats', 'scenarios']) {
        await page.locator(`#tab-${tab}`).click();
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `${tab} overflows at ${width}px`);
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
    for (const [id, extension] of [
      ['downloadScenarioJsonBtn', '.json'], ['downloadScenarioMatrixBtn', '.csv'],
    ]) {
      const downloadPromise = page.waitForEvent('download');
      await page.locator(`#${id}`).click();
      const download = await downloadPromise;
      assert(download.suggestedFilename().endsWith(extension));
    }
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
