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
