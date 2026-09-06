// Run against a local server. Requires Playwright and a Chromium browser.
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const browser = await chromium.launch({
    headless: true,
    channel: process.env.BROWSER_CHANNEL || undefined,
  });
  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(120000);
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(process.env.F1SIM_URL || 'http://127.0.0.1:8080');
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
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      for (const tab of ['race', 'qualifying', 'stats', 'scenarios']) {
        await page.locator(`#tab-${tab}`).click();
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `${tab} overflows at ${width}px`);
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
    await page.route('**/api/run', route => route.fulfill({
      status: 503, contentType: 'application/json',
      body: JSON.stringify({detail: 'Provider temporarily unavailable'}),
    }));
    await page.locator('#btnRun').click();
    await page.waitForFunction(() => !runInProgress);
    assert((await page.locator('#appStatus').innerText()).includes('Provider temporarily unavailable'));
    assert(await page.locator('#btnRun').isEnabled());
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
    console.log('Browser audit passed: live run, responsive views, exports, tabs, filters, recovery.');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
