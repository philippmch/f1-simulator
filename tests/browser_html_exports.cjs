// Verify generated HTML executes only its intended chart code, entirely offline.
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const fixture = JSON.parse(execFileSync(process.env.PYTHON || 'python',
    [path.join(__dirname, 'html_export_fixture.py')], {encoding: 'utf8'}));
  const browser = await chromium.launch({headless: true, channel: process.env.BROWSER_CHANNEL});
  try {
    const page = await browser.newPage();
    const errors = [], unexpected = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.hostname === 'cdn.plot.ly') return route.fulfill({contentType: 'text/javascript',
        body: 'window.plotCalls=[];window.Plotly={newPlot:(id,data)=>plotCalls.push({id,data})};'});
      if (url.hostname === 'f1sim.test') {
        const name = decodeURIComponent(url.pathname.slice(1));
        if (name === 'index.html' || name === fixture.filename) {
          return route.fulfill({contentType: 'text/html; charset=utf-8',
            body: name === 'index.html' ? fixture.index : fixture.report});
        }
      }
      unexpected.push(url.href);
      return route.abort();
    });
    await page.goto('http://f1sim.test/index.html');
    assert.equal(await page.locator('tbody tr').count(), 1);
    assert((await page.locator('tbody').innerText()).includes(fixture.track));
    assert((await page.locator('thead').innerText()).includes('Race model'));
    assert((await page.locator('tbody').innerText()).includes('Standard'));
    assert.equal(await page.locator('img, svg, script').count(), 0);
    const links = await page.locator('tbody a').evaluateAll(nodes => nodes.map(n => n.href));
    assert.equal(links.length, 2);
    for (const link of links) assert(link.startsWith('http://f1sim.test/'));
    assert.equal(decodeURIComponent(new URL(links[1]).pathname.slice(1)), fixture.stats_name);
    await page.getByRole('link', {name: 'report', exact: true}).click();
    await page.waitForFunction(() => window.plotCalls?.length === 2);
    assert.equal(await page.title(), `F1 Simulation Report - ${fixture.track}`);
    assert((await page.locator('.meta').innerText()).includes(fixture.track));
    assert.equal(await page.locator('img, svg').count(), 0);
    assert.equal(await page.locator('script').count(), 2);
    assert.deepEqual(await page.evaluate(() => plotCalls.map(call => call.data[0].x)),
      [[fixture.driver], [fixture.team]]);
    assert.equal(await page.evaluate(() => Boolean(globalThis.exportInjected)), false);
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, []);
    console.log('HTML export browser checks passed: text, script data, and filename links.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
