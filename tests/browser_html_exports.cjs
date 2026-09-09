// Verify generated HTML executes only its intended chart code, entirely offline.
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
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
        if (name === 'comparison.html') {
          return route.fulfill({contentType: 'text/html; charset=utf-8', body: fixture.comparison});
        }
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
    assert((await page.locator('.meta').innerText()).includes(`Starting tyres: ${fixture.driver}=soft`));
    assert((await page.locator('#race-distance').innerText()).includes('Mean winning distance: Not recorded'));
    assert((await page.locator('#race-distance').innerText()).includes('Lapped finishers: Not recorded'));
    await page.locator('#strategy-statistics summary').click();
    assert((await page.locator('#strategy-statistics summary').innerText()).includes(fixture.driver));
    assert((await page.locator('#strategy-statistics tbody th, #strategy-statistics tbody td').first().innerText()).startsWith('soft → </script>'));
    assert((await page.locator('#strategy-statistics').innerText()).includes('1 (100.0%)'));
    assert.equal(await page.locator('img, svg').count(), 0);
    assert.equal(await page.locator('script').count(), 2);
    assert.deepEqual(await page.evaluate(() => plotCalls.map(call => call.data[0].x)),
      [[fixture.driver], [fixture.team]]);
    assert.equal(await page.evaluate(() => Boolean(globalThis.exportInjected)), false);
    await page.goto('http://f1sim.test/comparison.html');
    assert.equal(await page.getByRole('heading', {name: 'Simulation comparison', exact: true}).count(), 1);
    assert.equal(await page.locator('script, img, svg, link').count(), 0);
    assert.equal(await page.locator('details[open]').count(), 1);
    assert((await page.locator('body').innerText()).includes(fixture.track));
    assert((await page.getByRole('region', {name: 'Scenario context', exact: true}).innerText())
      .includes('weather draws independent of race decisions'));
    assert((await page.locator('details').first().innerText()).includes(fixture.driver));
    assert((await page.locator('details').first().innerText()).includes('Not recorded'));
    assert((await page.locator('details').first().innerText()).includes('100.0%'));
    const sequences = page.locator('details').first().getByRole('region').nth(1);
    assert((await sequences.innerText()).includes('Recorded tyre sequences'));
    assert((await sequences.innerText()).includes('soft → </script>'));
    assert((await sequences.innerText()).includes('1 / 1 (100.0%)'));
    const distance = page.getByRole('region', {name: 'Race distance', exact: true});
    assert((await distance.innerText()).includes('Mean winning distance'));
    assert((await distance.innerText()).includes('Not recorded'));
    assert((await distance.innerText()).includes('(0 / 1 recorded races)'));
    assert.equal(await page.evaluate(() => Boolean(globalThis.exportInjected)), false);
    for (const width of [320, 390, 1440]) {
      await page.setViewportSize({width, height: 1100});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      await distance.focus();
      assert(await distance.evaluate(node => document.activeElement === node));
      const summary = page.locator('details summary').first();
      await summary.focus();
      await page.keyboard.press('Enter');
      assert.equal(await page.locator('details[open]').count(), 0);
      await page.keyboard.press('Enter');
      assert.equal(await page.locator('details[open]').count(), 1);
      if (process.env.F1SIM_SCREENSHOTS && width !== 320) {
        fs.mkdirSync(process.env.F1SIM_SCREENSHOTS, {recursive: true});
        await page.screenshot({path: path.join(process.env.F1SIM_SCREENSHOTS,
          `comparison-${width}.png`), fullPage: true});
      }
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, []);
    console.log('HTML export browser checks passed: text, script data, and filename links.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
