// Native race decisions drive the UI; malformed/hostile cases only test presentation.
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

async function checkAbandonment(page) {
  const fixture = JSON.parse(execFileSync(process.env.PYTHON || 'python',
    [path.join(__dirname, 'abandonment_browser_fixture.py')],
    {encoding: 'utf8', timeout: 120000, maxBuffer: 8 * 1024 * 1024}));
  assert.equal(fixture.native, true);
  const viewport = page.viewportSize();
  const original = await page.evaluate(() => {
    window.savedAbandonmentResults = simResults;
    return {source: document.getElementById('controlSourceSelect').value,
      text: document.getElementById('controlScheduleInput').value};
  });
  await page.locator('#tab-race').click();
  try {
    await page.locator('#controlSourceSelect').selectOption('scheduled');
    await page.locator('#controlScheduleInput').fill('1:sc:5\n3:red:resume\n5:red:abandon');
    assert.deepEqual((await page.evaluate(() => buildRunPayload())).control_schedule, [
      {lap: 1, control: 'safety_car', duration_laps: 5},
      {lap: 3, control: 'red_flag', action: 'resume'},
      {lap: 5, control: 'red_flag', action: 'abandon'},
    ]);
    for (const invalid of ['2:red:restart', '2:red:ABANDON', '2:red:2',
      '2:red:resume\n3:vsc:1']) {
      await page.locator('#controlScheduleInput').fill(invalid);
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
    }
    for (const [name, scenario] of Object.entries(fixture.payload.scenarios)) {
      await page.evaluate(({payload, name, scenario}) => {
        simResults = {...payload, scenarios: {[name]: scenario}};
        renderRace();
      }, {payload: fixture.payload, name, scenario});
      const context = scenario.sample_race_abandonment_context;
      assert((await page.locator('.abandonment-note').innerText()).includes(context.description));
      assert((await page.locator('#raceContent .race-header').innerText()).includes(
        `${context.countback_lap} of 8 scheduled laps (countback)`));
      const rows = page.locator('#raceContent .driver-row');
      assert.equal(await rows.count(), 2);
      if (context.countback_lap === 0) {
        assert.equal(await page.locator('#raceContent .dnf, #raceContent .podium').count(), 0);
        for (const row of await rows.all()) {
          assert((await row.locator('.status').innerText()).startsWith('No result'));
          assert.equal(await row.locator('.pos').innerText(), 'NC');
          assert.equal(await row.locator('.gap').innerText(), '—');
          assert(!(await row.innerText()).includes('Retired'));
        }
      } else {
        for (const row of await rows.all()) {
          assert((await row.locator('.status').innerText()).includes('+30s dry-compound penalty'));
          assert((await row.locator('.status').getAttribute('aria-label')).includes('+30s'));
        }
      }
      await page.locator('.control-schedule-history summary').click();
      assert((await page.locator('.control-schedule-history').innerText()).includes('Race-control'));
      assert((await page.locator('.control-schedule-history tbody').innerText()).includes('abandon'));
      for (const width of [320, 390, 1440]) {
        await page.setViewportSize({width, height: 900});
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `Countback results must fit at ${width}px`);
      }
    }
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      scenario.sample_race_abandonment_context.description =
        '<img src=x onerror="globalThis.abandonmentInjected=true">';
      scenario.control_schedule_statistics.entries = {};
      renderRace();
    });
    assert.equal(await page.locator('.abandonment-note img').count(), 0);
    assert.equal(await page.evaluate(() => Boolean(globalThis.abandonmentInjected)), false);
    const download = page.waitForEvent('download');
    await page.evaluate(data => downloadControlScheduleCsv(data), fixture.payload);
    const file = await download;
    const csv = fs.readFileSync(await file.path(), 'utf8');
    assert.equal(csv.trim().split(/\r?\n/).length, 5);
    assert(csv.split(/\r?\n/)[0].includes('"action"'));
    assert.equal(csv.match(/"abandon"/g).length, 4);
    for (const backendCsv of Object.values(fixture.csvs)) {
      assert(backendCsv.split(/\r?\n/)[0].includes('action'));
      assert.equal(backendCsv.trim().split(/\r?\n/).length, 2);
    }
    const exportContext = await page.context().browser().newContext();
    try {
      const exported = await exportContext.newPage();
      const errors = [];
      exported.on('pageerror', error => errors.push(error.message));
      await exported.route('**/*', route => route.fulfill({contentType: 'text/javascript',
        body: 'window.Plotly={newPlot:()=>{}};'}));
      for (const html of [...Object.values(fixture.reports), fixture.comparison]) {
        await exported.setContent(html);
        assert((await exported.locator('body').innerText()).includes('Recorded abandoned races: 1'));
        assert(await exported.locator('.abandonment-evidence').count());
        for (const width of [390, 1440]) {
          await exported.setViewportSize({width, height: 900});
          assert(await exported.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        }
      }
      assert.deepEqual(errors, []);
    } finally {
      await exportContext.close();
    }
    console.log('Native countback, no-result labels, tyre penalties and abandonment exports passed.');
  } finally {
    await page.evaluate(original => {
      simResults = savedAbandonmentResults;
      delete window.savedAbandonmentResults;
      document.getElementById('controlSourceSelect').value = original.source;
      document.getElementById('controlScheduleInput').value = original.text;
      updateControlScheduleControls();
      renderRace();
    }, original);
    await page.setViewportSize(viewport);
  }
}

module.exports = {checkAbandonment};
