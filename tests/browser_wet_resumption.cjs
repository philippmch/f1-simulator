// Real simulation evidence checks the UI's complete compulsory-wet path.
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

async function checkWetResumption(page) {
  const fixture = JSON.parse(execFileSync(process.env.PYTHON || 'python',
    [path.join(__dirname, 'wet_resumption_browser_fixture.py')],
    {encoding: 'utf8', timeout: 120000, maxBuffer: 8 * 1024 * 1024}));
  assert.equal(fixture.native, true);
  const viewport = page.viewportSize();
  const original = await page.evaluate(() => {
    window.savedWetResumptionResults = simResults;
    return {source: document.getElementById('controlSourceSelect').value,
      text: document.getElementById('controlScheduleInput').value};
  });
  await page.locator('#tab-race').click();
  try {
    await page.locator('#controlSourceSelect').selectOption('scheduled');
    await page.locator('#controlScheduleInput').fill('2:red:resume_wet\n4:vsc:1');
    assert.deepEqual((await page.evaluate(() => buildRunPayload())).control_schedule, [
      {lap: 2, control: 'red_flag', action: 'resume_wet'},
      {lap: 4, control: 'vsc', duration_laps: 1},
    ]);
    for (const invalid of ['2:red:RESUME_WET', '2:red:resume_wet\n3:vsc:1']) {
      await page.locator('#controlScheduleInput').fill(invalid);
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
      assert.equal(await page.evaluate(() => document.activeElement.id), 'controlScheduleInput');
    }
    for (const [name, scenario] of Object.entries(fixture.payload.scenarios)) {
      assert.equal(scenario.simulation_inputs.schema_version, 13);
      assert.equal(scenario.control_schedule_statistics.applied, 1);
      for (const driver of scenario.sample_race) {
        const wet = driver.tire_set_history.filter(row => row.set_id === 'W');
        assert.equal(wet.length, 1);
        assert.deepEqual([wet[0].age_at_fit, wet[0].age_at_end, wet[0].laps_used,
          wet[0].remaining_laps_at_end], [9, 10, 1, 0]);
        assert.equal(wet[0].kind, 'red_flag');
        assert.equal(driver.pit_plan_history[0].status, 'overridden');
        assert.equal(driver.pit_plan_history[0].reason, 'mandatory_wet_tires');
        assert.deepEqual(driver.pit_laps, [4]);
      }
      await page.evaluate(({payload, name, scenario}) => {
        simResults = {...payload, scenarios: {[name]: scenario}};
        renderRace();
      }, {payload: fixture.payload, name, scenario});
      await page.locator('.control-schedule-history summary').click();
      assert((await page.locator('.control-schedule-history tbody').innerText())
        .includes('Resume on compulsory full wets'));
      await page.locator('#sampleTireSetLedgers summary').click();
      const rows = page.locator('#sampleTireSetLedgers tbody tr');
      const wetRows = (await rows.allTextContents()).filter(text => text.includes('red_flag'));
      assert.equal(wetRows.length, 2);
      assert(wetRows.every(text => text.includes('wet')));
      await page.locator('#samplePitPlanHistory summary').click();
      assert((await page.locator('#samplePitPlanHistory').innerText())
        .includes('Compulsory full-wet tyres'));
      for (const width of [320, 390, 1440]) {
        await page.setViewportSize({width, height: 900});
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `Wet-resumption results must fit at ${width}px`);
      }
    }
    const download = page.waitForEvent('download');
    await page.evaluate(data => downloadControlScheduleCsv(data), fixture.payload);
    const file = await download;
    const csv = fs.readFileSync(await file.path(), 'utf8');
    assert.equal(csv.trim().split(/\r?\n/).length, 3);
    assert.equal(csv.match(/"resume_wet"/g).length, 2);
    for (const backendCsv of Object.values(fixture.csvs)) {
      assert.equal(backendCsv.trim().split(/\r?\n/).length, 2);
      assert(backendCsv.includes('resume_wet'));
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
        assert((await exported.locator('body').innerText()).includes('full-wet tyres compulsory'));
        for (const width of [390, 1440]) {
          await exported.setViewportSize({width, height: 900});
          assert(await exported.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        }
      }
      assert.deepEqual(errors, []);
    } finally {
      await exportContext.close();
    }
    console.log('Native compulsory-wet inputs, physical tyre use, CSV and HTML checks passed.');
  } finally {
    await page.evaluate(original => {
      simResults = savedWetResumptionResults;
      delete window.savedWetResumptionResults;
      document.getElementById('controlSourceSelect').value = original.source;
      document.getElementById('controlScheduleInput').value = original.text;
      updateControlScheduleControls();
      renderRace();
    }, original);
    await page.setViewportSize(viewport);
  }
}

module.exports = {checkWetResumption};
