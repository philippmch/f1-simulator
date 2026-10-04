// Use fresh native execution evidence; parser/escape checks are separate synthetic inputs.
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

async function checkControlSchedules(page) {
  const fixture = JSON.parse(execFileSync(process.env.PYTHON || 'python',
    [path.join(__dirname, 'control_schedule_browser_fixture.py')],
    {encoding: 'utf8', timeout: 120000, maxBuffer: 8 * 1024 * 1024}));
  assert.equal(fixture.native, true);
  const original = await page.evaluate(() => ({
    source: document.getElementById('controlSourceSelect').value,
    text: document.getElementById('controlScheduleInput').value,
  }));
  const viewport = page.viewportSize();
  const hostId = 'control-schedule-browser-evidence';
  try {
    assert(!Object.hasOwn(await page.evaluate(() => buildRunPayload()), 'control_schedule'));
    await page.locator('#controlSourceSelect').selectOption('scheduled');
    await page.locator('#controlScheduleInput').fill('');
    assert.deepEqual((await page.evaluate(() => buildRunPayload())).control_schedule, []);
    await page.locator('#controlScheduleInput').fill('2:sc:2\n6:vsc:1\n12:sc:3');
    const request = await page.evaluate(() => buildRunPayload());
    assert.deepEqual(request.control_schedule, fixture.schedule);
    assert(!Object.hasOwn(request, 'ok') && !Object.hasOwn(request, 'payload'));
    for (const invalid of ['0:sc:1', '2:SC:1', '2:vsc:7', '2.0:sc:1',
      '2:sc:2\n4:vsc:1', '2:sc:true', '2:sc:1\ninvalid', '2:sc:1,,4:vsc:1']) {
      await page.locator('#controlScheduleInput').fill(invalid);
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
      assert.equal(await page.evaluate(() => document.activeElement.id), 'controlScheduleInput');
    }
    await page.locator('#controlScheduleInput').fill('2:sc:2\n6:vsc:1\n12:sc:3');
    await page.locator('#controlSourceSelect').selectOption('automatic');
    assert(!Object.hasOwn(await page.evaluate(() => buildRunPayload()), 'control_schedule'));
    await page.evaluate(({id, payload}) => {
      const host = document.createElement('section');
      host.id = id;
      host.innerHTML = renderControlScheduleEvidence(payload.scenarios.controlled);
      document.body.append(host);
    }, {id: hostId, payload: fixture.payload});
    const host = page.locator('#' + hostId);
    await host.locator('summary').click();
    const stats = fixture.payload.scenarios.controlled.control_schedule_statistics;
    assert.equal(stats.valid_history_races, 2);
    assert((await host.innerText()).includes('2 complete, 0 missing, 0 invalid'));
    assert.equal(await host.locator('tbody tr').count(), fixture.schedule.length);
    const table = host.locator('[role=region]');
    await table.focus();
    assert(await table.evaluate(node => document.activeElement === node));
    for (const width of [390, 1440]) {
      await page.setViewportSize({width, height: 900});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
      if (process.env.F1SIM_SCREENSHOTS) await host.screenshot({
        path: path.join(process.env.F1SIM_SCREENSHOTS, `control-schedule-${width}.png`),
        animations: 'disabled',
      });
    }
    // Saved context and outcome cells must remain escaped, even if a payload is hostile.
    await page.evaluate(id => {
      const text = '<img src=x onerror="window.controlEvidenceInjected=true">';
      document.getElementById(id).innerHTML = renderControlScheduleEvidence({
        control_schedule_context: text,
        control_schedule_statistics: {source: 'controlled', entries: [{lap: text,
          control: text, duration_laps: text, applied: text, suppressed: text, not_reached: text}]},
      });
    }, hostId);
    assert.equal(await host.locator('img').count(), 0);
    assert.equal(await page.evaluate(() => window.controlEvidenceInjected), undefined);
    // Global CSV histories appear once per trial, without multiplying by grid size.
    const download = page.waitForEvent('download');
    await page.evaluate(data => downloadControlScheduleCsv(data), fixture.payload);
    const file = await download;
    const text = fs.readFileSync(await file.path(), 'utf8');
    const lines = text.trim().split(/\r?\n/);
    assert.equal(lines.length, 1 + 2 * fixture.schedule.length);
    assert(lines[0].includes('history_status'));
    assert(lines.some(line => line.includes('"applied"')));
    const exportContext = await page.context().browser().newContext();
    const exported = await exportContext.newPage();
    try {
      const errors = [];
      exported.on('pageerror', error => errors.push(error.message));
      await exported.route('**/*', route => route.fulfill({contentType: 'text/javascript',
        body: 'window.Plotly={newPlot:()=>{}};'}));
      for (const html of [fixture.report, fixture.comparison]) {
        await exported.setContent(html);
        assert((await exported.locator('body').innerText()).includes('2 complete, 0 missing, 0 invalid'));
        assert(await exported.locator('.control-schedule-evidence').count());
        const region = exported.locator('.control-schedule-evidence [role=region]').first();
        await region.focus();
        assert(await region.evaluate(node => document.activeElement === node));
        for (const width of [390, 1440]) {
          await exported.setViewportSize({width, height: 900});
          assert(await exported.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
        }
      }
      assert.deepEqual(errors, []);
    } finally {
      await exportContext.close();
    }
    console.log('SC/VSC inputs, native execution evidence, global CSV and HTML checks passed.');
  } finally {
    await page.evaluate(({id, original}) => {
      document.getElementById(id)?.remove();
      document.getElementById('controlSourceSelect').value = original.source;
      document.getElementById('controlScheduleInput').value = original.text;
      updateControlScheduleControls();
    }, {id: hostId, original});
    await page.setViewportSize(viewport);
  }
}

module.exports = {checkControlSchedules};
