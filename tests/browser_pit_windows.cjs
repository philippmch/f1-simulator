// The native Python cases cover execution; these check input and visible saved evidence.
const assert = require('node:assert/strict');
const path = require('node:path');

async function checkPitWindows(page) {
  const input = page.locator('#pitPlansInput');
  const original = await input.inputValue();
  const viewport = page.viewportSize();
  try {
    const parsed = await page.evaluate(() => {
      const valid = ['3-6@sc:hard,8:soft', '3-6@vsc:hard', '3-6@neutralized:hard'];
      const invalid = ['6-3@sc:hard', '3-3@sc:hard', '1-6@sc:hard', '3-6@green:hard',
        '3-6:hard', '3@sc:hard', '3-6@sc:hard,6:soft', '3-6@sc:hard,5-8@vsc:soft',
        '3.0-6@sc:hard', '3-9007199254740992@sc:hard'];
      return {valid: valid.map(text => parsePitPlanInstructions(text)),
        invalid: invalid.map(text => parsePitPlanInstructions(text).ok)};
    });
    assert(parsed.valid.every(value => value.ok));
    assert(parsed.invalid.every(value => value === false));
    const plan = [{lap: 6, compound: 'hard', earliest_lap: 3, trigger: 'safety_car'}];
    assert.deepEqual(parsed.valid[0].instructions, [...plan, {lap: 8, compound: 'soft'}]);
    assert.equal(parsed.valid[1].instructions[0].trigger, 'vsc');
    assert.equal(parsed.valid[2].instructions[0].trigger, 'neutralized');
    assert(await page.evaluate(() => {
      const instruction = {lap: 6, earliest_lap: 3, trigger: 'safety_car',
        executed: 2, overridden: 0};
      return [[], [{lap: 4, count: 1}], [{lap: 4, count: 2}, {lap: 4, count: 1}],
        [{lap: 7, count: 2}], [{lap: 4, count: 3}]].every(service_laps =>
        pitPlanServiceLapsText({...instruction, service_laps}) === 'Not recorded');
    }), 'Malformed service frequencies must remain unknown');
    await input.fill('S00=3-6@sc:hard;S01=none');
    const built = await page.evaluate(() => buildRunPayload());
    assert(built, 'Valid windows must build a run request');
    assert.deepEqual(built.pit_plans, {S00: plan, S01: []});
    assert.equal(await page.evaluate(value => frozenPitInstructionsText(value), plan), '3-6@sc:hard');
    await page.evaluate(value => {
      const host = document.createElement('div');
      host.id = 'pit-window-browser-evidence';
      const outcome = {...value[0], status: 'executed', reason: 'user_plan', actual_lap: 4,
        actual_compound: 'hard', actual_set_id: '<img onerror="window.windowEvidenceExecuted=true">'};
      const scenario = {simulation_inputs: {pit_plans: {S00: value}}};
      const statistics = {status: 'available', recorded_trials: 2, drivers: [{
        driver_id: 'S00', no_elective_stops: false, valid_histories: 2, missing_histories: 0,
        invalid_histories: 0, instructions: [{...value[0], executed: 2, overridden: 0,
          skipped: 0, not_reached: 0, service_laps: [{lap: 4, count: 1}, {lap: 6, count: 1}]}],
      }]};
      host.innerHTML = renderPitPlanSnapshot(scenario)
        + renderPitPlanHistory([{driver_id: 'S00', pit_plan_history: [outcome]}])
        + renderPitPlanStatistics(statistics, scenario);
      document.body.append(host);
    }, plan);
    const evidence = page.locator('#pit-window-browser-evidence');
    for (const summary of await evidence.locator('summary').all()) await summary.click();
    assert((await evidence.innerText()).includes('3–6 (SC; deadline 6)'));
    assert((await evidence.innerText()).includes('Hard · service L4'));
    assert((await evidence.innerText()).includes('Service laps: L4 × 1, L6 × 1'));
    assert.equal(await evidence.locator('img').count(), 0);
    assert.equal(await page.evaluate(() => window.windowEvidenceExecuted), undefined);
    if (process.env.F1SIM_SCREENSHOTS) {
      for (const width of [390, 1440]) {
        await page.setViewportSize({width, height: 900});
        await evidence.screenshot({path: path.join(process.env.F1SIM_SCREENSHOTS,
          `pit-window-evidence-${width}.png`), animations: 'disabled'});
      }
    }
  } finally {
    if (viewport) await page.setViewportSize(viewport);
    await input.fill(original);
    await page.evaluate(() => document.getElementById('pit-window-browser-evidence')?.remove());
  }
  console.log('Pit-window input, frozen plans, service laps and escaped evidence passed.');
}

module.exports = {checkPitWindows};
