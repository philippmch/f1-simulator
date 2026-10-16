// Optional race-wide tyre rules: validate before request construction and read saved results.
const assert = require('node:assert/strict');

async function checkRaceTireRules(page) {
  const original = await page.evaluate(() => ({
    max: document.getElementById('maxSetLapsInput').value,
    minimum: document.getElementById('minimumSetsInput').value,
    open: document.getElementById('advancedSettings').open,
  }));
  try {
    await page.evaluate(() => { document.getElementById('advancedSettings').open = true; });
    const cap = page.getByLabel('Max tyre set race laps (optional)', {exact: true});
    const minimum = page.getByLabel('Minimum distinct tyre sets (optional)', {exact: true});
    await cap.fill('');
    await minimum.fill('');
    const baseline = await page.evaluate(() => buildRunPayload());
    assert(baseline);
    assert.equal(Object.hasOwn(baseline, 'race_tire_rules'), false);
    await minimum.fill('1');
    assert.equal(Object.hasOwn(await page.evaluate(() => buildRunPayload()), 'race_tire_rules'), false,
      'The ordinary one-set default must not enable a new simulation policy');
    await cap.fill('25');
    assert.deepEqual((await page.evaluate(() => buildRunPayload())).race_tire_rules, {max_set_laps: 25});
    await minimum.fill('3');
    const enabled = await page.evaluate(() => buildRunPayload());
    assert.deepEqual(enabled.race_tire_rules, {max_set_laps: 25, minimum_sets: 3});
    assert.equal(enabled.race_grid_mode, baseline.race_grid_mode,
      'Tyre constraints must preserve the qualifying/grid choice');
    assert((await page.locator('#advancedSettingsHint').innerText()).includes('Custom race settings'));
    await cap.fill('');
    assert.deepEqual((await page.evaluate(() => buildRunPayload())).race_tire_rules, {minimum_sets: 3});
    for (const [id, value] of [
      ['maxSetLapsInput', '0'], ['minimumSetsInput', '-1'],
      ['maxSetLapsInput', '1.5'], ['minimumSetsInput', '9007199254740992'],
    ]) {
      await cap.fill('');
      await minimum.fill('');
      await page.locator('#' + id).fill(value);
      await page.evaluate(() => { document.getElementById('advancedSettings').open = false; });
      assert.equal(await page.evaluate(() => buildRunPayload()), null,
        `${id}=${value} must be rejected before sending a run`);
      assert.equal(await page.evaluate(() => document.activeElement.id), id);
      assert(await page.locator('#' + id).isVisible(), 'Invalid advanced input must be revealed');
      assert((await page.locator('#appStatus').innerText()).includes('positive whole numbers'));
    }
    await cap.fill('99');
    await minimum.fill('9');
    const snapshot = await page.evaluate(() => {
      const previous = simResults;
      try {
        simResults = {request: {race_tire_rules: {max_set_laps: 25, minimum_sets: 3}}};
        const requestText = renderRaceTireRulesSnapshot({});
        const savedText = renderRaceTireRulesSnapshot({simulation_inputs: {
          race_tire_rules: {max_set_laps: 18},
        }});
        const empty = renderRaceTireRulesSnapshot({simulation_inputs: {race_tire_rules: {}}});
        simResults = {request: {}};
        const unlimitedStock = [15, 16].map(schema_version => renderInventorySnapshot({
          simulation_inputs: {schema_version},
        }));
        const unknownStock = renderInventorySnapshot({simulation_inputs: {schema_version: 17}});
        const node = document.createElement('div');
        node.innerHTML = renderRaceTireRulesSnapshot({simulation_inputs: {
          race_tire_rules: {max_set_laps: '<img src=x onerror=alert(1)>'},
        }});
        return {requestText, savedText, empty, unlimitedStock, unknownStock, invalid: node.textContent,
          images: node.querySelectorAll('img').length};
      } finally { simResults = previous; }
    });
    assert(snapshot.requestText.includes('25 completed race laps per set'));
    assert(snapshot.requestText.includes('3 distinct sets run'));
    assert(!snapshot.requestText.includes('99 completed'), 'Saved results must ignore current controls');
    assert(snapshot.savedText.includes('18 completed') && !snapshot.savedText.includes('25 completed'),
      'The recorded scenario takes priority over a request fallback');
    assert.equal(snapshot.empty, '', 'Ordinary saved runs should gain no extra disclosure');
    assert(snapshot.unlimitedStock.every(text => text.includes('unlimited sets')),
      'Known schema 15/16 without an inventory records the unlimited-stock policy');
    assert(snapshot.unknownStock.includes('input not recorded'),
      'An unknown future schema must not imply unlimited stock');
    assert.equal(snapshot.images, 0);
    assert(snapshot.invalid.includes('invalid saved rules'));
    await cap.fill('');
    await minimum.fill('');
    assert.deepEqual(await page.evaluate(() => buildRunPayload()), baseline,
      'Clearing both optional rules must restore the complete original request');
    console.log('Race tyre rules: optional payloads, validation, grid preservation and saved snapshots passed.');
  } finally {
    await page.evaluate(original => {
      document.getElementById('maxSetLapsInput').value = original.max;
      document.getElementById('minimumSetsInput').value = original.minimum;
      document.getElementById('advancedSettings').open = original.open;
      updateRunBudget();
      setStatusMessage('');
    }, original);
  }
}

module.exports = {checkRaceTireRules};
