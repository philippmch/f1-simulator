// Common setup remains usable before expanding experimental controls.
const assert = require('node:assert/strict');
const path = require('node:path');
const { checkRaceTireRules } = require('./browser_race_tire_rules.cjs');

async function checkBasicSetup(page) {
  const viewport = page.viewportSize();
  const original = await page.evaluate(() => ({
    count: document.getElementById('simCount').value,
    seed: document.getElementById('seedInput').value,
    scenarios: getSelectedScenarioNames(),
    compare: document.getElementById('compareAutomaticInput').checked,
    grid: document.getElementById('raceGridModeSelect').value,
    pace: document.getElementById('postqualPacePolicySelect').value,
  }));
  try {
    assert.equal(await page.locator('#advancedSettings').getAttribute('open'), null);
    assert(await page.locator('#seedInput').isHidden());
    for (const id of ['trackSelect', 'weatherSelect', 'raceGridModeSelect', 'postqualPacePolicySelect', 'simCount', 'btnRun']) {
      assert(await page.locator('#' + id).isVisible(), `${id} is part of basic setup`);
    }
    const before = await page.evaluate(() => buildRunPayload());
    assert.equal(before.postqual_pace_policy, original.pace);
    await page.getByLabel('Race pace', {exact: true}).selectOption('season_points');
    assert.equal((await page.evaluate(() => buildRunPayload())).postqual_pace_policy, 'season_points');
    await page.getByLabel('Race pace', {exact: true}).selectOption(original.pace);
    assert(before && before.race && before.simulations >= 10);
    const stock = await page.evaluate(() => {
      const node = document.createElement('div');
      node.innerHTML = renderInventorySnapshot({simulation_inputs: {tire_inventory: {
        '<img src=x onerror="window.stockInjected=true">': [
          {compound: 'medium', age: 0}, {compound: 'wet', age: 9, remaining_laps: 1},
        ],
      }}});
      return {text: node.textContent, images: node.querySelectorAll('img').length};
    });
    assert.equal(stock.images, 0);
    assert(stock.text.includes('2 sets') && stock.text.includes('Wet (9 prior laps, 1 race lap left)'));
    assert(!stock.text.includes('{"compound"'), 'Input stock should be readable rather than raw JSON');
    await page.locator('#simCount').fill('10');
    assert((await page.locator('#runBudgetSummary').innerText()).includes('10 simulated races'));
    await page.evaluate(() => setScenarioSelection(['dry', 'light_rain'], {persist: false}));
    assert((await page.locator('#runBudgetSummary').innerText()).includes('20 simulated races'));
    await page.locator('#advancedSettings > summary').focus();
    await page.keyboard.press('Enter');
    assert(await page.locator('#seedInput').isVisible(), 'Keyboard opens the experimental controls');
    await checkRaceTireRules(page);
    await page.getByLabel('Race grid', {exact: true}).selectOption('simulated');
    assert((await page.locator('#raceGridHelp').innerText()).includes('Each race trial'));
    await page.getByLabel('Q1 condition', {exact: true}).selectOption('heavy_rain');
    await page.getByLabel('Q1 rain intensity (0–1)', {exact: true}).fill('0.6');
    const simulated = await page.evaluate(() => buildRunPayload());
    assert.equal(simulated.race_grid_mode, 'simulated');
    assert.equal(simulated.qualifying_weather.Q1.rain_intensity, 0.6);
    await page.getByLabel('Race grid', {exact: true}).selectOption('official');
    assert(await page.getByLabel('Q1 condition', {exact: true}).isDisabled());
    assert.equal(await page.evaluate(() => buildRunPayload()), null,
      'Choosing the official grid must reject existing hypothetical qualifying weather');
    assert.equal(await page.getByLabel('Race grid', {exact: true}).inputValue(), 'official',
      'Validation must preserve the selected grid source');
    await page.getByRole('button', {name: 'Clear custom qualifying weather', exact: true}).click();
    const official = await page.evaluate(() => buildRunPayload());
    assert.equal(official.race_grid_mode, 'official');
    assert.equal(Object.hasOwn(official, 'qualifying_weather'), false);
    await page.evaluate(() => {
      document.getElementById('raceGridModeSelect').value = 'simulated';
      loadUiPrefs();
      updateRaceGridControls();
    });
    assert.equal(await page.getByLabel('Race grid', {exact: true}).inputValue(), 'official',
      'Saved preferences must restore the official source rather than silently falling back');
    await page.getByLabel('Race grid', {exact: true}).selectOption(original.grid);
    assert(await page.getByLabel('Q1 condition', {exact: true}).isEnabled());
    await page.locator('#compareAutomaticInput').check();
    assert((await page.locator('#runBudgetSummary').innerText()).includes('40 simulated races'));
    await page.locator('#compareAutomaticInput').uncheck();
    await page.locator('#seedInput').fill('-1');
    await page.locator('#advancedSettings > summary').click();
    assert.equal(await page.locator('#advancedSettings').getAttribute('open'), null);
    assert.equal(await page.evaluate(() => buildRunPayload()), null);
    assert(await page.locator('#seedInput').isVisible(), 'Invalid hidden settings must be revealed');
    assert.equal(await page.evaluate(() => document.activeElement.id), 'seedInput');
    await page.locator('#seedInput').fill(original.seed);
    await page.evaluate(original => {
      document.getElementById('simCount').value = original.count;
      setScenarioSelection(original.scenarios, {persist: false});
      updateRunBudget();
      document.getElementById('advancedSettings').open = false;
      setStatusMessage('');
    }, original);
    assert.deepEqual(await page.evaluate(() => buildRunPayload()), before,
      'Collapsing controls must preserve the complete request');
    for (const width of [320, 390, 1440]) {
      await page.setViewportSize({width, height: 900});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      if (process.env.F1SIM_SCREENSHOTS) await page.screenshot({
        path: path.join(process.env.F1SIM_SCREENSHOTS, `basic-setup-${width}.png`), fullPage: false});
    }
    console.log('Basic setup, visible race budgets, keyboard disclosure and hidden-error recovery passed.');
  } finally {
    await page.evaluate(original => {
      document.getElementById('simCount').value = original.count;
      document.getElementById('seedInput').value = original.seed;
      document.getElementById('compareAutomaticInput').checked = original.compare;
      document.getElementById('raceGridModeSelect').value = original.grid;
      document.getElementById('postqualPacePolicySelect').value = original.pace;
      clearQualifyingWeather();
      saveUiPrefs();
      setScenarioSelection(original.scenarios, {persist: false});
      document.getElementById('advancedSettings').open = true;
      setStatusMessage('');
    }, original);
    await page.setViewportSize(viewport);
  }
}

module.exports = {checkBasicSetup};
