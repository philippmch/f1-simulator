const assert = require('node:assert/strict');

async function checkLearnedForecast(page) {
  const original = await page.evaluate(() => ({
    controls: Object.fromEntries([
      'forecastPolicySelect', 'raceGridModeSelect', 'postqualPacePolicySelect',
      'raceEngineSelect', 'rngPolicySelect', 'weatherModeSelect', 'simCount', 'weatherSelect',
    ].map(id => [id, document.getElementById(id).value])),
    scenarios: getSelectedScenarioNames(),
  }));
  try {
    await page.getByLabel('Forecast', {exact: true}).selectOption('learned_afterqual_v1');
    const payload = await page.evaluate(() => buildRunPayload());
    assert.equal(payload.forecast_policy, 'learned_afterqual_v1');
    assert.equal(payload.race_grid_mode, 'official');
    assert.equal(payload.postqual_pace_policy, 'qualifying_led');
    assert.equal(payload.weather_mode, 'fixed_rainfall');
    assert.equal(payload.scenarios, 'dry');
    assert(payload.simulations >= 400);

    const rendered = await page.evaluate(() => {
      const id = '<img src=x onerror="window.forecastInjected=true">';
      const node = document.createElement('div');
      node.innerHTML = renderLearnedForecast({forecast_estimate: {
        win_probabilities: {[id]: .6}, podium_probabilities: {[id]: .8},
        expected_positions: {[id]: 2.5}, nonfinish_probabilities: {[id]: .1},
      }});
      return {text: node.textContent, images: node.querySelectorAll('img').length};
    });
    assert.equal(rendered.images, 0);
    for (const text of ['Experimental confirmed-grid forecast', '60.0%', '80.0%', '2.50', '10.0%']) {
      assert(rendered.text.includes(text), text);
    }
    assert(rendered.text.includes('raw simulation'));
    await page.getByLabel('Forecast', {exact: true}).selectOption('native');
    assert.equal(Object.hasOwn(await page.evaluate(() => buildRunPayload()), 'forecast_policy'), false);
  } finally {
    await page.evaluate(original => {
      for (const [id, value] of Object.entries(original.controls)) document.getElementById(id).value = value;
      setScenarioSelection(original.scenarios, {persist: false});
      updateRaceGridControls();
      updateRunBudget();
      saveUiPrefs();
    }, original);
  }
  console.log('Learned forecast setup, separate fractional estimates and escaping passed.');
}

module.exports = {checkLearnedForecast};
