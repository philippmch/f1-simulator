// Common setup remains usable before expanding experimental controls.
const assert = require('node:assert/strict');
const path = require('node:path');

async function checkBasicSetup(page) {
  const viewport = page.viewportSize();
  const original = await page.evaluate(() => ({
    count: document.getElementById('simCount').value,
    seed: document.getElementById('seedInput').value,
    scenarios: getSelectedScenarioNames(),
    compare: document.getElementById('compareAutomaticInput').checked,
  }));
  try {
    assert.equal(await page.locator('#advancedSettings').getAttribute('open'), null);
    assert(await page.locator('#seedInput').isHidden());
    for (const id of ['trackSelect', 'weatherSelect', 'simCount', 'btnRun']) {
      assert(await page.locator('#' + id).isVisible(), `${id} is part of basic setup`);
    }
    const before = await page.evaluate(() => buildRunPayload());
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
      setScenarioSelection(original.scenarios, {persist: false});
      document.getElementById('advancedSettings').open = true;
      setStatusMessage('');
    }, original);
    await page.setViewportSize(viewport);
  }
}

module.exports = {checkBasicSetup};
