// Run against a local server. Requires Playwright and a Chromium browser.
// F1SIM_OFFLINE=1 generates synthetic inputs with PYTHON (default: python),
// intercepts all requests, and needs neither a server nor network access.
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const offline = process.env.F1SIM_OFFLINE === '1';
  const fixture = offline ? JSON.parse(execFileSync(process.env.PYTHON || 'python',
    [path.join(__dirname, 'browser_fixture.py')], {encoding: 'utf8', maxBuffer: 8 * 1024 * 1024,
      timeout: 120000})) : null;
  const browser = await chromium.launch({
    headless: true,
    channel: process.env.BROWSER_CHANNEL || undefined,
  });
  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(120000);
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    const unexpectedRequests = [];
    let holdNextRun = false;
    let delayedRunRoute = null;
    if (offline) {
      await page.route('**/*', route => {
        const url = new URL(route.request().url());
        if (url.hostname !== 'f1sim.test') {
          // The dashboard uses external fonts; system fallbacks keep this test offline.
          if (!['fonts.googleapis.com', 'fonts.gstatic.com'].includes(url.hostname)) {
            unexpectedRequests.push(url.href);
          }
          return route.abort();
        }
        const body = url.pathname === '/api/run' ? fixture.payload
          : url.pathname === '/api/calendar' ? fixture.calendar
          : url.pathname === '/api/health' ? {status: 'ok', season: 2026} : null;
        if (url.pathname === '/api/run' && holdNextRun) {
          holdNextRun = false;
          delayedRunRoute = route;
          return;
        }
        if (body) return route.fulfill({json: body});
        if (url.pathname === '/') return route.fulfill({
          contentType: 'text/html', body: fixture.html,
        });
        if (url.pathname !== '/favicon.ico') unexpectedRequests.push(url.href);
        return route.fulfill({status: 404, body: ''});
      });
    }
    await page.goto(offline ? 'http://f1sim.test' :
      process.env.F1SIM_URL || 'http://127.0.0.1:8080');
    await page.waitForFunction(() => !connectionRefreshInProgress);
    assert(await page.locator('#btnRun').isEnabled(), 'Live calendar must be available');
    assert(await page.locator('#btnTyreSetup').isEnabled(), 'Tyre setup editor must be available');

    // The editor is a draft over the existing shorthand fields. Exercise the
    // full finite-pool path, including duplicate physical sets and the 20-set
    // limit, before checking that Apply emits the same API shape as shorthand.
    await page.locator('#btnTyreSetup').click();
    await page.locator('#tyreSetupDialog').waitFor({state: 'visible'});
    assert.equal(await page.locator('#tyreSetupDialog [data-tyre-driver]').count(), 0);
    await page.locator('#tyreEditorAddDriver').click();
    await page.locator('[data-driver-index="0"] [data-tyre-driver-code]').fill('S00');
    await page.locator('[data-driver-index="0"] [data-tyre-opening-compound]').selectOption('hard');
    await page.locator('[data-driver-index="0"] [data-tyre-opening-age]').fill('5');
    await page.locator('[data-driver-index="0"] [data-tyre-pool-mode]').check();
    await page.locator('[data-driver-index="0"] [data-tyre-add-set]').click();
    await page.locator('[data-driver-index="0"] [data-set-index="0"] [data-tyre-set-compound]').selectOption('hard');
    await page.locator('[data-driver-index="0"] [data-set-index="0"] [data-tyre-set-age]').fill('5');
    await page.locator('[data-driver-index="0"] [data-tyre-add-set]').click();
    await page.locator('[data-driver-index="0"] [data-set-index="1"] [data-tyre-set-compound]').selectOption('hard');
    await page.locator('[data-driver-index="0"] [data-set-index="1"] [data-tyre-set-age]').fill('5');
    const duplicateDraft = await page.evaluate(() => serializeTyreEditorDraft(tyreEditorDriversFromDom()));
    assert.equal(duplicateDraft.ok, true);
    assert.equal(duplicateDraft.inventoryRaw, 'S00=hard@5,hard@5');
    for (const [index, compound] of [[2, 'soft'], [3, 'intermediate'], [4, 'wet']]) {
      await page.locator('[data-driver-index="0"] [data-tyre-add-set]').click();
      await page.locator(`[data-driver-index="0"] [data-set-index="${index}"] [data-tyre-set-compound]`).selectOption(compound);
    }
    assert.equal(await page.locator('[data-driver-index="0"] [data-tyre-set-row]').count(), 5);
    await page.locator('[data-driver-index="0"] [data-set-index="1"] [data-tyre-remove-set]').click();
    for (let count = 4; count < 20; count += 1) {
      await page.locator('[data-driver-index="0"] [data-tyre-add-set]').click();
    }
    assert.equal(await page.locator('[data-driver-index="0"] [data-tyre-set-row]').count(), 20);
    assert(await page.locator('[data-driver-index="0"] [data-tyre-add-set]').isDisabled());
    for (let count = 20; count > 4; count -= 1) {
      await page.locator('[data-driver-index="0"] [data-tyre-set-row]').last()
        .locator('[data-tyre-remove-set]').click();
    }
    assert.equal(await page.locator('[data-driver-index="0"] [data-tyre-set-row]').count(), 4);
    await page.locator('#tyreEditorAddDriver').click();
    await page.locator('[data-driver-index="1"] [data-tyre-driver-code]').fill('S01');
    await page.locator('[data-driver-index="1"] [data-tyre-opening-compound]').selectOption('soft');
    const finiteToggle = page.locator('[data-driver-index="0"] [data-tyre-pool-mode]');
    const finiteSetList = page.locator('[data-driver-index="0"] [data-tyre-set-list]');
    await finiteToggle.uncheck();
    assert(await finiteSetList.isHidden(), 'Unlimited pools must hide physical set rows');
    assert.equal(await page.evaluate(() => tyreEditorFocusables()
      .some(node => node.id === 'tyreSetCompound-0-0')), false,
    'Hidden physical sets must leave keyboard navigation');
    await finiteToggle.check();
    assert.equal(await page.locator('[data-driver-index="0"] [data-tyre-set-row]').count(), 4,
      'Switching back to finite mode must restore the draft set rows');
    for (const width of [390, 1440]) {
      await page.setViewportSize({width, height: 900});
      await page.locator('#tyreEditorContent').evaluate(node => { node.scrollTop = 0; });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
        `Tyre setup editor overflows at ${width}px`);
      if (process.env.F1SIM_SCREENSHOTS) {
        await page.screenshot({path: path.join(process.env.F1SIM_SCREENSHOTS,
          `tyre-setup-editor-${width}.png`), fullPage: false});
        await page.locator('#tyreEditorContent').evaluate(node => { node.scrollTop = node.scrollHeight; });
        await page.screenshot({path: path.join(process.env.F1SIM_SCREENSHOTS,
          `tyre-setup-editor-${width}-bottom.png`), fullPage: false});
        await page.locator('#tyreEditorContent').evaluate(node => { node.scrollTop = 0; });
      }
    }
    for (let count = 4; count < 20; count += 1) {
      await page.locator('[data-driver-index="0"] [data-tyre-add-set]').click();
    }
    await page.setViewportSize({width: 390, height: 900});
    await page.locator('#tyreEditorContent').evaluate(node => { node.scrollTop = 0; });
    const firstDriverCode = page.locator('[data-driver-index="0"] [data-tyre-driver-code]');
    const lastVisibleSetAge = page.locator('#tyreSetAge-0-19');
    await firstDriverCode.focus();
    for (let tab = 0; tab < 100; tab += 1) {
      if (await lastVisibleSetAge.evaluate(node => document.activeElement === node)) break;
      await page.keyboard.press('Tab');
    }
    assert(await lastVisibleSetAge.evaluate(node => document.activeElement === node),
      'Tab navigation must reach the last finite-pool age field');
    const editorScrollState = await page.evaluate(() => {
      const content = document.getElementById('tyreEditorContent');
      const field = document.getElementById('tyreSetAge-0-19');
      const contentBox = content.getBoundingClientRect();
      const fieldBox = field.getBoundingClientRect();
      return {
        scrollTop: content.scrollTop,
        visible: fieldBox.top >= contentBox.top && fieldBox.bottom <= contentBox.bottom,
      };
    });
    assert(editorScrollState.scrollTop > 0 && editorScrollState.visible,
      'Tab navigation must scroll the focused finite-pool field into view');
    for (let count = 20; count > 4; count -= 1) {
      await page.locator('[data-driver-index="0"] [data-tyre-set-row]').last()
        .locator('[data-tyre-remove-set]').click();
    }
    await page.setViewportSize({width: 1440, height: 900});
    await page.locator('#tyreEditorContent').evaluate(node => { node.scrollTop = 0; });
    await page.locator('[data-driver-index="1"] [data-tyre-driver-code]').fill('__proto__');
    const prototypeDraft = await page.evaluate(() => serializeTyreEditorDraft(tyreEditorDriversFromDom()));
    assert(prototypeDraft.ok && prototypeDraft.startingRaw.includes('__proto__=soft'));
    await page.locator('[data-driver-index="1"] [data-tyre-driver-code]').fill('<img src=x>');
    assert.equal(await page.locator('#tyreEditorContent img').count(), 0);
    await page.locator('[data-driver-index="1"] [data-tyre-driver-code]').fill('S01');
    await page.locator('#tyreEditorApply').click();
    await page.locator('#tyreSetupDialog').waitFor({state: 'hidden'});
    assert.equal(await page.locator('#startingTiresInput').inputValue(), 'S00=hard@5, S01=soft');
    assert.equal(await page.locator('#tireInventoryInput').inputValue(), 'S00=hard@5,soft,intermediate,wet');
    assert.deepEqual(await page.evaluate(() => buildRunPayload().starting_tire_ages), {S00: 5});
    assert.equal(await page.evaluate(() => buildRunPayload().tire_inventory.S00.length), 4);

    // A failed Apply leaves the serialized inputs untouched, while Escape
    // cancels the draft and restores the trigger focus.
    const editorRawBeforeInvalid = await page.evaluate(() => ({
      starting: document.getElementById('startingTiresInput').value,
      inventory: document.getElementById('tireInventoryInput').value,
    }));
    await page.locator('#btnTyreSetup').click();
    await page.locator('[data-driver-index="1"] [data-tyre-driver-code]').fill('S00');
    await page.locator('#tyreEditorApply').click();
    assert((await page.locator('#tyreEditorMessage').innerText()).includes('listed more than once'));
    await page.locator('[data-driver-index="1"] [data-tyre-driver-code]').fill('S01');
    await page.locator('[data-driver-index="0"] [data-tyre-opening-compound]').selectOption('medium');
    await page.locator('#tyreEditorApply').click();
    assert(await page.locator('#tyreSetupDialog').isVisible());
    assert((await page.locator('#tyreEditorMessage').innerText()).includes('must match one physical set'));
    assert.deepEqual(await page.evaluate(() => ({
      starting: document.getElementById('startingTiresInput').value,
      inventory: document.getElementById('tireInventoryInput').value,
    })), editorRawBeforeInvalid);
    for (const invalidAge of ['-1', '1.5', '1001']) {
      await page.locator('[data-driver-index="0"] [data-tyre-opening-age]').fill(invalidAge);
      await page.locator('#tyreEditorApply').click();
      assert((await page.locator('#tyreEditorMessage').innerText()).includes('whole number from 0 to 1000'));
    }
    await page.keyboard.press('Escape');
    await page.locator('#tyreSetupDialog').waitFor({state: 'hidden'});
    assert.equal(await page.evaluate(() => document.activeElement?.id), 'btnTyreSetup');

    // Direct shorthand edits are imported on the next open; malformed raw
    // input is reported on the original field and never gets erased.
    await page.locator('#startingTiresInput').fill('S00=soft@4');
    await page.locator('#tireInventoryInput').fill('S00=soft@4,hard');
    await page.locator('#btnTyreSetup').click();
    assert.equal(await page.locator('[data-driver-index="0"] [data-tyre-opening-compound]').inputValue(), 'soft');
    assert.equal(await page.locator('[data-driver-index="0"] [data-tyre-opening-age]').inputValue(), '4');
    await page.locator('#tyreEditorCancel').click();
    assert.equal(await page.locator('#startingTiresInput').inputValue(), 'S00=soft@4');
    await page.locator('#startingTiresInput').fill('S00=soft,S00=hard');
    await page.locator('#btnTyreSetup').click();
    assert.equal(await page.locator('#tyreSetupDialog').isVisible(), false);
    assert((await page.locator('#appStatus').innerText()).includes('use each driver once'));
    assert.equal(await page.evaluate(() => document.activeElement?.id), 'startingTiresInput');
    await page.locator('#startingTiresInput').fill(offline ? 'S00=hard@5, S01=soft' : '');
    await page.locator('#tireInventoryInput').fill('');

    await page.locator('#simCount').fill('10');
    assert.equal(await page.locator('#raceEngineSelect').inputValue(), 'standard');
    await page.locator('#raceEngineSelect').selectOption('chronological');
    await page.locator('#parallelSelect').selectOption('false');
    assert.equal(await page.locator('#weatherModeSelect').inputValue(), 'evolving');
    await page.locator('#weatherModeSelect').selectOption('fixed_rainfall');
    await page.locator('#startingTiresInput').fill('S00=soft,S00=hard');
    assert.equal(await page.evaluate(() => buildRunPayload()), null);
    assert((await page.locator('#appStatus').innerText()).includes('use each driver once'));
    for (const invalid of ['S00=hard@-1', 'S00=soft@1.5', 'S00=soft@1001', 'S00=soft@']) {
      await page.locator('#startingTiresInput').fill(invalid);
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
    }
    await page.locator('#startingTiresInput').fill(offline ? 'S00=hard@5, S01=soft' : '');
    await page.locator('#tireInventoryInput').fill('S00=hard@1.5');
    assert.equal(await page.evaluate(() => buildRunPayload()), null);
    await page.locator('#tireInventoryInput').fill('S00=hard@5,soft,intermediate,wet');
    assert.deepEqual(await page.evaluate(() => buildRunPayload().tire_inventory.S00), [
      {id: 'set-1', compound: 'hard', age: 5}, {id: 'set-2', compound: 'soft', age: 0},
      {id: 'set-3', compound: 'intermediate', age: 0}, {id: 'set-4', compound: 'wet', age: 0},
    ]);
    await page.locator('#tireInventoryInput').fill(offline ? 'S00=hard@5,soft,intermediate,wet' : '');
    const pitPlanPrototype = await page.evaluate(() => {
      const input = document.getElementById('pitPlansInput');
      const previous = input.value;
      input.value = '__proto__=24:wet';
      const parsed = parsePitPlanInput();
      input.value = previous;
      return {
        ok: parsed.ok,
        prototype: Object.getPrototypeOf(parsed.pitPlans),
        ownPrototypeKey: Object.hasOwn(parsed.pitPlans, '__proto__'),
        record: parsed.pitPlans.__proto__?.[0],
      };
    });
    assert.equal(pitPlanPrototype.ok, true);
    assert.equal(pitPlanPrototype.prototype, null);
    assert.equal(pitPlanPrototype.ownPrototypeKey, true);
    assert.deepEqual(pitPlanPrototype.record, {lap: 24, compound: 'wet'});
    const pitPlanInput = page.locator('#pitPlansInput');
    await pitPlanInput.fill(offline ? 'S00=18:hard,36:soft;S01=none' : '');
    const customPayload = await page.evaluate(() => {
      const payload = buildRunPayload();
      return {
        payload,
        pitPlansNullPrototype: Object.getPrototypeOf(payload?.pit_plans) === null,
      };
    });
    if (offline) {
      assert.deepEqual(customPayload.payload.pit_plans, {
        S00: [{lap: 18, compound: 'hard'}, {lap: 36, compound: 'soft'}], S01: [],
      });
      assert.equal(customPayload.pitPlansNullPrototype, true);
    } else {
      assert.equal(Object.hasOwn(customPayload.payload, 'pit_plans'), false);
    }
    for (const invalid of [
      'S00=1:hard', 'S00=18.5:hard', 'S00=true:hard', 'S00=18:Hard',
      'S00=18:hard,18:soft', 'S00=18:hard;S00=24:soft', 'S00=18:',
      'S00=18:hard,', 'S00=', 'S00=none,24:hard',
      `S00=${Array.from({length: 21}, (_, index) => `${index + 2}:hard`).join(',')}`,
    ]) {
      await pitPlanInput.fill(invalid);
      assert.equal(await page.evaluate(() => buildRunPayload()), null,
        `Invalid custom pit plan was accepted: ${invalid}`);
      assert((await page.locator('#appStatus').innerText()).includes('Custom pit plans'));
    }
    await pitPlanInput.fill(offline ? 'S00=18:hard,36:soft;S01=none' : '');
    const ledgerHtml = await page.evaluate(() => renderTireSetLedgers([{
      driver_id: 'S00', tire_set_history: [{lap: 1, kind: 'start',
        set_id: '<img src=x onerror=alert(1)>', compound: 'hard', age_at_fit: 5,
        age_at_end: 8, laps_used: 3}], tire_inventory: [{id: 'set-1',
        compound: 'hard', age: 8, current: true, available: false, unavailable: false}],
    }]));
    assert(ledgerHtml.includes('&lt;img') && !ledgerHtml.includes('<img'));
    assert(ledgerHtml.includes('Final race set pool') && ledgerHtml.includes('Physical set fittings'));
    await page.evaluate(() => setScenarioSelection(['dry', 'light_rain', 'heavy_rain']));
    const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/run'));
    await page.locator('#btnRun').click();
    const response = await responsePromise;
    assert.equal(response.status(), 200);
    assert.equal(response.request().postDataJSON().race_engine, 'chronological');
    assert.equal(response.request().postDataJSON().weather_mode, 'fixed_rainfall');
    if (offline) {
      await page.locator('#sampleTireSetLedgers summary').click();
      await page.locator('#sampleTireSetLedgers table').first().waitFor({state: 'visible'});
      const setText = await page.locator('#sampleTireSetLedgers').innerText();
      assert(setText.includes('set-1') && setText.includes('Physical set fittings'));
      assert(setText.includes('Final race set pool'));
      assert.deepEqual(response.request().postDataJSON().tire_inventory, fixture.payload.request.tire_inventory);
    }
    assert.deepEqual(response.request().postDataJSON().starting_tires,
      offline ? {S00: 'hard', S01: 'soft'} : {});
    assert.deepEqual(response.request().postDataJSON().starting_tire_ages,
      offline ? {S00: 5} : {});
    if (offline) {
      assert.deepEqual(response.request().postDataJSON().pit_plans, {
        S00: [{lap: 18, compound: 'hard'}, {lap: 36, compound: 'soft'}], S01: [],
      });
    } else {
      assert.equal(Object.hasOwn(response.request().postDataJSON(), 'pit_plans'), false);
    }
    const payload = await response.json();
    assert.equal(payload.request.race_engine, 'chronological');
    // The matrix initially shows aggregate top contenders, which need not
    // include the winner of the representative individual race.
    const driverId = Object.values(payload.scenarios)[0].win_probabilities[0][0];
    await page.waitForFunction(() => !runInProgress);
    assert((await page.locator('#panel-race').textContent()).includes('Lap-aware model (experimental)'));
    assert((await page.locator('#panel-race').textContent()).includes('Fixed rainfall; surface wetness still evolves'));
    const representativeScenario = Object.values(payload.scenarios)[0];
    const sampleSuspension = representativeScenario.sample_race_suspension_seconds;
    assert((await page.locator('#panel-race').textContent()).includes(
      `Completed race suspension: ${typeof sampleSuspension === 'number'
        ? `${sampleSuspension.toFixed(3)} s` : 'Not recorded'}`));
    assert((await page.locator('#panel-race').textContent()).includes(
      'Race-wide collection + restart pause'));
    assert.equal(await page.evaluate(() => sampleRaceSuspensionSeconds({
      sample_race: [{race_suspension_seconds: 12}, {}],
    })), null);
    assert.equal(await page.evaluate(() => sampleRaceSuspensionSeconds({
      sample_race_suspension_seconds: null,
      sample_race: [{race_suspension_seconds: 12}],
    })), null);
    await page.locator('#weatherModeSelect').selectOption('evolving');
    await page.evaluate(() => renderRace());
    assert((await page.locator('#panel-race').textContent()).includes('Fixed rainfall; surface wetness still evolves'));
    await page.locator('#weatherModeSelect').selectOption('fixed_rainfall');
    if (offline) {
      await page.locator('#samplePitStopDetails summary').focus();
      await page.keyboard.press('Enter');
      const stopRows = await page.evaluate(() => getScenarioEntry().data.sample_race
        .reduce((count, driver) => count + Math.max(1, driver.pit_stop_details.length), 0));
      assert.equal(await page.locator('#samplePitStopDetails tbody tr').count(), stopRows);
      await page.evaluate(() => {
        window.savedPitSample = getScenarioEntry().data.sample_race;
        getScenarioEntry().data.sample_race = savedPitSample.slice(0, 3).map((row, i) => ({
          ...row, pit_stop_details: i === 0 ? null : i === 1 ? [] : [{lap: 4,
            from_compound: '<img src=x onerror="window.pitInjected=true">', to_compound: 'hard',
            tire_age: 3, condition: 'dry', rain_intensity: 0, track_wetness: 0,
            control: 'green', lane_loss: 22, service_time: 3, queue_time: 2, total_loss: 27,
            decision_reason: 'dry_forecast', forecast_saving_seconds: -0.05}],
        }));
        renderRace();
      });
      await page.locator('#samplePitStopDetails summary').click();
      const pitText = await page.locator('#samplePitStopDetails').innerText();
      for (const label of ['Pit-stop details not recorded', 'No paid stops', '27.000 s',
        'Lane 22.000 s', 'Service 3.000 s', 'Queue 2.000 s', '3 completed laps',
        'Dry strategy forecast', 'Forecast advantage: -0.050 s']) {
        assert(pitText.includes(label), `Missing paid-stop detail: ${label}`);
      }
      assert.equal(await page.locator('#samplePitStopDetails img').count(), 0);
      assert.equal(await page.evaluate(() => Boolean(window.pitInjected)), false);
      const legacyDecisions = await page.evaluate(() => renderPitStopDetails([{driver_id: 'Legacy',
        pit_stop_details: [{decision_reason: '<img src=x onerror="window.pitInjected=true">',
          forecast_saving_seconds: Infinity}]}]));
      assert(legacyDecisions.includes('Forecast advantage: Not recorded'));
      assert(!legacyDecisions.includes('<img'));
      await page.evaluate(() => {
        getScenarioEntry().data.sample_race = savedPitSample;
        delete window.savedPitSample;
        renderRace();
      });
      const weatherSummary = page.locator('#sampleWeatherHistory summary');
      await weatherSummary.focus();
      await page.keyboard.press('Enter');
      const observedWeather = await page.evaluate(() => getScenarioEntry().data.sample_weather_history);
      assert(observedWeather.length > 0);
      assert.equal(await page.locator('#sampleWeatherHistory tbody tr').count(), observedWeather.length);
      assert((await page.locator('#sampleWeatherHistory').innerText()).includes('Shared leading-interval updates'));
      await page.evaluate(() => {
        window.savedWeatherTrace = getScenarioEntry().data.sample_weather_history;
        getScenarioEntry().data.sample_weather_history = [{lap: 2,
          condition: '<img src=x onerror="window.weatherInjected=true">',
          rain_intensity: .35, track_wetness: null}];
        renderRace();
      });
      await page.locator('#sampleWeatherHistory summary').click();
      assert((await page.locator('#sampleWeatherHistory').innerText()).includes('35.0%'));
      assert((await page.locator('#sampleWeatherHistory').innerText()).includes('Not recorded'));
      assert.equal(await page.locator('#sampleWeatherHistory img').count(), 0);
      assert.equal(await page.evaluate(() => Boolean(window.weatherInjected)), false);
      await page.evaluate(() => {
        delete getScenarioEntry().data.sample_weather_history;
        renderRace();
      });
      await page.locator('#sampleWeatherHistory summary').click();
      assert((await page.locator('#sampleWeatherHistory').innerText()).includes('not recorded for this trial'));
      await page.evaluate(() => {
        getScenarioEntry().data.sample_weather_history = savedWeatherTrace;
        delete window.savedWeatherTrace;
        renderRace();
      });
      assert((await page.locator('#panel-race').textContent()).includes('S00=hard@5, S01=soft'));
      for (const scenario of Object.values(payload.scenarios)) {
        assert.deepEqual(scenario.simulation_inputs.starting_tires, {S00: 'hard', S01: 'soft'});
        assert.deepEqual(scenario.simulation_inputs.starting_tire_ages, {S00: 5});
      }
      assert(payload.scenarios.dry.strategy_statistics.S00.strategies.every(row => row.compounds[0] === 'hard'));
      // Critical weather corrections still replace an unsuitable unrun set.
      assert(payload.scenarios.heavy_rain.strategy_statistics.S00.strategies.every(row => row.compounds[0] === 'wet'));

      // Custom-plan metadata and outcomes come from the saved result snapshot,
      // so later edits to the controls cannot rewrite an already rendered race.
      await page.evaluate(() => {
        const scenario = getScenarioEntry().data;
        window.savedCustomPitPlanRace = scenario.sample_race;
        window.savedCustomPitPlanInputs = scenario.simulation_inputs;
        window.savedCustomPitPlanRequest = simResults.request;
        const history = [
          {lap: 18, compound: 'hard', status: 'executed', reason: 'user_plan',
            actual_compound: 'hard', actual_set_id: 'set-2'},
          {lap: 36, compound: 'soft', status: 'overridden', reason: 'forced_repair',
            actual_compound: 'wet', actual_set_id: 'set-3'},
          {lap: 48, compound: 'medium', status: 'skipped',
            reason: 'requested_compound_unavailable', actual_compound: null, actual_set_id: null},
          {lap: 56, compound: 'hard', status: 'not_reached', reason: 'race_finished',
            actual_compound: null, actual_set_id: null},
        ];
        const pitPlans = Object.create(null);
        pitPlans.S00 = history.map(({lap, compound}) => ({lap, compound}));
        pitPlans.S01 = [];
        pitPlans['<img src=x onerror="window.planInjected=true">'] = [{lap: 20, compound: 'wet'}];
        scenario.simulation_inputs = {...(scenario.simulation_inputs || {}), pit_plans: pitPlans};
        simResults.request = {...(simResults.request || {}), pit_plans: pitPlans};
        scenario.sample_race = scenario.sample_race.map((row, index) => ({
          ...row,
          pit_plan_history: index === 0 ? history : index === 1 ? [] : null,
        }));
        renderRace();
      });
      assert((await page.locator('#raceContent').innerText()).includes('S00: L18 Hard'));
      assert((await page.locator('#raceContent').innerText()).includes('S01: No elective stops'));
      assert((await page.locator('#raceContent').innerText()).includes('unlisted drivers automatic'));
      await pitPlanInput.fill('S00=2:medium');
      await page.evaluate(() => renderRace());
      const raceSnapshotText = await page.locator('#raceContent').innerText();
      assert(raceSnapshotText.includes('S00: L18 Hard'));
      assert(!raceSnapshotText.includes('S00: L2 Medium'));
      await page.locator('#samplePitPlanHistory summary').click();
      const pitPlanHistoryText = await page.locator('#samplePitPlanHistory').innerText();
      for (const label of [
        'Executed', 'Overridden by compulsory rule', 'Skipped', 'Not reached',
        'User-directed plan', 'Forced tyre replacement', 'Requested compound unavailable',
        'Race finished before instruction', 'No elective instructions',
        'use automatic strategy or have no recorded custom history',
        'set-2', '—',
      ]) {
        assert(pitPlanHistoryText.includes(label), `Missing custom pit-plan detail: ${label}`);
      }
      assert.equal(await page.locator('#samplePitPlanHistory img, #samplePitPlanHistory svg').count(), 0);
      assert.equal(await page.locator('#samplePitPlanHistory [role="region"]').getAttribute('tabindex'), '0');
      const hostilePitPlanHtml = await page.evaluate(() => renderPitPlanHistory([{
        driver_id: '<img src=x onerror="window.planInjected=true">',
        pit_plan_history: [{lap: 2, compound: 'hard', status: '<svg>', reason: '<img>',
          actual_compound: '<b>', actual_set_id: '<script>'}],
      }]));
      assert(hostilePitPlanHtml.includes('&lt;img') && !hostilePitPlanHtml.includes('<img'));
      assert.equal(await page.evaluate(() => Boolean(window.planInjected)), false);
      const planViewport = page.viewportSize();
      const planRegion = page.locator('#samplePitPlanHistory [role="region"]');
      for (const width of [320, 390, 1440]) {
        await page.setViewportSize({width, height: 1100});
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `Custom pit-plan history overflows at ${width}px`);
        await planRegion.focus();
        assert(await planRegion.evaluate(node => document.activeElement === node));
        if (process.env.F1SIM_SCREENSHOTS && width !== 320) {
          await page.locator('#samplePitPlanHistory').screenshot({path: path.join(
            process.env.F1SIM_SCREENSHOTS, `custom-plan-dashboard-${width}.png`)});
        }
      }
      await page.setViewportSize(planViewport);
      await page.evaluate(() => {
        const scenario = getScenarioEntry().data;
        scenario.sample_race = savedCustomPitPlanRace;
        scenario.simulation_inputs = savedCustomPitPlanInputs;
        simResults.request = savedCustomPitPlanRequest;
        delete window.savedCustomPitPlanRace;
        delete window.savedCustomPitPlanInputs;
        delete window.savedCustomPitPlanRequest;
        renderRace();
      });
      await pitPlanInput.fill('S00=18:hard,36:soft;S01=none');
    }
    await page.locator('#tab-stats').click();
    const reliabilityCard = page.locator('.stat-card').filter({hasText: 'Mechanical Failures'}).first();
    if (offline) {
      const reliabilityText = await reliabilityCard.innerText();
      assert(reliabilityText.includes('Observed simulated failure shares only'));
      assert(reliabilityText.includes('No reference component shares are configured'));
      assert(reliabilityText.includes('Engine') && reliabilityText.includes('66.7%'));
      assert(reliabilityText.includes('Gearbox') && reliabilityText.includes('33.3%'));
      assert.deepEqual((await reliabilityCard.locator('th').allTextContents())
        .map(text => text.trim()), ['Component', 'Observed Share']);
      assert(!reliabilityText.includes('Suggestion'));
      assert(!reliabilityText.includes('Adjustment'));
    }
    await page.locator('#probabilityIntervals summary').click();
    const statistics = Object.values(payload.scenarios)[0].driver_statistics;
    const distance = Object.values(payload.scenarios)[0].race_distance_statistics;
    const suspension = Object.values(payload.scenarios)[0].suspension_statistics;
    assert.equal(distance.recorded_races, 10);
    assert(Number.isInteger(suspension.recorded_races));
    assert(Number.isInteger(suspension.races_with_recorded_suspension));
    assert((await page.locator('#suspensionStatistics').innerText()).includes(
      `${suspension.recorded_races} recorded ${suspension.recorded_races === 1 ? 'race' : 'races'}`));
    assert((await page.locator('#suspensionStatistics').innerText()).includes(
      typeof suspension.mean_completed_suspension_seconds === 'number'
        ? `${suspension.mean_completed_suspension_seconds.toFixed(3)} s`
        : 'Not recorded'));
    assert((await page.locator('#suspensionStatistics').innerText()).includes(
      `${suspension.races_with_recorded_suspension} of ${suspension.recorded_races} recorded`));
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      window.savedSuspensionStatistics = scenario.suspension_statistics;
      scenario.suspension_statistics = {
        recorded_races: 3, races_with_recorded_suspension: 1,
        mean_completed_suspension_seconds: 6,
      };
      renderStats();
    });
    const suspensionText = await page.locator('#suspensionStatistics').innerText();
    assert(suspensionText.includes('6.000 s'));
    assert(suspensionText.includes('3 recorded races'));
    assert(suspensionText.includes('1 of 3 recorded races'));
    await page.evaluate(() => {
      getScenarioEntry().data.suspension_statistics = savedSuspensionStatistics;
      delete window.savedSuspensionStatistics;
      renderStats();
    });
    await page.locator('#probabilityIntervals summary').click();
    assert((await page.locator('#raceDistanceStatistics').innerText()).includes(
      `${distance.mean_winner_laps.toFixed(1)} laps`));
    assert((await page.locator('#raceDistanceStatistics').innerText()).includes(
      `${distance.lapped_finishers} of ${distance.finishers_with_comparable_distance} finishers with known distance`));
    assert.equal(await page.locator('#probabilityIntervals tbody tr').count(),
      Object.keys(statistics).length);
    assert((await page.locator('#probabilityIntervals').innerText()).includes(
      'These ranges do not measure model accuracy'));
    for (const [id, stats] of Object.entries(statistics)) {
      const row = page.locator('#probabilityIntervals tbody tr').filter({
        has: page.locator('th', {hasText: new RegExp(`^${id}$`)}),
      });
      assert.equal(await row.locator('td').first().innerText(), '10');
      const interval = stats.probability_intervals.win;
      assert((await row.innerText()).includes(
        `${interval.lower.toFixed(1)}–${interval.upper.toFixed(1)}%`));
      if (stats.win_rate === 0) assert(interval.upper > 0);
    }
    const pitStatistics = Object.values(payload.scenarios)[0].pit_stop_statistics;
    const pitDecisionStatistics = Object.values(payload.scenarios)[0].pit_decision_statistics;
    const strategyStatistics = Object.values(payload.scenarios)[0].strategy_statistics;
    assert(pitDecisionStatistics && Object.keys(pitDecisionStatistics).length,
      'Dashboard payload must include paid-stop decision statistics');
    assert.equal(await page.locator('#pitDecisionStatistics details').count(),
      Object.keys(pitDecisionStatistics).length);
    const [, decisionStats] = Object.entries(pitDecisionStatistics)
      .sort(([a], [b]) => a.localeCompare(b))[0];
    const decisionDetails = page.locator('#pitDecisionStatistics details').first();
    await decisionDetails.locator('summary').click();
    assert((await decisionDetails.innerText()).includes(
      `${decisionStats.stops_with_recorded_reasons} recognized reasons / ${decisionStats.recorded_stops} paid stops in complete records`));
    const firstReason = Object.entries(decisionStats.reasons || {})[0];
    assert(firstReason, 'Dashboard payload must include a recorded decision reason');
    const firstReasonLabel = await page.evaluate(reason => PIT_DECISION_LABELS[reason], firstReason[0]);
    assert((await decisionDetails.innerText()).includes(firstReasonLabel));
    await page.evaluate(() => {
      window.savedPitDecisionStatistics = getScenarioEntry().data.pit_decision_statistics;
      getScenarioEntry().data.pit_decision_statistics = {
        '<img src=x>': {
          races: 3, races_with_recorded_details: 2, missing_details_races: 1,
          recorded_stops: 1, stops_with_recorded_reasons: 1, missing_reason_stops: 1,
          reasons: {dry_forecast: {stops: 1, share: 1}, '<svg onload=alert(1)>': {stops: 1, share: 1}},
        },
        zero: {
          races: 1, races_with_recorded_details: 1, missing_details_races: 0,
          recorded_stops: 0, stops_with_recorded_reasons: 0, missing_reason_stops: 0,
          reasons: {},
        },
      };
      renderStats();
    });
    const decisionCard = page.locator('#pitDecisionStatistics');
    for (const summary of await decisionCard.locator('summary').all()) {
      await summary.focus();
      await page.keyboard.press('Enter');
    }
    assert.equal(await decisionCard.locator('img, svg').count(), 0);
    assert((await decisionCard.innerText()).includes('No paid stops recorded'));
    assert((await decisionCard.innerText()).includes('missing or unknown reasons'));
    assert.equal(await decisionCard.locator('tbody tr').count(), 1);
    assert.deepEqual(await decisionCard.locator('tbody td').allTextContents(), ['1', '100.0%']);
    await page.evaluate(() => {
      delete getScenarioEntry().data.pit_decision_statistics;
      renderStats();
    });
    assert((await decisionCard.innerText()).includes('No aggregate paid-stop decision observations'));
    await page.evaluate(() => {
      getScenarioEntry().data.pit_decision_statistics = savedPitDecisionStatistics;
      delete window.savedPitDecisionStatistics;
      renderStats();
    });
    assert.equal(await page.locator('#strategyStatistics details').count(),
      Object.keys(strategyStatistics).length);
    const [sequenceId, sequenceStats] = Object.entries(strategyStatistics).sort(([a], [b]) => a.localeCompare(b))[0];
    const sequenceDetails = page.locator('#strategyStatistics details').first();
    await sequenceDetails.locator('summary').click();
    assert((await sequenceDetails.innerText()).includes(`${sequenceId}: ${sequenceStats.races_with_recorded_strategy} recorded of ${sequenceStats.races}`));
    assert.equal(await sequenceDetails.locator('tbody tr').count(), sequenceStats.strategies.length);
    assert.equal(await sequenceDetails.locator('tbody th').first().innerText(),
      sequenceStats.strategies[0].compounds.join(' → '));
    await page.evaluate(() => {
      window.savedStrategyStatistics = getScenarioEntry().data.strategy_statistics;
      getScenarioEntry().data.strategy_statistics = {'<img src=x>': {
        races: 3, races_with_recorded_strategy: 2, missing_strategy_races: 1,
        strategies: [{compounds: ['soft', '<svg onload=alert(1)>', 'soft'], races: 2,
          share: 1, finished_races: 1, dnf_races: 1}],
      }};
      renderStats();
    });
    await page.locator('#strategyStatistics summary').click();
    assert.equal(await page.locator('#strategyStatistics img, #strategyStatistics svg').count(), 0);
    assert((await page.locator('#strategyStatistics').innerText()).includes('Missing tyre sequences: 1'));
    assert.equal(await page.locator('#strategyStatistics tbody th').innerText(),
      'soft → <svg onload=alert(1)> → soft');
    assert.deepEqual(await page.locator('#strategyStatistics tbody td').allTextContents(),
      ['2 (100.0%)', '1', '1']);
    await page.evaluate(() => { delete getScenarioEntry().data.strategy_statistics; renderStats(); });
    assert((await page.locator('#strategyStatistics').innerText()).includes('No tyre sequences were recorded'));
    await page.evaluate(() => {
      getScenarioEntry().data.strategy_statistics = savedStrategyStatistics;
      delete window.savedStrategyStatistics;
      renderStats();
    });
    assert.equal(await page.locator('#pitStopStatistics tbody tr').count(),
      Object.keys(pitStatistics).length);
    await page.locator('#pitLossDetails summary').focus();
    await page.keyboard.press('Enter');
    const pitLosses = await page.evaluate(() => getScenarioEntry().data.pit_loss_statistics);
    assert.equal(await page.locator('#pitLossStatistics tbody tr').count(), Object.keys(pitLosses).length);
    for (const [id, stats] of Object.entries(pitLosses)) {
      const row = page.locator('#pitLossStatistics tbody tr').filter({
        has: page.locator('th', {hasText: new RegExp(`^${id}$`)}),
      });
      assert.equal(await row.locator('td').nth(1).innerText(), `${stats.mean_total_loss_per_race.toFixed(3)} s`);
      assert.equal(await row.locator('td').nth(3).innerText(), `${stats.mean_service_time_per_race.toFixed(3)} s`);
    }
    await page.evaluate(() => {
      window.savedPitLosses = getScenarioEntry().data.pit_loss_statistics;
      getScenarioEntry().data.pit_loss_statistics = {
        '<img src=x>': {races: 1, races_with_recorded_details: 0},
        zero: {races: 1, races_with_recorded_details: 1, recorded_stops: 0, queued_stops: 0,
          mean_total_loss_per_race: 0, mean_lane_loss_per_race: 0, mean_service_time_per_race: 0,
          mean_queue_time_per_race: 0, queue_race_rate: 0, races_with_queue: 0},
      };
      renderStats();
    });
    await page.locator('#pitLossDetails summary').click();
    assert.equal(await page.locator('#pitLossStatistics img').count(), 0);
    assert.equal(await page.locator('#pitLossStatistics tbody tr').first().locator('td').nth(1).innerText(), 'Not recorded');
    assert.equal(await page.locator('#pitLossStatistics tbody tr').last().locator('td').nth(1).innerText(), '0.000 s');
    assert.equal(await page.locator('#pitLossStatistics tbody tr').last().locator('td').last().innerText(), '0.0% (0)');
    await page.evaluate(() => {
      getScenarioEntry().data.pit_loss_statistics = savedPitLosses;
      delete window.savedPitLosses;
      renderStats();
    });
    for (const [id, stats] of Object.entries(pitStatistics)) {
      const row = page.locator('#pitStopStatistics tbody tr').filter({
        has: page.locator('th', {hasText: new RegExp(`^${id}$`)}),
      });
      assert.equal(await row.locator('td').nth(0).innerText(), String(stats.races));
      assert.equal(await row.locator('td').nth(1).innerText(), stats.average_stops.toFixed(2));
    }
    // Group high-stop outcomes and preserve a real zero; escape driver labels.
    await page.evaluate(() => {
      window.savedPitStatistics = getScenarioEntry().data.pit_stop_statistics;
      getScenarioEntry().data.pit_stop_statistics = {'<img src=x>': {
        races: 4, average_stops: 2.25, stop_count_distribution: {0: 1, 1: 1, 3: 1, 5: 1},
      }};
      renderStats();
    });
    assert.equal(await page.locator('#pitStopStatistics img').count(), 0);
    assert.equal(await page.locator('#pitStopStatistics tbody th').textContent(), '<img src=x>');
    assert.deepEqual(await page.locator('#pitStopStatistics tbody td').allTextContents(),
      ['4', '2.25', '25.0%', '25.0%', '0.0%', '50.0%']);
    await page.evaluate(() => {
      getScenarioEntry().data.pit_stop_statistics = {};
      renderStats();
    });
    assert.equal(await page.locator('#pitStopStatistics').count(), 0);
    assert((await page.locator('.pit-summary-card').innerText()).includes('No aggregate pit-stop observations'));
    await page.evaluate(() => {
      getScenarioEntry().data.pit_stop_statistics = savedPitStatistics;
      delete window.savedPitStatistics;
      renderStats();
    });
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      window.savedDistance = scenario.race_distance_statistics;
      delete scenario.race_distance_statistics;
      renderStats();
    });
    assert((await page.locator('#raceDistanceStatistics').innerText()).includes('not recorded'));
    await page.evaluate(() => {
      getScenarioEntry().data.race_distance_statistics = savedDistance;
      delete window.savedDistance;
      renderStats();
    });
    // Explicit classified and unclassified retirements must remain distinct.
    await page.locator('#tab-race').click();
    for (const row of Object.values(payload.scenarios)[0].sample_race) {
      assert(Array.isArray(row.pit_laps), 'Current race results must include actual pit laps');
      assert.equal(row.pit_laps.length, row.pit_stops);
    }
    await page.evaluate(() => {
      window.savedClassificationSample = getScenarioEntry().data.sample_race;
      getScenarioEntry().data.sample_race = savedClassificationSample.slice(0, 3).map((row, index) => ({
        ...row, position: index + 1, status: index === 0 ? 'finished' : 'dnf',
        classified: index < 2, laps_completed: [60, 54, 53][index],
        fastest_lap: [90, 89, 91][index], dnf_reason: index ? 'Engine failure' : null,
        pit_stops: [3, 0, 1][index], pit_laps: [[7, 17, 34], [], null][index],
      }));
      renderRace();
    });
    const classifiedRetirement = page.locator('#raceContent .driver-row').nth(1);
    assert.equal(await classifiedRetirement.locator('.pos').innerText(), '2');
    assert((await classifiedRetirement.locator('.status').getAttribute('aria-label'))
      .includes('Retired · Classified · 54 laps · Engine failure'));
    assert((await classifiedRetirement.getAttribute('class')).includes('podium'));
    assert.equal(await classifiedRetirement.locator('.is-fastest').count(), 1);
    const unclassifiedRetirement = page.locator('#raceContent .driver-row').nth(2);
    assert.equal(await unclassifiedRetirement.locator('.pos').innerText(), 'NC');
    assert(!(await unclassifiedRetirement.getAttribute('class')).includes('podium'));
    assert.equal(await page.locator('#raceContent .pit-laps').first().innerText(), 'L7 · L17 · L34');
    assert.equal(await classifiedRetirement.locator('.pits').getAttribute('title'), 'No paid pit stops');
    assert.equal(await unclassifiedRetirement.locator('.pits').getAttribute('title'), 'Pit laps unavailable');
    assert.equal(await page.locator('#raceContent .time-limit-note').count(), 0);
    await page.evaluate(() => {
      getScenarioEntry().data.sample_race.forEach((row, index) => {
        row.race_time_limited = true;
        row.laps_completed = [5, 4, 3][index];
        row.points_awarded = [19, 12, 0][index];
      });
      renderRace();
    });
    assert((await page.locator('#raceContent .time-limit-note').innerText()).includes('two-hour limit'));
    assert((await page.locator('#raceContent .race-header').innerText()).includes('5 of'));
    assert((await page.locator('#raceContent .driver-row .status').first().innerText()).includes('19 pts'));
    assert((await page.locator('#raceContent .driver-row .status').nth(2).innerText()).includes('0 pts'));
    assert.equal(await page.locator('#raceContent .driver-row .gap').nth(1).innerText(), 'DNF');
    await page.evaluate(() => {
      getScenarioEntry().data.sample_race.forEach(row => {
        row.status = 'finished';
        row.classified = true;
        row.dnf_reason = null;
      });
      renderRace();
    });
    assert.equal(await page.locator('#raceContent .driver-row .gap').nth(1).innerText(), '+1 lap');
    assert.equal(await page.locator('#raceContent .driver-row .gap').nth(2).innerText(), '+2 laps');
    assert.equal(await page.evaluate(() => formatGap({
      position: 2, status: 'finished', gap_to_leader: 90,
    }, 5)), '+90.000');
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({width, height: 900});
      await page.locator('#sampleWeatherHistory').evaluate(node => { node.open = true; });
      await page.locator('#samplePitStopDetails').evaluate(node => { node.open = true; });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
        `Retirement classification overflows at ${width}px`);
      assert(await page.locator('#raceContent .pit-laps').first().isVisible(),
        `Pit laps must remain visible at ${width}px`);
      if (process.env.F1SIM_SCREENSHOT_DIR && [390, 1440].includes(width)) {
        await page.screenshot({path: path.join(process.env.F1SIM_SCREENSHOT_DIR,
          `pit-laps-${width}.png`), fullPage: true, animations: 'disabled'});
      }
    }
    await page.evaluate(() => {
      getScenarioEntry().data.sample_race = savedClassificationSample;
      delete window.savedClassificationSample;
      renderRace();
    });
    await page.locator('#tab-qualifying').click();
    await page.evaluate(() => {
      window.savedQualifyingSample = getScenarioEntry().data.sample_qualifying;
      getScenarioEntry().data.sample_qualifying = [...savedQualifyingSample, {
        driver_id: 'MISSING', driver_name: 'Missing car', team: 'Unknown',
        position: savedQualifyingSample.length + 1, best_time: null,
        q1_time: null, q2_time: null, q3_time: null, eliminated_in: 'Q1',
      }];
      renderQualifying();
    });
    const missingQualifying = page.locator('#qualiContent .quali-row').filter({
      has: page.locator('.driver-code', {hasText: /^MISSING$/}),
    });
    assert.deepEqual((await missingQualifying.locator('.quali-time').allTextContents())
      .map(text => text.trim()), ['--', '--', '--']);
    assert((await missingQualifying.getAttribute('class')).includes('eliminated'));
    assert.equal(await missingQualifying.locator('.best').count(), 0);
    await page.evaluate(() => {
      getScenarioEntry().data.sample_qualifying = savedQualifyingSample;
      delete window.savedQualifyingSample;
      renderQualifying();
    });
    await page.locator('#tab-stats').click();
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      for (const tab of ['race', 'qualifying', 'stats', 'scenarios']) {
        await page.locator(`#tab-${tab}`).click();
        if (tab === 'stats') await page.locator('#pitLossDetails').evaluate(node => { node.open = true; });
        if (tab === 'race') await page.locator('#samplePitStopDetails').evaluate(node => { node.open = true; });
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `${tab} overflows at ${width}px`);
        if (tab === 'race' && process.env.F1SIM_SCREENSHOTS && [390, 1440].includes(width)) {
          await page.locator('#samplePitStopDetails').screenshot({
            path: path.join(process.env.F1SIM_SCREENSHOTS, `pit-decisions-${width}.png`),
          });
        }
        if (tab === 'scenarios') {
          assert(await page.locator('#compareChart .bar-track').first().evaluate(
            node => node.getBoundingClientRect().height >= 10), 'Win bars must have visible height');
          assert(await page.locator('#compareTrends .bar-track').first().evaluate(
            node => node.getBoundingClientRect().height >= 10), 'Event bars must have visible height');
          if (process.env.F1SIM_SCREENSHOTS && [390, 1440].includes(width)) {
            await page.locator('#compareChart').screenshot({
              path: path.join(process.env.F1SIM_SCREENSHOTS, `scenario-ranges-${width}.png`),
            });
          }
        }
        if (tab === 'stats' && process.env.F1SIM_SCREENSHOTS && [390, 1440].includes(width)) {
          if (width === 1440) await page.setViewportSize({width, height: 1600});
          await page.locator('#strategyStatistics details').first().evaluate(node => { node.open = true; });
          await page.locator('#strategyStatistics').screenshot({
            path: path.join(process.env.F1SIM_SCREENSHOTS, `tyre-sequences-${width}.png`),
          });
          if (width === 1440) await page.setViewportSize({width, height: 1300});
          await page.locator('.pit-summary-card').evaluate(card => card.scrollIntoView({block: 'center'}));
          await page.locator('.pit-summary-card').screenshot({
            path: path.join(process.env.F1SIM_SCREENSHOTS, `pit-stops-${width}.png`),
          });
          if (await page.locator('#pitDecisionStatistics details').count()) {
            if (width === 1440) await page.setViewportSize({width, height: 1800});
            await page.locator('#pitDecisionStatistics details').first().evaluate(node => { node.open = true; });
            await page.locator('#pitDecisionStatistics').evaluate(card => card.scrollIntoView({block: 'center'}));
            await page.locator('#pitDecisionStatistics').screenshot({
              path: path.join(process.env.F1SIM_SCREENSHOTS, `pit-decision-statistics-${width}.png`),
            });
          }
          if (width === 1440) await page.setViewportSize({width, height: 900});
        }
        if (tab === 'race' || tab === 'qualifying') {
          assert(await page.locator('.tab-panel.active .team-stripe').first().evaluate(stripe =>
            stripe.getBoundingClientRect().right <= stripe.nextElementSibling.getBoundingClientRect().left),
          `Team stripe overlaps driver code at ${width}px`);
        }
      }
    }
    assert.equal(await page.evaluate(() => formatGap({position: 2, gap_to_leader: 0})), '+0.000');
    assert.equal(await page.evaluate(() => formatGap({position: 1, gap_to_leader: 0})), 'LEADER');
    assert.equal(await page.locator('#scenarioContent pre').count(), 0);
    const firstScenario = Object.keys(payload.scenarios)[0];
    const firstChartDriver = payload.scenarios[firstScenario].win_probabilities[0][0];
    const expectedRange = payload.scenarios[firstScenario].driver_statistics[firstChartDriver].probability_intervals;
    const firstChartRow = page.locator('#compareChart .chart-row').first();
    assert((await firstChartRow.innerText()).includes(`${expectedRange.trials} observed trials`));
    const extent = await firstChartRow.locator('.bar-range').evaluate(node => {
      const outer = node.parentElement.getBoundingClientRect(), inner = node.getBoundingClientRect();
      return [(inner.left-outer.left)/outer.width*100, inner.width/outer.width*100];
    });
    assert(Math.abs(extent[0]-expectedRange.win.lower) < .1);
    assert(Math.abs(extent[1]-(expectedRange.win.upper-expectedRange.win.lower)) < .1);
    await page.evaluate(() => {
      window.originalComparisonScenarios = simResults.scenarios;
      simResults.scenarios = {
        observed: {win_probabilities: [['TEST', 0]], driver_statistics: {TEST: {
          probability_intervals: {trials: 4, confidence: .95, method: 'wilson',
            scope: 'monte_carlo_sampling', win: {lower: 0, upper: 49}},
        }}},
        missing: {win_probabilities: []},
        empty: {win_probabilities: [['TEST', 0]], driver_statistics: {TEST: {
          probability_intervals: {trials: 0},
        }}},
        legacy: {win_probabilities: [['TEST', 50]]},
      };
      renderScenarioWinChart(simResults);
      renderDriverMatrix(simResults);
    });
    assert.equal(await page.locator('#compareChart .bar-range').count(), 1);
    assert.equal(await page.locator('#compareChart .bar-track').count(), 2);
    const matrixCells = await page.locator('#compareMatrix tbody tr').first().locator('td').allTextContents();
    assert(matrixCells[1].includes('0.0%') && matrixCells[1].includes('4 observed trials'));
    assert.deepEqual(matrixCells.slice(2, 4), ['Not recorded', 'Not recorded']);
    assert(matrixCells[4].includes('50.0%') && matrixCells[4].includes('Sampling range unavailable'));
    const missingCsvPromise = page.waitForEvent('download');
    await page.locator('#downloadScenarioMatrixBtn').click();
    const missingCsv = await missingCsvPromise;
    assert(readFileSync(await missingCsv.path(), 'utf8').includes('"TEST","0.0","","","50.0"'));
    await page.evaluate(() => {
      simResults.scenarios = window.originalComparisonScenarios;
      delete window.originalComparisonScenarios;
      updateScenarioViews();
    });
    for (const [id, extension] of [
      ['downloadScenarioJsonBtn', '.json'], ['downloadScenarioMatrixBtn', '.csv'],
      ['downloadScenarioReportBtn', '.html'],
    ]) {
      const downloadPromise = page.waitForEvent('download');
      await page.locator(`#${id}`).click();
      const download = await downloadPromise;
      assert(download.suggestedFilename().endsWith(extension));
      if (extension === '.json') {
        const saved = JSON.parse(readFileSync(await download.path(), 'utf8'));
        assert.equal(saved.comparison_report_html, undefined);
        for (const [name, scenario] of Object.entries(saved.scenarios)) {
          assert.deepEqual(scenario.simulation_inputs, payload.scenarios[name].simulation_inputs);
          assert.deepEqual(scenario.strategy_statistics, payload.scenarios[name].strategy_statistics);
          assert.equal(scenario.simulation_inputs.schema_version, offline ? 4 : 2);
          if (offline) {
            assert.deepEqual(scenario.simulation_inputs.tire_inventory, fixture.payload.request.tire_inventory);
            assert(scenario.sample_race.find(row => row.driver_id === 'S00').tire_set_history.length);
          }
          assert.equal(scenario.simulation_inputs.rng_policy, 'isolated_weather_v1');
        }
      } else if (extension === '.html') {
        const report = readFileSync(await download.path(), 'utf8');
        assert.equal(report, payload.comparison_report_html);
        assert(report.includes('Simulation comparison'));
        assert(!report.includes('<script'));
      }
    }
    await page.evaluate(() => {
      window.savedComparisonReport = simResults.comparison_report_html;
      delete simResults.comparison_report_html;
      updateScenarioViews();
    });
    assert(await page.locator('#downloadScenarioReportBtn').isDisabled());
    await page.evaluate(() => {
      simResults.comparison_report_html = window.savedComparisonReport;
      delete window.savedComparisonReport;
      updateScenarioViews();
    });
    assert(await page.locator('#downloadScenarioReportBtn').isEnabled());
    await page.locator('#compareDriverFilter').fill(driverId);
    assert((await page.locator('#compareMatrix').innerText()).includes(driverId));
    await page.locator('#tab-race').focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('[role=tab][aria-selected=true]').getAttribute('id'),
      'tab-qualifying');
    if (offline) {
      const cancelledPayload = fixture.payload;
      const previousResults = await page.evaluate(() => ({
        year: simResults?.year,
        race: simResults?.race,
        raceHtml: document.getElementById('raceContent')?.innerHTML,
      }));
      holdNextRun = true;
      await page.locator('#btnRun').click();
      await page.locator('#btnStop').waitFor({state: 'visible'});
      assert(await page.locator('#btnRun').isDisabled());
      assert(await page.locator('#btnTyreSetup').isDisabled());
      assert(await page.locator('#pitPlansInput').isDisabled());
      assert(await page.locator('#btnStop').isEnabled());
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'btnStop');
      await page.locator('#trackSelect').focus();
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'btnStop');
      await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'btnStop');
      await page.keyboard.press('Shift+Tab');
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'btnStop');
      await page.waitForFunction(() => {
        const overlay = document.getElementById('simOverlay');
        return overlay && overlay.classList.contains('active')
          && getComputedStyle(overlay).opacity === '1';
      });
      if (process.env.F1SIM_SCREENSHOTS) {
        const overlayStyles = await page.evaluate(() => {
          const overlay = document.getElementById('simOverlay');
          const stop = document.getElementById('btnStop');
          const run = document.getElementById('btnRun');
          return {
            overlayOpacity: getComputedStyle(overlay).opacity,
            overlayBackground: getComputedStyle(overlay).backgroundColor,
            overlayZIndex: getComputedStyle(overlay).zIndex,
            stopOpacity: getComputedStyle(stop).opacity,
            stopColor: getComputedStyle(stop).color,
            stopBackground: getComputedStyle(stop).backgroundColor,
            stopZIndex: getComputedStyle(stop).zIndex,
            runOpacity: getComputedStyle(run).opacity,
          };
        });
        assert.equal(overlayStyles.overlayOpacity, '1');
        assert.equal(overlayStyles.stopOpacity, '1');
        console.log(`Stable cancellation overlay styles: ${JSON.stringify(overlayStyles)}`);
        for (const width of [390, 1440]) {
          await page.setViewportSize({width, height: 900});
          await page.screenshot({path: path.join(process.env.F1SIM_SCREENSHOTS,
            `simulation-cancel-${width}.png`), fullPage: false});
        }
      }
      await page.locator('#btnStop').focus();
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'btnStop');
      await page.keyboard.press('Enter');
      assert(await page.locator('#btnStop').isDisabled());
      await page.waitForFunction(() => !runInProgress);
      assert(await page.locator('#btnRun').isEnabled());
      assert(await page.locator('#btnStop').isHidden());
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'btnRun');
      assert((await page.locator('#appStatus').innerText()).includes('Run cancelled'));
      assert((await page.locator('#appStatus').innerText()).includes(
        'server trials may finish before capacity is available'));
      assert.deepEqual(await page.evaluate(() => ({
        year: simResults?.year,
        race: simResults?.race,
        raceHtml: document.getElementById('raceContent')?.innerHTML,
      })), previousResults);

      const followUpPayload = JSON.parse(JSON.stringify(cancelledPayload));
      followUpPayload.year = 2027;
      followUpPayload.race = 'FOLLOW-UP synthetic';
      followUpPayload.track = followUpPayload.race;
      fixture.payload = followUpPayload;
      const followUpResponsePromise = page.waitForResponse(response => response.url().endsWith('/api/run'));
      await page.locator('#btnRun').click();
      const followUpResponse = await followUpResponsePromise;
      assert.equal(followUpResponse.status(), 200);
      await page.waitForFunction(() => !runInProgress);
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'btnRun');
      assert((await page.locator('#appStatus').innerText()).includes('Backend run complete for 2027 FOLLOW-UP synthetic'));
      assert.deepEqual(await page.evaluate(() => ({year: simResults?.year, race: simResults?.race})), {
        year: 2027, race: 'FOLLOW-UP synthetic',
      });

      if (delayedRunRoute) {
        try {
          await delayedRunRoute.fulfill({json: cancelledPayload});
        } catch {
          // The browser may have already torn down the aborted request.
        }
        delayedRunRoute = null;
      }
      await page.waitForTimeout(100);
      assert.deepEqual(await page.evaluate(() => ({year: simResults?.year, race: simResults?.race})), {
        year: 2027, race: 'FOLLOW-UP synthetic',
      });
    }
    for (const [status, detail] of [
      [503, 'Provider temporarily unavailable'], [429, 'Simulation capacity is busy'],
    ]) {
      await page.route('**/api/run', route => route.fulfill({
        status, contentType: 'application/json', body: JSON.stringify({detail}),
      }));
      await page.locator('#btnRun').click();
      await page.waitForFunction(() => !runInProgress);
      assert((await page.locator('#appStatus').innerText()).includes(detail));
      assert(await page.locator('#btnRun').isEnabled());
    }
    await page.route('**/api/run', route => route.abort());
    await page.locator('#btnRun').click();
    await page.waitForFunction(() => !runInProgress);
    assert(await page.locator('#btnRun').isDisabled());
    await page.locator('#btnRefresh').click();
    await page.waitForFunction(() => !connectionRefreshInProgress);
    assert(await page.locator('#btnRun').isEnabled());
    await page.route('**/api/calendar*', route => route.fulfill({
      status: 503, contentType: 'application/json', body: '{"detail":"Calendar unavailable"}',
    }));
    await page.locator('#btnRefresh').click();
    await page.waitForFunction(() => !connectionRefreshInProgress);
    assert(await page.locator('#btnRun').isDisabled());
    await page.unroute('**/api/calendar*');
    await page.locator('#btnRefresh').click();
    await page.waitForFunction(() => !connectionRefreshInProgress);
    assert(await page.locator('#btnRun').isEnabled());
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpectedRequests, [], 'Offline test encountered unexpected network traffic');
    console.log(`Browser audit passed (${offline ? 'SYNTHETIC offline' : 'live'}): ` +
      'run, responsive views, exports, tabs, filters, recovery.');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
