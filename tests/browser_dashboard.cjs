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
    let holdNextRosterRequest = false;
    let delayedRosterRoute = null;
    let resolveHeldRosterRoute = null;
    const heldRosterRoutePromise = new Promise(resolve => { resolveHeldRosterRoute = resolve; });
    const alternateRaceName = 'ALT_TEST_CIRCUIT';
    const selectionFixtureResponse = requestPayload => {
      const response = JSON.parse(JSON.stringify(fixture.payload));
      const request = requestPayload.pit_plan_selection;
      const candidateLabels = Object.keys(request.plans);
      const referenceLabel = request.reference_label;
      const selectedLabel = candidateLabels.find(label => label !== referenceLabel) || referenceLabel;
      const labels = candidateLabels;
      const frozenPlans = Object.create(null);
      for (const label of candidateLabels) {
        if (!request.driver_id) {
          frozenPlans[label] = JSON.parse(JSON.stringify(request.plans[label]));
          continue;
        }
        const memberPlans = Object.create(null);
        const instructions = request.plans[label];
        if (instructions !== null) memberPlans[request.driver_id] = instructions;
        else if (label === referenceLabel) {
          // Model the backend's frozen full mapping: an unrelated custom plan
          // must not be mistaken for the automatic plan of the selected driver.
          memberPlans.S01 = [{lap: 2, compound: 'soft'}];
        }
        frozenPlans[label] = memberPlans;
      }
      response.request.pit_plan_selection = request;
      response.strategy_selections = Object.create(null);
      for (const scenarioName of Object.keys(response.scenarios)) {
        const scenario = response.scenarios[scenarioName];
        const heldOutDifference = scenarioName === 'light_rain' ? 2.5 : -1;
        const sourceSeed = requestPayload.seed;
        const sourceCount = requestPayload.simulations;
        const trainingFirstSeed = sourceSeed + sourceCount;
        const trainingLastSeed = trainingFirstSeed + request.training_simulations - 1;
        const validationFirstSeed = trainingLastSeed + 1;
        const validationLastSeed = validationFirstSeed + request.validation_simulations - 1;
        const candidateRows = labels.map((label, index) => ({
          label,
          total_points: (label === selectedLabel ? 8 : 4) * request.training_simulations,
          mean_points: label === selectedLabel ? 8 : 4,
          trials: request.training_simulations,
        }));
        const trainingScenarios = Object.create(null);
        const validationScenarios = Object.create(null);
        labels.forEach(label => { trainingScenarios[label] = scenario; });
        [referenceLabel, selectedLabel].forEach(label => { validationScenarios[label] = scenario; });
        const metadata = {
          schema_version: 1,
          target_mode: request.driver_id ? 'driver' : 'constructor',
          target_id: request.driver_id || request.constructor_id,
          target_member_ids: request.driver_id ? [request.driver_id]
            : Object.keys(request.plans[selectedLabel] || request.plans[referenceLabel] || {}),
          candidate_order: labels,
          selected_label: selectedLabel,
          reference_label: referenceLabel,
          selection_status: selectedLabel === referenceLabel ? 'no_change' : 'selected',
          tiebreak_applied: 'unique_highest_training_mean',
          training_score_table: candidateRows,
          seed_ranges: {
            training: {first_seed: trainingFirstSeed, last_seed: trainingLastSeed,
              trials: request.training_simulations},
            validation: {first_seed: validationFirstSeed, last_seed: validationLastSeed,
              trials: request.validation_simulations},
          },
          validation_status: selectedLabel === referenceLabel ? 'no_change' : 'evaluated',
          validation_target_metrics: selectedLabel === referenceLabel ? {
            reference_mean_points: 5, selected_mean_points: 5, mean_points_difference: 0,
            points_difference_standard_error: null, paired_races: request.validation_simulations,
            points_outcome_profile: null,
          } : {
            reference_mean_points: 5, selected_mean_points: 5 + heldOutDifference,
            mean_points_difference: heldOutDifference,
            points_difference_standard_error: 0.5, paired_races: request.validation_simulations,
            points_outcome_profile: {paired_races: request.validation_simulations,
              more_points_races: 5, equal_points_races: 10, fewer_points_races: 35,
              mean_points_gain_when_ahead: 2, mean_points_loss_when_behind: 3},
          },
        };
        response.strategy_selections[scenarioName] = {
          selection: metadata,
          plans: frozenPlans,
          source: {seed: sourceSeed, num_simulations: sourceCount,
            simulation_inputs: scenario.simulation_inputs},
          training: {scenarios: trainingScenarios},
          validation: {scenarios: validationScenarios},
          validation_report_html: fixture.comparison_payload.strategy_comparison_reports[scenarioName],
        };
        if (request.rival_scenarios) {
          const entry = response.strategy_selections[scenarioName];
          const rivals = Object.entries(request.rival_scenarios);
          const totalWeight = rivals.reduce((sum, [, item]) => sum + item.weight, 0);
          metadata.method = 'weighted_rival_scenario_training_then_disjoint_seed_validation';
          metadata.tiebreak_applied = 'unique_highest_weighted_training_mean';
          metadata.training_score_table = candidateRows.map(row => ({...row,
            mean_points_behind_selected: 8 - row.mean_points,
            tied_for_best: row.label === selectedLabel,
          }));
          metadata.rival_scenarios = rivals.map(([name, item]) => ({
            name, weight: item.weight, normalized_weight: item.weight / totalWeight,
            rival_pit_plans: item.pit_plans,
          }));
          metadata.training_scenario_score_tables = Object.fromEntries(rivals.map(([name, item]) => [name, {
            weight: item.weight, normalized_weight: item.weight / totalWeight, scores: candidateRows,
          }]));
          metadata.validation_scenario_metrics = Object.fromEntries(rivals.map(([name]) => [name, metadata.validation_target_metrics]));
          entry.plans = request.plans;
          entry.plans_by_rival_scenario = Object.fromEntries(rivals.map(([name, assumption]) => [name,
            Object.fromEntries(labels.map(label => {
              const effective = {...requestPayload.pit_plans};
              for (const [driver, instructions] of Object.entries(assumption.pit_plans)) {
                if (instructions === null) delete effective[driver];
                else effective[driver] = instructions;
              }
              for (const member of metadata.target_member_ids) delete effective[member];
              if (request.plans[label] !== null) {
                if (request.driver_id) effective[request.driver_id] = request.plans[label];
                else Object.assign(effective, request.plans[label]);
              }
              return [label, effective];
            }))]));
          entry.training_by_rival_scenario = Object.fromEntries(rivals.map(([name]) => [name, entry.training]));
          entry.validation_by_rival_scenario = Object.fromEntries(rivals.map(([name]) => [name, entry.validation]));
          delete entry.training;
          delete entry.validation;
          entry.validation_report_html = '<!doctype html><title>Weighted rival selection</title><p>Frozen weighted validation evidence</p>';
        }
      }
      return response;
    };
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
        if (url.pathname === '/api/ratings') {
          if (holdNextRosterRequest) {
            holdNextRosterRequest = false;
            delayedRosterRoute = route;
            resolveHeldRosterRoute(route);
            return;
          }
          const race = url.searchParams.get('race');
          const ratings = race === alternateRaceName ? {drivers: [
            {id: 'ALT', name: 'Alternate Test Driver', team: 'unknown_display_key',
              constructor_id: 'Mystery Factory / Team'},
          ]} : {drivers: fixture.payload.ratings.drivers.map(driver => ({
            id: driver.id, name: driver.name, team: driver.team_key,
            constructor_id: driver.constructor_id,
          }))};
          return route.fulfill({json: ratings});
        }
        let runBody = fixture.payload;
        if (url.pathname === '/api/run' && fixture.comparison_payload) {
          let requestPayload = null;
          try { requestPayload = route.request().postDataJSON(); } catch { requestPayload = null; }
          if (requestPayload?.pit_plan_selection) runBody = selectionFixtureResponse(requestPayload);
          else if (requestPayload?.compare_automatic === true) runBody = fixture.comparison_payload;
        }
        const body = url.pathname === '/api/run' ? runBody
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

    if (offline) {
      const trackSelect = page.locator('#trackSelect');
      const originalRace = await trackSelect.inputValue();
      const firstRosterRequest = page.waitForRequest(request =>
        new URL(request.url()).pathname === '/api/ratings');
      holdNextRosterRequest = true;
      await page.locator('#pitPlanSelectionEnabled').check();
      await firstRosterRequest;
      const heldRosterRoute = await heldRosterRoutePromise;
      await page.evaluate(race => {
        const select = document.getElementById('trackSelect');
        select.add(new Option('R2 · Alternate roster test', race));
      }, alternateRaceName);
      const alternateRosterRequest = page.waitForRequest(request => {
        const url = new URL(request.url());
        return url.pathname === '/api/ratings' && url.searchParams.get('race') === alternateRaceName;
      });
      await trackSelect.selectOption(alternateRaceName);
      await alternateRosterRequest;
      await page.waitForFunction(() => document.getElementById('pitPlanSelectionStatus')
        .textContent.includes('Loaded 1 drivers'));
      assert.deepEqual(await page.locator('#pitPlanSelectionTargetId option').evaluateAll(options =>
        options.map(option => option.value)), ['', 'ALT']);
      await page.locator('#pitPlanSelectionTargetMode').selectOption('constructor');
      assert.deepEqual(await page.locator('#pitPlanSelectionTargetId option').evaluateAll(options =>
        options.map(option => option.value)), ['', 'Mystery Factory / Team'],
      'Constructor choices must use the raw constructor ID, not the normalized display key');
      assert.equal(await page.locator('#pitPlanSelectionTargetId').inputValue(), 'Mystery Factory / Team');
      await page.locator('#pitPlanSelectionTargetMode').selectOption('driver');
      try {
        await heldRosterRoute.fulfill({json: fixture.payload.ratings});
      } catch {
        // The aborted request may already have been discarded by the browser.
      }
      await page.waitForTimeout(30);
      assert.deepEqual(await page.locator('#pitPlanSelectionTargetId option').evaluateAll(options =>
        options.map(option => option.value)), ['', 'ALT'],
      'A late response for the previous race must not replace the active roster');
      const currentRosterRequest = page.waitForRequest(request => {
        const url = new URL(request.url());
        return url.pathname === '/api/ratings' && url.searchParams.get('race') === originalRace;
      });
      await trackSelect.selectOption(originalRace);
      await currentRosterRequest;
      await page.waitForFunction(() => document.getElementById('pitPlanSelectionTargetId')
        .querySelector('option[value="S00"]'));
      assert.equal(await page.locator('#pitPlanSelectionTargetId').inputValue(), 'S00');
      await page.locator('#pitPlanSelectionEnabled').uncheck();
    }

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
    const compareAutomaticInput = page.locator('#compareAutomaticInput');
    const rngPolicyInput = page.locator('#rngPolicySelect');
    assert.equal(await rngPolicyInput.inputValue(), 'isolated_weather_v1');
    assert.equal(await compareAutomaticInput.isChecked(), false);
    await pitPlanInput.fill(offline ? 'S00=18:hard,36:soft;S01=none' : '');
    if (offline) {
      await page.locator('#pitPlanSelectionEnabled').check();
      const candidateRows = page.locator('#pitPlanSelectionCandidates .pit-selection-candidate');
      assert.equal(await candidateRows.count(), 2);
      await candidateRows.nth(0).locator('.pit-selection-label').fill('__proto__');
      await candidateRows.nth(1).locator('.pit-selection-label').fill('<img src=x onerror=alert(1)>');
      await candidateRows.nth(1).locator('[data-member-id="S00"]').fill('none');
      const driverSelection = await page.evaluate(() => {
        const payload = buildRunPayload();
        const selection = payload?.pit_plan_selection;
        return {
          selection,
          plansNullPrototype: Object.getPrototypeOf(selection?.plans) === null,
          ownsProtoLabel: Object.hasOwn(selection?.plans || {}, '__proto__'),
          plansJson: JSON.stringify(selection?.plans),
          estimate: document.getElementById('pitPlanSelectionEstimate').textContent,
        };
      });
      assert.equal(driverSelection.plansNullPrototype, true);
      assert.equal(driverSelection.ownsProtoLabel, true);
      assert.equal(driverSelection.selection.driver_id, 'S00');
      assert.equal(Object.hasOwn(driverSelection.selection, 'constructor_id'), false);
      assert.equal(driverSelection.selection.reference_label, '__proto__');
      assert.equal(driverSelection.selection.training_simulations, 50);
      assert.equal(driverSelection.selection.validation_simulations, 50);
      const parsedDriverPlans = JSON.parse(driverSelection.plansJson);
      assert.equal(Object.hasOwn(parsedDriverPlans, '__proto__'), true);
      assert.equal(Object.getOwnPropertyDescriptor(parsedDriverPlans, '__proto__').value, null,
        'Automatic mode must emit null rather than an empty no-stop list');
      assert.deepEqual(driverSelection.selection.plans['<img src=x onerror=alert(1)>'], [],
        'The none shorthand must emit an explicit empty stop list');
      assert(driverSelection.estimate.includes('210 trials'));

      await candidateRows.nth(1).locator('[data-member-id="S00"]').fill('18:hard');
      const driverInstructions = await page.evaluate(() => buildRunPayload().pit_plan_selection.plans[
        '<img src=x onerror=alert(1)>'
      ]);
      assert.deepEqual(driverInstructions, [{lap: 18, compound: 'hard'}]);
      await candidateRows.nth(1).locator('[data-member-id="S00"]').fill('18:Hard');
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
      assert.equal(await page.evaluate(() => document.activeElement?.dataset.memberId), 'S00');
      await candidateRows.nth(1).locator('[data-member-id="S00"]').fill('18:hard');

      await page.locator('#pitPlanSelectionTargetMode').selectOption('constructor');
      await page.locator('#pitPlanSelectionTargetId').selectOption('team0');
      await candidateRows.nth(1).locator('[data-member-id="S00"]').fill('18:hard');
      await candidateRows.nth(1).locator('[data-member-id="S01"]').fill('none');
      const constructorSelection = await page.evaluate(() => buildRunPayload().pit_plan_selection);
      assert.equal(constructorSelection.constructor_id, 'team0');
      assert.equal(Object.hasOwn(constructorSelection, 'driver_id'), false);
      assert.deepEqual(constructorSelection.plans['<img src=x onerror=alert(1)>'], {
        S00: [{lap: 18, compound: 'hard'}], S01: [],
      });
      assert.deepEqual(await page.evaluate(() => [
        frozenPitPlanText({S00: [], S01: null, S02: [{lap: 6, compound: 'soft'}]}, ['S00', 'S01']),
        frozenPitPlanText({}, ['S00']),
        frozenPitPlanText({S01: [{lap: 6, compound: 'soft'}]}, ['S00']),
      ]), [
        'S00: No elective stops; S01: Automatic strategy',
        'S00: Automatic strategy',
        'S00: Automatic strategy',
      ], 'Frozen constructor plans should show only current target members');

      await page.locator('#pitRivalEditor summary').click();
      await page.locator('#pitRivalEnabled').check();
      const rivalRows = page.locator('#pitRivalScenarios > .pit-selection-candidate');
      assert.equal(await rivalRows.count(), 1);
      await rivalRows.first().locator('.pit-rival-name').fill('__proto__');
      await rivalRows.first().locator('[data-add-rival-driver]').click();
      const rivalOverride = rivalRows.first().locator('.pit-rival-override');
      assert.equal(await rivalOverride.locator('.pit-rival-driver option[value="S00"]').count(), 0);
      assert.equal(await rivalOverride.locator('.pit-rival-driver option[value="S01"]').count(), 0);
      await rivalOverride.locator('.pit-rival-driver').selectOption('S02');
      assert.equal(await rivalOverride.locator('.pit-rival-stops').isVisible(), false);
      let rivals = JSON.parse(await page.evaluate(() => JSON.stringify(buildRunPayload().pit_plan_selection.rival_scenarios)));
      assert.deepEqual(rivals.__proto__.pit_plans, {});
      await rivalOverride.locator('.pit-rival-mode').selectOption('automatic');
      rivals = JSON.parse(await page.evaluate(() => JSON.stringify(buildRunPayload().pit_plan_selection.rival_scenarios)));
      assert.equal(rivals.__proto__.pit_plans.S02, null);
      await rivalOverride.locator('.pit-rival-mode').selectOption('none');
      rivals = JSON.parse(await page.evaluate(() => JSON.stringify(buildRunPayload().pit_plan_selection.rival_scenarios)));
      assert.deepEqual(rivals.__proto__.pit_plans.S02, []);
      await rivalOverride.locator('.pit-rival-mode').selectOption('custom');
      await rivalOverride.locator('.pit-rival-stops').fill('18:hard');
      await page.locator('#addPitRivalScenario').click();
      await rivalRows.last().locator('.pit-rival-weight').fill('2');
      rivals = JSON.parse(await page.evaluate(() => JSON.stringify(buildRunPayload().pit_plan_selection.rival_scenarios)));
      assert.deepEqual(rivals.__proto__.pit_plans.S02, [{lap: 18, compound: 'hard'}]);
      assert.equal(rivals['Rival scenario 2'].weight, 2);
      assert((await page.locator('#pitPlanSelectionEstimate').innerText()).includes('410 trials'));
      await rivalRows.last().locator('.pit-rival-weight').fill('0');
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
      await rivalRows.last().locator('.pit-rival-weight').fill('2');
      await rivalRows.last().locator('.pit-rival-name').fill('__proto__');
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
      await rivalRows.last().locator('.pit-rival-name').fill('Rival scenario 2');
      await page.locator('#pitPlanSelectionTargetMode').selectOption('driver');
      await page.locator('#pitPlanSelectionTargetId').selectOption('S02');
      // Preserve the stale choice visibly, then reject it rather than silently reassigning it.
      assert.equal(await rivalOverride.locator('.pit-rival-driver').inputValue(), 'S02');
      assert.equal(await page.evaluate(() => buildPitRivalScenarios().ok), false);
      await page.locator('#pitPlanSelectionTargetId').selectOption('S00');
      assert.equal(await page.evaluate(() => buildPitRivalScenarios().ok), true);
      await page.locator('#pitRivalEnabled').uncheck();
      assert.equal(await page.locator('#pitRivalScenarios').isVisible(), false);
      await page.locator('#pitRivalEditor summary').click();
      await candidateRows.nth(1).locator('[data-member-id="S00"]').fill('18:hard');

      await page.locator('#pitPlanSelectionTargetMode').selectOption('driver');
      await page.locator('#pitPlanSelectionTargetId').selectOption('S00');
      await page.locator('#simCount').fill('100');
      await page.locator('#pitPlanTrainingTrials').fill('400');
      await page.locator('#pitPlanValidationTrials').fill('50');
      const exactBudget = await page.evaluate(() => buildRunPayload()?.pit_plan_selection);
      assert(exactBudget, 'A worst-case budget of exactly 1,000 per weather must be accepted');
      await page.locator('#pitPlanValidationTrials').fill('51');
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
      assert((await page.locator('#appStatus').innerText()).includes('above the 1,000 limit'));
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'pitPlanTrainingTrials');
      await page.locator('#pitPlanTrainingTrials').fill('0');
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'pitPlanTrainingTrials');
      await page.locator('#pitPlanTrainingTrials').fill('50');
      await page.locator('#pitPlanValidationTrials').fill('50');
      await page.locator('#simCount').fill('10');

      for (let count = 2; count < 10; count += 1) {
        await page.locator('#addPitPlanCandidateBtn').click();
      }
      assert.equal(await candidateRows.count(), 10);
      assert(await page.locator('#addPitPlanCandidateBtn').isDisabled());
      for (let count = 10; count > 2; count -= 1) {
        await candidateRows.last().locator('[data-remove-candidate]').click();
      }
      assert.equal(await candidateRows.count(), 2);

      await compareAutomaticInput.check();
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
      assert((await page.locator('#appStatus').innerText()).includes('cannot run together'));
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'pitPlanSelectionEnabled');
      await compareAutomaticInput.uncheck();
      await page.locator('#pitPlanSelectionEnabled').uncheck();
      await page.locator('#pitPlanSelectionEnabled').check();
      await page.locator('#pitPlanSelectionTargetMode').selectOption('driver');
      await page.locator('#pitPlanSelectionTargetId').selectOption('S00');
      const selectionCandidateRows = page.locator('#pitPlanSelectionCandidates .pit-selection-candidate');
      await selectionCandidateRows.nth(0).locator('.pit-selection-label').fill('__proto__');
      await selectionCandidateRows.nth(1).locator('.pit-selection-label').fill('<img src=x onerror=alert(1)>');
      await selectionCandidateRows.nth(1).locator('[data-member-id="S00"]').fill('18:hard');
      const requestShape = await page.evaluate(() => buildRunPayload().pit_plan_selection);
      assert.equal(requestShape.reference_label, '__proto__');
      assert.equal(requestShape.driver_id, 'S00');
      assert.deepEqual(requestShape.plans['<img src=x onerror=alert(1)>'], [
        {lap: 18, compound: 'hard'},
      ]);
      await page.locator('#pitPlanSelectionEnabled').uncheck();

      await compareAutomaticInput.check();
      await pitPlanInput.fill('');
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
      assert((await page.locator('#appStatus').innerText()).includes('at least one custom pit plan'));
      await pitPlanInput.fill('S00=18:hard,36:soft;S01=none');
      await page.locator('#simCount').fill('');
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
      assert((await page.locator('#appStatus').innerText()).includes('Enter a simulation count'));
      await page.locator('#simCount').fill('501');
      assert.equal(await page.evaluate(() => buildRunPayload()), null);
      assert((await page.locator('#appStatus').innerText()).includes('at most 500'));
      await page.locator('#simCount').fill('500');
    }
    const customPayload = await page.evaluate(() => {
      const payload = buildRunPayload();
      return {
        payload,
        pitPlansNullPrototype: Object.getPrototypeOf(payload?.pit_plans) === null,
      };
    });
    assert.equal(customPayload.payload.rng_policy, 'isolated_weather_v1');
    await rngPolicyInput.selectOption('isolated_weather_mechanical_v1');
    assert.equal(await page.evaluate(() => buildRunPayload().rng_policy),
      'isolated_weather_mechanical_v1');
    const invalidRngPolicy = await page.evaluate(() => {
      const select = document.getElementById('rngPolicySelect');
      select.value = 'unknown_policy';
      const payload = buildRunPayload();
      return {payload, message: document.getElementById('appStatus').innerText};
    });
    assert.equal(invalidRngPolicy.payload, null);
    assert(invalidRngPolicy.message.includes('valid random draw matching policy'));
    await rngPolicyInput.selectOption('isolated_weather_v1');
    if (offline) {
      assert.deepEqual(customPayload.payload.pit_plans, {
        S00: [{lap: 18, compound: 'hard'}, {lap: 36, compound: 'soft'}], S01: [],
      });
      assert.equal(customPayload.pitPlansNullPrototype, true);
    } else {
      assert.equal(Object.hasOwn(customPayload.payload, 'pit_plans'), false);
    }
    if (offline) {
      assert.equal(customPayload.payload.compare_automatic, true);
      await compareAutomaticInput.uncheck();
      await page.locator('#simCount').fill('1000');
      const normalPayload = await page.evaluate(() => buildRunPayload());
      assert.equal(normalPayload.compare_automatic, false);
      assert.equal(normalPayload.simulations, 1000);
      await page.locator('#simCount').fill('10');
    } else {
      await compareAutomaticInput.uncheck();
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
    assert.equal(response.request().postDataJSON().rng_policy, 'isolated_weather_v1');
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
    assert.equal(payload.request.rng_policy, 'isolated_weather_v1');
    assert.equal(Object.values(payload.scenarios)[0].simulation_inputs.rng_policy,
      'isolated_weather_v1');
    // The matrix initially shows aggregate top contenders, which need not
    // include the winner of the representative individual race.
    const driverId = Object.values(payload.scenarios)[0].win_probabilities[0][0];
    await page.waitForFunction(() => !runInProgress);
    assert((await page.locator('#panel-race').textContent()).includes('Lap-aware model (experimental)'));
    assert((await page.locator('#panel-race').textContent()).includes('Fixed rainfall; surface wetness still evolves'));
    assert((await page.locator('#panel-race').textContent())
      .includes('Match random draws: Weather (default)'));
    await rngPolicyInput.selectOption('isolated_weather_mechanical_v1');
    await page.evaluate(() => renderRace());
    assert((await page.locator('#panel-race').textContent())
      .includes('Match random draws: Weather (default)'),
    'Changing the current RNG control must not relabel saved results');
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
    const pitPlanAggregateDraft = await pitPlanInput.inputValue();
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      window.savedPitPlanAggregateState = {
        hadStatistics: Object.hasOwn(scenario, 'pit_plan_statistics'),
        statistics: scenario.pit_plan_statistics,
        hadInputs: Object.hasOwn(scenario, 'simulation_inputs'),
        inputs: scenario.simulation_inputs,
        request: simResults.request,
      };
      scenario.simulation_inputs = {...(scenario.simulation_inputs || {}), pit_plans: {
        S00: [{lap: 18, compound: 'hard'}], S01: [],
      }};
      scenario.pit_plan_statistics = {
        status: 'available', recorded_trials: 10,
        drivers: [
          {driver_id: '<img src=x onerror="window.planInjected=true">', no_elective_stops: false,
            valid_histories: 7, missing_histories: 2, invalid_histories: 1,
            instructions: [
              {lap: 18, compound: 'hard', executed: 3, overridden: 1, skipped: 2, not_reached: 1},
              {lap: 36, compound: '<svg onload=alert(1)>', executed: '3', overridden: -1,
                skipped: 11, not_reached: null},
              {lap: 48, compound: 'soft', executed: 1, overridden: 1, skipped: 1, not_reached: 1},
              {lap: 56, compound: 'hard', executed: {valueOf: 'x', toString: 'x'},
                overridden: 1, skipped: 1, not_reached: 4},
            ]},
          {driver_id: 'S01', no_elective_stops: true, valid_histories: 10,
            missing_histories: 0, invalid_histories: 0, instructions: []},
          {driver_id: 'S02', no_elective_stops: false,
            valid_histories: {valueOf: 'x', toString: 'x'}, missing_histories: 0, invalid_histories: 10,
            instructions: [{lap: 2, compound: 'soft', executed: 1, overridden: 1,
              skipped: 1, not_reached: 1}]},
        ],
      };
      renderStats();
    });
    const pitPlanStatistics = page.locator('#pitPlanStatistics');
    const hostilePlanDriver = pitPlanStatistics.locator('details').first();
    assert((await hostilePlanDriver.locator('summary').innerText())
      .includes('Coverage: 7 valid, 2 missing, 1 invalid of 10 recorded trials.'));
    assert.equal(await pitPlanStatistics.locator('img, svg').count(), 0);
    await hostilePlanDriver.locator('summary').click();
    assert.equal(await hostilePlanDriver.locator('tbody tr').count(), 4);
    assert.deepEqual(await hostilePlanDriver.locator('tbody tr').first().locator('td').allTextContents(),
      ['Hard', '3', '1', '2', '1']);
    assert((await hostilePlanDriver.innerText()).includes('<Svg Onload=Alert(1)>'));
    assert.deepEqual(await hostilePlanDriver.locator('tbody tr').nth(1).locator('td').allTextContents(),
      ['<Svg Onload=Alert(1)>', 'Not recorded', 'Not recorded', 'Not recorded', 'Not recorded']);
    assert.deepEqual(await hostilePlanDriver.locator('tbody tr').nth(2).locator('td').allTextContents(),
      ['Soft', 'Not recorded', 'Not recorded', 'Not recorded', 'Not recorded']);
    assert.deepEqual(await hostilePlanDriver.locator('tbody tr').nth(3).locator('td').allTextContents(),
      ['Hard', 'Not recorded', 'Not recorded', 'Not recorded', 'Not recorded']);
    const noElectiveDriver = pitPlanStatistics.locator('details').nth(1);
    assert((await noElectiveDriver.locator('summary').innerText()).includes('No elective instructions'));
    assert((await noElectiveDriver.locator('summary').innerText())
      .includes('Coverage: 10 valid, 0 missing, 0 invalid of 10 recorded trials.'));
    await noElectiveDriver.locator('summary').click();
    assert((await noElectiveDriver.innerText()).includes('compulsory stops may still occur'));
    assert((await pitPlanStatistics.locator('details').nth(2).locator('summary').innerText())
      .includes('Coverage: Not recorded valid, Not recorded missing, Not recorded invalid of 10 recorded trials.'));
    await pitPlanInput.fill('S00=2:medium');
    await page.evaluate(() => renderStats());
    assert((await pitPlanStatistics.innerText()).includes('Coverage: 7 valid, 2 missing, 1 invalid of 10 recorded trials.'));
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      delete scenario.pit_plan_statistics;
      scenario.simulation_inputs = {...(scenario.simulation_inputs || {}), pit_plans: null};
      renderStats();
    });
    assert((await pitPlanStatistics.innerText()).includes('used automatic strategy'));
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      scenario.pit_plan_statistics = {status: 'not_recorded'};
      scenario.simulation_inputs = {...(scenario.simulation_inputs || {}), pit_plans: {
        S00: [{lap: 18, compound: 'hard'}],
      }};
      renderStats();
    });
    assert((await pitPlanStatistics.innerText()).includes('although this saved result contains custom plan input'));
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      delete scenario.pit_plan_statistics;
      delete scenario.simulation_inputs.pit_plans;
      simResults.request = {};
      renderStats();
    });
    assert((await pitPlanStatistics.innerText()).includes('This can occur with legacy results'));
    await page.evaluate(() => {
      getScenarioEntry().data.pit_plan_statistics = {status: 'invalid'};
      renderStats();
    });
    assert((await pitPlanStatistics.innerText()).includes('invalid pit-plan summary context'));
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      const saved = window.savedPitPlanAggregateState;
      if (saved.hadStatistics) scenario.pit_plan_statistics = saved.statistics;
      else delete scenario.pit_plan_statistics;
      if (saved.hadInputs) scenario.simulation_inputs = saved.inputs;
      else delete scenario.simulation_inputs;
      simResults.request = saved.request;
      delete window.savedPitPlanAggregateState;
      renderStats();
    });
    await pitPlanInput.fill(pitPlanAggregateDraft);
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
    const overtakePanel = page.locator('#overtakingStatistics');
    const overtakeText = await overtakePanel.innerText();
    assert(overtakeText.includes('20 / 10 / 10'), 'Fixture must show all-field attempts/outcomes');
    assert(overtakeText.includes('219 recorded / 1 missing available driver-race rows'));
    assert(overtakeText.includes('50.0%'));
    assert(overtakeText.includes('S00'));
    assert(overtakeText.includes('0 / 0 / 0'), 'Recorded zero attempts must stay explicit');
    assert(overtakeText.includes('Not defined (0 attempts)'));
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      window.savedOvertakingStatistics = scenario.overtaking_statistics;
      scenario.overtaking_statistics = {
        overall: window.savedOvertakingStatistics.overall,
        drivers: {...window.savedOvertakingStatistics.drivers,
          '<img src=x onerror=globalThis.overtakeInjected=true>':
            window.savedOvertakingStatistics.drivers.S00},
      };
      renderStats();
    });
    assert.equal(await overtakePanel.locator('img').count(), 0);
    assert.equal(await page.evaluate(() => Boolean(globalThis.overtakeInjected)), false);
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      scenario.overtaking_statistics = window.savedOvertakingStatistics;
      delete window.savedOvertakingStatistics;
      renderStats();
    });
    assert((await overtakePanel.innerText()).includes('Full field'));
    await page.evaluate(() => {
      const scenario = getScenarioEntry().data;
      window.savedOvertakingStatistics = scenario.overtaking_statistics;
      delete scenario.overtaking_statistics;
      renderStats();
    });
    assert((await overtakePanel.innerText()).includes('not recorded for this run'));
    await page.evaluate(() => {
      getScenarioEntry().data.overtaking_statistics = window.savedOvertakingStatistics;
      delete window.savedOvertakingStatistics;
      renderStats();
    });
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
          if (width === 1440) await page.setViewportSize({width, height: 2600});
          if (width === 390) {
            const metrics = await overtakePanel.locator('.stats-table-wrap').evaluate(node => ({
              clientWidth: node.clientWidth,
              scrollWidth: node.scrollWidth,
            }));
            assert(metrics.scrollWidth > metrics.clientWidth,
              'Narrow overtake statistics should scroll horizontally without page overflow');
          }
          await overtakePanel.evaluate(card => card.scrollIntoView({block: 'center'}));
          await overtakePanel.screenshot({
            path: path.join(process.env.F1SIM_SCREENSHOTS, `overtaking-${width}.png`),
          });
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
        assert.equal(saved.strategy_comparison_reports, undefined);
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
    assert(await page.locator('#downloadStrategyReportBtn').isDisabled(),
      'Ordinary scenario runs do not have a paired strategy report');
    if (offline) {
      let savedLightRainStrategyReport = '';
      await pitPlanInput.fill('S00=18:hard;S01=none');
      await compareAutomaticInput.check();
      await page.locator('#simCount').fill('10');
      const comparisonResponsePromise = page.waitForResponse(response => response.url().endsWith('/api/run'));
      await page.locator('#btnRun').click();
      const comparisonResponse = await comparisonResponsePromise;
      assert.equal(comparisonResponse.status(), 200);
      assert.equal(comparisonResponse.request().postDataJSON().compare_automatic, true);
      assert.equal(comparisonResponse.request().postDataJSON().simulations, 10);
      assert.equal(comparisonResponse.request().postDataJSON().rng_policy,
        'isolated_weather_mechanical_v1');
      const comparisonPayload = await comparisonResponse.json();
      assert.equal(comparisonPayload.request.rng_policy, 'isolated_weather_mechanical_v1');
      assert.equal(comparisonPayload.automatic_reference.request.rng_policy,
        'isolated_weather_mechanical_v1');
      for (const scenario of Object.values(comparisonPayload.scenarios)) {
        assert.equal(scenario.simulation_inputs.rng_policy, 'isolated_weather_mechanical_v1');
      }
      for (const scenario of Object.values(comparisonPayload.automatic_reference.scenarios)) {
        assert.equal(scenario.simulation_inputs.rng_policy, 'isolated_weather_mechanical_v1');
      }
      await page.waitForFunction(() => !runInProgress);
      await page.locator('#tab-race').click();
      assert(await page.locator('#strategyComparisonPanel').isVisible());
      assert((await page.locator('#panel-race').textContent())
        .includes('Match random draws: Weather and mechanical checks'));
      const comparisonText = await page.locator('#strategyComparisonPanel').innerText();
      for (const label of [
        'Custom plan minus automatic strategy', '+1.250 pts', '-3.333 pp',
        'Valid pairs: 3 / 10', 'Valid pairs: 2 / 10',
        'Paid-stop cost components',
        '20 completed trials across both alternatives (10 custom; 10 automatic).',
      ]) {
        assert(comparisonText.includes(label), `Missing strategy comparison detail: ${label}`);
      }
      const team0Stats = fixture.comparison_payload.strategy_comparisons.dry
        .variants.custom.constructor_statistics.team0;
      assert.deepEqual(team0Stats.driver_ids, ['S00', 'S01']);
      const constructorRegion = page.getByRole('region', {
        name: 'Constructor paired points', exact: true,
      });
      const constructorText = (await constructorRegion.innerText()).toLowerCase();
      for (const label of [
        'Synthetic team0', '(team0)', 'S00, S01',
        `Complete pairs: ${team0Stats.paired_races}; excluded pairs: ${team0Stats.excluded_pairs}`,
        `${team0Stats.reference_mean_points.toFixed(3)} → ${team0Stats.variant_mean_points.toFixed(3)}`,
        `${team0Stats.mean_points_difference < 0 ? '' : '+'}${team0Stats.mean_points_difference.toFixed(3)} pts`,
        `${team0Stats.more_points_races} / ${team0Stats.equal_points_races} / ${team0Stats.fewer_points_races}`,
      ]) {
        assert(constructorText.includes(label.toLowerCase()),
          `Missing constructor comparison detail: ${label}`);
      }
      assert(comparisonText.toLowerCase().includes('points are summed within each seed'));
      await page.locator('#strategyComparisonCosts summary').click();
      const costText = await page.locator('#strategyComparisonCosts').innerText();
      assert(costText.includes('zero is a recorded zero'));
      assert(costText.includes('Unavailable (one valid pair)'));
      for (const label of ['Pit-lane loss', 'Service time', 'Queue time']) {
        assert(costText.includes(label), `Missing strategy cost detail: ${label}`);
      }
      for (const width of [320, 390, 1440]) {
        await page.setViewportSize({width, height: 1100});
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `Strategy comparison overflows at ${width}px`);
      }
      const comparisonRegion = page.locator('#strategyComparisonPanel [role="region"]').first();
      await comparisonRegion.focus();
      assert(await comparisonRegion.evaluate(node => document.activeElement === node));

      await page.locator('#tab-scenarios').click();
      assert(await page.locator('#downloadAutomaticReferenceBtn').isEnabled());
      const referenceDownloadPromise = page.waitForEvent('download');
      await page.locator('#downloadAutomaticReferenceBtn').click();
      const referenceDownload = await referenceDownloadPromise;
      assert(referenceDownload.suggestedFilename().includes('automatic_strategy_reference'));
      const referenceBundle = JSON.parse(readFileSync(await referenceDownload.path(), 'utf8'));
      assert.deepEqual(referenceBundle.request.pit_plans, {});
      assert.equal(referenceBundle.request.compare_automatic, false);
      assert.equal(referenceBundle.request.simulations, 10);
      assert.equal(referenceBundle.request.race_engine, 'chronological');
      assert.equal(referenceBundle.request.rng_policy, 'isolated_weather_mechanical_v1');
      assert(Object.values(referenceBundle.scenarios).every(scenario =>
        scenario.simulation_inputs.rng_policy === 'isolated_weather_mechanical_v1'));
      assert.equal(referenceBundle.year, 2026);
      assert(referenceBundle.track && referenceBundle.ratings && referenceBundle.provenance);
      assert(referenceBundle.scenarios && Object.keys(referenceBundle.scenarios).length === 3);

      const savedStrategyReports = await page.evaluate(() => simResults.strategy_comparison_reports);
      assert.deepEqual(Object.keys(savedStrategyReports), ['dry', 'light_rain', 'heavy_rain']);
      const comparisonJsonPromise = page.waitForEvent('download');
      await page.locator('#downloadScenarioJsonBtn').click();
      const comparisonJsonDownload = await comparisonJsonPromise;
      const comparisonJson = JSON.parse(readFileSync(await comparisonJsonDownload.path(), 'utf8'));
      assert.equal(comparisonJson.strategy_comparison_reports, undefined,
        'Normal JSON downloads must omit generated HTML reports');
      assert(comparisonJson.strategy_comparisons, 'Paired statistics remain in JSON');
      assert(comparisonJson.automatic_reference, 'The automatic reference remains in JSON');
      assert(Object.values(comparisonJson.scenarios).every(scenario =>
        scenario.simulation_inputs.rng_policy === 'isolated_weather_mechanical_v1'));

      await page.locator('#weatherSelect').selectOption('LIGHT_RAIN');
      assert(await page.locator('#downloadStrategyReportBtn').isEnabled(),
        'A paired report is available for the selected saved weather');
      const lightRainReportPromise = page.waitForEvent('download');
      await page.locator('#downloadStrategyReportBtn').click();
      const lightRainReportDownload = await lightRainReportPromise;
      assert.equal(lightRainReportDownload.suggestedFilename(), 'strategy_comparison_light_rain.html');
      savedLightRainStrategyReport = readFileSync(await lightRainReportDownload.path(), 'utf8');
      assert.equal(savedLightRainStrategyReport, savedStrategyReports.light_rain);
      assert(savedLightRainStrategyReport.includes('Paired changes compare each choice with automatic'));
      assert(savedLightRainStrategyReport.includes('<strong>custom</strong>'));
      assert(savedLightRainStrategyReport.includes('Paired coverage:'));

      await page.locator('#weatherSelect').selectOption('CLOUDY');
      assert(await page.locator('#downloadStrategyReportBtn').isDisabled(),
        'Reports are disabled when the focused weather is absent from the saved run');
      await page.locator('#weatherSelect').selectOption('LIGHT_RAIN');
      await page.evaluate(() => {
        window.savedFocusedStrategyReport = simResults.strategy_comparison_reports.light_rain;
        delete simResults.strategy_comparison_reports.light_rain;
        updateStrategyReportButton(simResults);
      });
      assert(await page.locator('#downloadStrategyReportBtn').isDisabled(),
        'Legacy comparison results without the focused report stay disabled');
      await page.evaluate(() => {
        simResults.strategy_comparison_reports.light_rain = window.savedFocusedStrategyReport;
        delete window.savedFocusedStrategyReport;
        updateStrategyReportButton(simResults);
      });
      assert(await page.locator('#downloadStrategyReportBtn').isEnabled());
      await page.locator('#weatherSelect').selectOption('DRY');
      assert(await page.locator('#downloadStrategyReportBtn').isEnabled());
      await page.locator('#presetChaosBtn').click();
      assert.equal(await page.locator('#weatherSelect').inputValue(), 'CLOUDY');
      assert(await page.locator('#downloadStrategyReportBtn').isDisabled(),
        'Preset focus changes must disable the report when the focused weather has no saved result');
      await page.locator('#presetMixedBtn').click();
      assert.equal(await page.locator('#weatherSelect').inputValue(), 'DRY');
      assert(await page.locator('#downloadStrategyReportBtn').isEnabled(),
        'Preset focus changes back to a saved weather must enable its report');
      await page.locator('#tab-race').click();

      const elapsedRow = page.locator('#strategyComparisonPanel tr').filter({hasText: 'Elapsed time (same-distance finishes)'}).first();
      assert((await elapsedRow.innerText()).includes('-2.500 s'));
      assert((await elapsedRow.innerText()).includes('Valid pairs: 2 / 10'));
      const warmupChecks = await page.evaluate(() => {
        const input = document.getElementById('tireWarmupInput');
        input.value = 'soft=0.5,medium=1';
        const valid = parseTireWarmupInput();
        const payload = buildRunPayload();
        input.value = 'soft=1,soft=2';
        const duplicate = parseTireWarmupInput();
        input.value = 'soft=Infinity';
        const invalid = parseTireWarmupInput();
        input.value = '';
        const saved = renderWarmupSnapshot({simulation_inputs: {
          tire_warmup: {soft: 0.5}, tire_warmup_policy: 'post_fit_first_lap_v1',
        }});
        const malformed = renderWarmupSnapshot({simulation_inputs: {
          tire_warmup: {'<img src=x onerror=alert(1)>': 1},
          tire_warmup_policy: 'post_fit_first_lap_v1',
        }});
        return {profile: valid.profile, payload: payload?.tire_warmup,
          duplicate: duplicate.ok, invalid: invalid.ok, saved, malformed};
      });
      assert.deepEqual(warmupChecks.profile, {soft: 0.5, medium: 1});
      assert.deepEqual(warmupChecks.payload, {soft: 0.5, medium: 1});
      assert.equal(warmupChecks.duplicate, false);
      assert.equal(warmupChecks.invalid, false);
      assert(warmupChecks.saved.includes('soft=0.5 s'));
      assert(warmupChecks.saved.includes('not calibrated physics'));
      assert(!warmupChecks.malformed.includes('<img'));
      const savedComparisonVariants = await page.evaluate(() => simResults.strategy_comparisons);
      const unavailableHtml = await page.evaluate(() => {
        simResults.strategy_comparisons.dry = {
          variants: {custom: {status: 'unavailable', reason: '<img src=x onerror=alert(1)>'}},
        };
        renderRace();
        return document.getElementById('strategyComparisonPanel').innerHTML;
      });
      assert(unavailableHtml.includes('Comparison unavailable'));
      assert(unavailableHtml.includes('&lt;img') && !unavailableHtml.includes('<img'));
      const nullAndZeroHtml = await page.evaluate(() => {
        simResults.strategy_comparisons.dry = {
          variants: {custom: {status: 'paired', available_seed_pairs: 10,
            constructor_statistics: {'<svg onload=alert(1)>': {
              team_name: '<img src=x onerror=alert(1)>', driver_ids: ['<svg onload=alert(1)>', 'S01'],
              paired_races: 0, excluded_pairs: 10,
              reference_mean_points: 0, variant_mean_points: null,
              mean_points_difference: 0, points_difference_standard_error: null,
              more_points_races: 0, equal_points_races: 0, fewer_points_races: 0,
            }},
            driver_statistics: {'<svg onload=alert(1)>': {
              paired_races: 0, excluded_pairs: 10,
              mean_points_difference: null, points_difference_standard_error: null,
              dnf_rate_difference_percentage_points: 0,
              dnf_rate_difference_standard_error_percentage_points: null,
              completed_distance: {paired_races: 1, excluded_pairs: 9,
                mean_laps_difference: 0, laps_difference_standard_error: null},
              paid_stop_costs: {paired_races: 0, excluded_pairs: 10},
            }}}},
        };
        return renderStrategyComparison('dry');
      });
      assert(nullAndZeroHtml.includes('Not recorded'));
      assert(nullAndZeroHtml.includes('Elapsed time (same-distance finishes)'));
      assert(nullAndZeroHtml.includes('&lt;img') && !nullAndZeroHtml.includes('<img'));
      assert(nullAndZeroHtml.includes('&lt;svg') && !nullAndZeroHtml.includes('<svg'));
      assert(nullAndZeroHtml.includes('Complete pairs: 0; excluded pairs: 10'));
      assert(nullAndZeroHtml.includes('+0.000 pts'));
      assert(nullAndZeroHtml.includes('0 / 0 / 0'));

      assert(nullAndZeroHtml.includes('+0.000 pp'));
      assert(nullAndZeroHtml.includes('Valid pairs: 0 / 10'));
      assert(nullAndZeroHtml.includes('&lt;svg') && !nullAndZeroHtml.includes('<svg'));
      await page.evaluate(saved => {
        simResults.strategy_comparisons = saved;
        renderRace();
      }, savedComparisonVariants);
      const savedPanelText = await page.locator('#strategyComparisonPanel').innerText();
      await rngPolicyInput.selectOption('isolated_weather_v1');
      await pitPlanInput.fill('S00=2:medium');
      await page.locator('#simCount').fill('500');
      await page.evaluate(() => renderRace());
      assert.equal(await page.locator('#strategyComparisonPanel').innerText(), savedPanelText,
        'Comparison results must come from the saved response, not edited controls');
      assert((await page.locator('#panel-race').textContent())
        .includes('Match random draws: Weather and mechanical checks'),
      'Saved RNG policy must be rendered from the result after the form changes');
      await page.locator('#tab-scenarios').click();
      await page.locator('#weatherSelect').selectOption('LIGHT_RAIN');
      const editedFormReportPromise = page.waitForEvent('download');
      await page.locator('#downloadStrategyReportBtn').click();
      const editedFormReport = await editedFormReportPromise;
      assert.equal(readFileSync(await editedFormReport.path(), 'utf8'), savedLightRainStrategyReport,
        'Strategy downloads must use the saved response after form edits');
      await page.locator('#weatherSelect').selectOption('DRY');
      await page.setViewportSize({width: 1440, height: 900});
      await compareAutomaticInput.uncheck();
      await pitPlanInput.fill('S00=18:hard,36:soft;S01=none');
      await page.locator('#simCount').fill('10');
      await page.locator('#tab-scenarios').click();

      await page.locator('#pitPlanSelectionEnabled').check();
      assert.equal(await page.locator('#tab-race').getAttribute('aria-selected'), 'true',
        'Enabling candidate selection from Scenario Lab should reveal its Race Results editor');
      await page.locator('#pitPlanSelectionPanel').waitFor({state: 'visible'});
      await page.waitForFunction(() => document.activeElement?.id === 'pitPlanSelectionTargetMode');
      await page.locator('#pitPlanSelectionTargetMode').selectOption('driver');
      await page.locator('#pitPlanSelectionTargetId').selectOption('S00');
      const selectionCandidateRows = page.locator('#pitPlanSelectionCandidates .pit-selection-candidate');
      await selectionCandidateRows.nth(0).locator('.pit-selection-label').fill('__proto__');
      await selectionCandidateRows.nth(1).locator('.pit-selection-label').fill('<img src=x onerror=alert(1)>');
      await selectionCandidateRows.nth(1).locator('[data-member-id="S00"]').fill('18:hard');
      await page.locator('#pitPlanTrainingTrials').fill('50');
      await page.locator('#pitPlanValidationTrials').fill('50');
      const selectionResponsePromise = page.waitForResponse(response =>
        response.url().endsWith('/api/run'));
      await page.locator('#btnRun').click();
      const selectionResponse = await selectionResponsePromise;
      assert.equal(selectionResponse.status(), 200);
      const selectionRequest = selectionResponse.request().postDataJSON();
      assert.deepEqual(selectionRequest.pit_plan_selection.plans, Object.fromEntries([
        ['__proto__', null], ['<img src=x onerror=alert(1)>', [{lap: 18, compound: 'hard'}]],
      ]));
      assert.deepEqual({
        reference_label: selectionRequest.pit_plan_selection.reference_label,
        driver_id: selectionRequest.pit_plan_selection.driver_id,
        training_simulations: selectionRequest.pit_plan_selection.training_simulations,
        validation_simulations: selectionRequest.pit_plan_selection.validation_simulations,
      }, {
        reference_label: '__proto__', driver_id: 'S00',
        training_simulations: 50, validation_simulations: 50,
      });
      assert.equal(selectionRequest.compare_automatic, false);
      await page.waitForFunction(() => !runInProgress);
      await page.locator('#tab-scenarios').click();
      await page.locator('#pitPlanSelectionResults').waitFor({state: 'visible'});
      let selectionText = await page.locator('#pitPlanSelectionResults').innerText();
      let selectionLabels = selectionText.toLowerCase();
      assert(selectionLabels.includes('training choice') && selectionLabels.includes('fresh validation'));
      assert(selectionLabels.includes('selected − reference') && selectionLabels.includes('monte carlo se'));
      assert(selectionLabels.includes('more points') && selectionLabels.includes('fewer points'));
      assert(selectionLabels.includes('selected and frozen') && selectionLabels.includes('fixed reference'));
      assert(selectionText.includes('18:hard'));
      assert(selectionText.includes('S00: Automatic strategy'),
        'A non-target custom plan must not replace the target driver’s automatic plan');
      assert(!selectionText.includes('S01: 2:soft'),
        'Frozen plan display must exclude unrelated members');
      assert(selectionText.includes('<img src=x onerror=alert(1)>'),
        'Hostile candidate text should remain visible as text');
      const selectionMarkup = await page.locator('#pitPlanSelectionResults').innerHTML();
      assert(selectionMarkup.includes('&lt;img src=x onerror=alert(1)&gt;'));
      assert(!selectionMarkup.includes('<img src=x onerror=alert(1)>'));
      assert.equal(await page.locator('#pitPlanSelectionResults img').count(), 0);
      assert.equal(await page.locator('#pitPlanSelectionResults [aria-label="Training scores for Dry weather"] thead th').count(), 3,
        'Ordinary training evidence must retain its three-column table');
      assert.equal(await page.locator('#pitPlanSelectionResults [role="region"]').count() >= 3, true);

      const selectionFormState = await page.evaluate(() => ({
        savedSelected: simResults.strategy_selections.dry.selection.selected_label,
        savedPlans: simResults.strategy_selections.dry.plans,
      }));
      await page.locator('#tab-race').click();
      await selectionCandidateRows.nth(0).locator('.pit-selection-label').fill('edited reference');
      await selectionCandidateRows.nth(1).locator('.pit-selection-label').fill('edited alternative');
      await page.evaluate(() => renderPitPlanSelectionResults(simResults));
      await page.locator('#tab-scenarios').click();
      selectionText = await page.locator('#pitPlanSelectionResults').innerText();
      assert(selectionText.includes('<img src=x onerror=alert(1)>'));
      assert(!selectionText.includes('edited alternative'),
        'Editing candidate controls must not rewrite the saved selection result');
      assert.deepEqual(await page.evaluate(() => ({
        selected: simResults.strategy_selections.dry.selection.selected_label,
        plans: simResults.strategy_selections.dry.plans,
      })), {
        selected: selectionFormState.savedSelected,
        plans: selectionFormState.savedPlans,
      });

      const expectedEvidence = JSON.parse(await page.evaluate(() => JSON.stringify({
        selection: simResults.strategy_selections.dry.selection,
        plans: simResults.strategy_selections.dry.plans,
        source: simResults.strategy_selections.dry.source,
      })));
      const evidenceDownloadPromise = page.waitForEvent('download');
      await page.locator('#downloadPitSelectionEvidenceBtn').click();
      const evidenceDownload = await evidenceDownloadPromise;
      assert.equal(evidenceDownload.suggestedFilename(), 'pit_plan_selection_evidence_dry.json');
      const evidenceJson = JSON.parse(readFileSync(await evidenceDownload.path(), 'utf8'));
      assert.deepEqual(evidenceJson, expectedEvidence);
      assert.equal(evidenceJson.validation_report_html, undefined);
      assert.deepEqual(evidenceJson.plans.__proto__, {S01: [{lap: 2, compound: 'soft'}]},
        'Replay evidence should retain non-target plans even though the display projects them out');
      // Chromium throttles rapid download bursts; keep each export a separate, spaced user action.
      await page.waitForTimeout(1100);

      for (const phase of ['training', 'validation']) {
        const expectedPhase = JSON.parse(await page.evaluate(phaseName =>
          JSON.stringify(simResults.strategy_selections.dry[phaseName]), phase));
        const phaseDownloadPromise = page.waitForEvent('download');
        await page.locator(`#downloadPitSelection${phase === 'training' ? 'Training' : 'Validation'}Btn`).click();
        const phaseDownload = await phaseDownloadPromise;
        assert.equal(phaseDownload.suggestedFilename(), `pit_plan_selection_${phase}_dry.json`);
        assert.deepEqual(JSON.parse(readFileSync(await phaseDownload.path(), 'utf8')), expectedPhase);
        await page.waitForTimeout(1100);
      }
      const validationReport = await page.evaluate(() =>
        simResults.strategy_selections.dry.validation_report_html);
      const validationReportState = await page.evaluate(() => {
        const button = document.getElementById('downloadPitSelectionReportBtn');
        const focused = pitPlanSelectionForFocus(simResults);
        return {
          disabled: button.disabled,
          reportType: typeof focused?.entry?.validation_report_html,
          reportLength: focused?.entry?.validation_report_html?.length || 0,
          scenarioName: focused?.scenarioName,
        };
      });
      assert.deepEqual(validationReportState, {
        disabled: false, reportType: 'string', reportLength: validationReport.length,
        scenarioName: 'dry',
      });
      const validationReportDownloadPromise = page.waitForEvent('download');
      await page.locator('#downloadPitSelectionReportBtn').click();
      const validationReportDownload = await validationReportDownloadPromise;
      assert.equal(validationReportDownload.suggestedFilename(), 'pit_plan_selection_validation_dry.html');
      assert.equal(readFileSync(await validationReportDownload.path(), 'utf8'), validationReport);
      await page.waitForTimeout(1100);

      const selectionJsonDownloadPromise = page.waitForEvent('download');
      await page.locator('#downloadScenarioJsonBtn').click();
      const selectionJsonDownload = await selectionJsonDownloadPromise;
      const selectionJson = JSON.parse(readFileSync(await selectionJsonDownload.path(), 'utf8'));
      assert(selectionJson.strategy_selections.dry);
      assert.equal(selectionJson.strategy_selections.dry.validation_report_html, undefined,
        'Normal result JSON must omit nested validation HTML');
      assert(Object.hasOwn(selectionJson.strategy_selections.dry.plans, '__proto__'));

      await page.locator('#weatherSelect').selectOption('LIGHT_RAIN');
      selectionText = await page.locator('#pitPlanSelectionResults').innerText();
      assert(selectionText.toLowerCase().includes('light rain'));
      assert(selectionText.includes('+2.500 pts'),
        'Weather focus must render the frozen result entry for that weather');
      await page.locator('#weatherSelect').selectOption('DRY');
      selectionText = await page.locator('#pitPlanSelectionResults').innerText();
      assert(selectionText.includes('-1.000 pts'));

      const originalDrySelection = await page.evaluate(() =>
        JSON.stringify(simResults.strategy_selections.dry.selection));
      const identityText = await page.evaluate(() => {
        const selection = simResults.strategy_selections.dry.selection;
        selection.selected_label = selection.reference_label;
        selection.validation_status = 'no_change';
        renderPitPlanSelectionResults(simResults);
        return document.getElementById('pitPlanSelectionResults').innerText;
      });
      const identityLabels = identityText.toLowerCase();
      assert(identityLabels.includes('identity comparison'));
      assert(identityLabels.includes('no independent alternative was evaluated'));
      assert(identityLabels.includes('or outcome profile is independently estimated'));
      assert(!identityLabels.includes('0.000 pts'));
      await page.evaluate(serialized => {
        simResults.strategy_selections.dry.selection = JSON.parse(serialized);
        renderPitPlanSelectionResults(simResults);
      }, originalDrySelection);

      for (const width of [390, 1440]) {
        await page.setViewportSize({width, height: 1000});
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `Held-out selection overflows the page at ${width}px`);
        const region = page.locator('#pitPlanSelectionResults [role="region"]').first();
        await region.focus();
        assert(await region.evaluate(node => document.activeElement === node));
        if (process.env.F1SIM_SCREENSHOTS) {
          await page.screenshot({
            path: path.join(process.env.F1SIM_SCREENSHOTS, `pit-plan-selection-${width}.png`),
            fullPage: true,
          });
        }
      }

      await page.locator('#tab-race').click();
      await page.locator('#pitRivalEditor').evaluate(node => { node.open = true; });
      await page.locator('#pitRivalEnabled').check();
      const weightedRows = page.locator('#pitRivalScenarios > .pit-selection-candidate');
      await weightedRows.first().locator('.pit-rival-name').fill('__proto__');
      await weightedRows.last().locator('.pit-rival-name').fill('<img src=x onerror=alert(1)>');
      await weightedRows.last().locator('.pit-rival-weight').fill('2');
      await page.locator('#btnRun').click();
      await page.waitForFunction(() => !runInProgress &&
        simResults?.strategy_selections?.dry?.selection?.rival_scenarios?.length === 2);
      await page.locator('#tab-scenarios').click();
      const weightedResult = page.locator('#pitPlanSelectionResults');
      assert((await weightedResult.innerText()).includes('Weighted training choice'));
      assert((await weightedResult.innerText()).includes('Weighted fresh validation'));
      assert((await weightedResult.innerText()).includes('not calibrated probabilities'));
      assert((await weightedResult.innerText()).includes('0.333333'));
      assert((await weightedResult.innerText()).includes('Unique highest weighted training score.'));
      const weightedTraining = weightedResult.locator('[aria-label="Training scores for Dry weather"]');
      assert.equal(await weightedTraining.locator('thead th').count(), 4);
      assert((await weightedResult.innerText()).includes('Displayed means are rounded'));
      assert.equal(await weightedResult.locator('img').count(), 0);
      await weightedResult.locator('details').first().locator('summary').click();
      assert((await weightedResult.innerText()).includes('per-scenario evidence'));
      assert((await weightedResult.innerText()).includes('Fresh validation: selected minus reference'));
      const nativeScoreRows = await page.evaluate(() =>
        simResults.strategy_selections.dry.selection.training_scenario_score_tables.__proto__.scores);
      const expandedScores = weightedResult.locator('details').first()
        .locator('[aria-label="Per-scenario training scores"] tbody tr');
      assert.equal(await expandedScores.count(), nativeScoreRows.length,
        'Expanded native per-scenario evidence must contain every candidate score');
      for (let index = 0; index < nativeScoreRows.length; index++) {
        const row = nativeScoreRows[index];
        assert.deepEqual(await expandedScores.nth(index).locator('th, td').allTextContents(),
          [row.label, `${row.mean_points.toFixed(3)} pts`, String(row.trials)]);
      }
      assert(!(await weightedResult.innerText()).includes('Training scores not recorded.'));
      const frozenWeighted = await page.evaluate(() => JSON.stringify(simResults.strategy_selections.dry));
      // Use supplied exact evidence even when displayed means are identical. Never reconstruct a winner.
      const renderEvidenceCase = async patch => page.evaluate(serialized => {
        const patch = JSON.parse(serialized);
        const selection = simResults.strategy_selections.dry.selection;
        Object.assign(selection, patch);
        renderPitPlanSelectionResults(simResults);
        return document.getElementById('pitPlanSelectionResults').innerText;
      }, JSON.stringify(patch));
      const weightedSelection = JSON.parse(frozenWeighted).selection;
      const selected = weightedSelection.selected_label;
      const reference = weightedSelection.reference_label;
      const evidenceRow = (label, gap, tied) => ({label,
        mean_points: typeof gap === 'number' && gap >= 0 ? 8 - gap : 8,
        trials: weightedSelection.seed_ranges.training.trials,
        mean_points_behind_selected: gap, tied_for_best: tied});
      let evidenceText = await renderEvidenceCase({training_score_table: [
        evidenceRow(reference, 1e-12, false), evidenceRow(selected, 0, true),
      ]});
      assert(evidenceText.includes('1.000e-12'));
      assert.deepEqual(await weightedTraining.locator('tbody tr td:first-of-type').allTextContents(), ['8.000', '8.000']);
      for (const [reason, phrase, winner] of [
        ['reference_preferred_on_exact_tie', 'Fixed reference preferred on an exact training tie.', reference],
        ['first_plan_order_on_exact_tie', 'Candidate order decided an exact training tie.', selected],
      ]) {
        // A candidate-order tie excludes the reference from the best score.
        const rows = reason === 'first_plan_order_on_exact_tie'
          ? [evidenceRow(reference, 1, false), evidenceRow(selected, 0, true), evidenceRow('other tied candidate', 0, true)]
          : [evidenceRow(reference, 0, true), evidenceRow(selected, 0, true)];
        evidenceText = await renderEvidenceCase({selected_label: winner, tiebreak_applied: reason,
          training_score_table: rows});
        assert(evidenceText.includes(phrase));
        assert(evidenceText.includes('0.000 (exact tie)'));
      }
      evidenceText = await renderEvidenceCase({selected_label: selected,
        training_score_table: [evidenceRow(reference, 0, false), evidenceRow(selected, 0, true)]});
      assert(evidenceText.includes('Below numeric reporting precision'));
      assert(!evidenceText.includes('0.000 (exact tie)'));
      for (const badRow of [
        {label: reference, mean_points: 8, trials: 50}, evidenceRow(reference, -1, false),
        evidenceRow(reference, '0', true), evidenceRow(reference, 0, 'true'),
        evidenceRow(reference, 1, true), evidenceRow(reference, null, false),
      ]) {
        await renderEvidenceCase({training_score_table: [badRow, evidenceRow(selected, 0, true)]});
        assert.equal(await weightedTraining.locator('tbody tr').first().locator('td').nth(1).innerText(), 'Not recorded');
      }
      evidenceText = await renderEvidenceCase({tiebreak_applied: '<img src=x onerror=alert(1)>'});
      assert(evidenceText.includes('Selection reason: Not recorded.'));
      const tinyMetrics = {...weightedSelection.validation_target_metrics,
        selected_mean_points: 5 - 1e-12,
        mean_points_difference: -1e-12, points_difference_standard_error: 2e-13,
        points_outcome_profile: {...weightedSelection.validation_target_metrics.points_outcome_profile,
          mean_points_gain_when_ahead: 3e-14, mean_points_loss_when_behind: 4e-15}};
      evidenceText = await renderEvidenceCase({validation_status: 'evaluated',
        tiebreak_applied: 'unique_highest_weighted_training_mean',
        training_score_table: [evidenceRow(reference, 1e-12, false), evidenceRow(selected, 0, true)],
        validation_target_metrics: tinyMetrics,
        validation_scenario_metrics: Object.fromEntries(weightedSelection.rival_scenarios.map(item => [item.name, tinyMetrics]))});
      await weightedResult.locator('details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
      evidenceText = await weightedResult.innerText();
      for (const value of ['-1.000e-12 pts', '2.000e-13 pts', '3.000e-14 pts', '4.000e-15 pts']) {
        assert(evidenceText.includes(value), `Tiny evidence must stay visible: ${value}`);
        for (const detail of await weightedResult.locator('details').all()) {
          assert((await detail.innerText()).includes(value), `Every rival must retain tiny evidence: ${value}`);
        }
      }
      evidenceText = await renderEvidenceCase({validation_target_metrics: {...tinyMetrics,
        selected_mean_points: 5 + 1e-12, mean_points_difference: 1e-12}});
      assert(evidenceText.includes('+1.000e-12 pts'));
      assert.deepEqual(await page.evaluate(() => [selectionNumber(0), selectionNumber(NaN),
        selectionPoints(Infinity), selectionSignedNumber('0')]),
      ['0.000', 'Not recorded', 'Not recorded', 'Not recorded']);
      for (const width of [390, 1440]) {
        await page.setViewportSize({width, height: 900});
        await weightedResult.locator('details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `Small nonzero evidence overflows at ${width}px`);
        if (process.env.F1SIM_SCREENSHOTS) await page.screenshot({
          path: path.join(process.env.F1SIM_SCREENSHOTS, `selection-small-evidence-${width}.png`), fullPage: true,
        });
      }
      await page.evaluate(serialized => {
        simResults.strategy_selections.dry = JSON.parse(serialized);
        renderPitPlanSelectionResults(simResults);
      }, frozenWeighted);
      await page.locator('#tab-race').click();
      await weightedRows.first().locator('.pit-rival-weight').fill('99');
      await weightedRows.first().locator('.pit-rival-name').fill('Edited after run');
      await page.locator('#tab-scenarios').click();
      await page.evaluate(() => renderPitPlanSelectionResults(simResults));
      assert.equal(await page.evaluate(() => JSON.stringify(simResults.strategy_selections.dry)), frozenWeighted);
      assert(!(await weightedResult.innerText()).includes('Edited after run'));
      for (const kind of ['evidence', 'training', 'validation']) {
        const button = kind === 'evidence' ? 'Evidence' : kind === 'training' ? 'Training' : 'Validation';
        const downloadPromise = page.waitForEvent('download');
        await page.locator(`#downloadPitSelection${button}Btn`).click();
        const download = await downloadPromise;
        const saved = JSON.parse(readFileSync(await download.path(), 'utf8'));
        const entry = JSON.parse(frozenWeighted);
        if (kind === 'evidence') {
          assert.deepEqual(saved.selection, entry.selection);
          assert.equal(saved.selection.tiebreak_applied, 'unique_highest_weighted_training_mean');
          assert(saved.selection.training_score_table.every(row =>
            typeof row.mean_points_behind_selected === 'number' && typeof row.tied_for_best === 'boolean'));
          assert.deepEqual(saved.plans_by_rival_scenario, entry.plans_by_rival_scenario);
          assert.equal(saved.selection.rival_scenarios[0].weight, 1);
        } else {
          const grouped = entry[`${kind}_by_rival_scenario`];
          assert.deepEqual(saved.selection, entry.selection);
          assert.equal(saved.rival_scenario_index.length, 4);
          assert.equal(Object.keys(saved.scenarios).length, 4);
          for (const item of saved.rival_scenario_index) {
            assert.deepEqual(JSON.parse(item.scenario_key), [item.rival_scenario, item.candidate]);
            assert.deepEqual(saved.scenarios[item.scenario_key],
              grouped[item.rival_scenario].scenarios[item.candidate]);
            assert(saved.scenarios[item.scenario_key].simulation_inputs);
          }
        }
        await page.waitForTimeout(1100);
      }
      const weightedHtmlPromise = page.waitForEvent('download');
      await page.locator('#downloadPitSelectionReportBtn').click();
      assert.equal(readFileSync(await (await weightedHtmlPromise).path(), 'utf8'),
        JSON.parse(frozenWeighted).validation_report_html);
      await page.waitForTimeout(1100);
      const weightedIdentity = await page.evaluate(() => {
        const entry = simResults.strategy_selections.dry;
        entry.selection.selected_label = entry.selection.reference_label;
        entry.selection.validation_status = 'no_change';
        renderPitPlanSelectionResults(simResults);
        document.querySelectorAll('#pitPlanSelectionResults details').forEach(node => { node.open = true; });
        return document.getElementById('pitPlanSelectionResults').innerText;
      });
      assert(weightedIdentity.includes('Identity comparison'));
      assert(!weightedIdentity.includes('Fresh validation: selected minus reference'));
      await page.evaluate(serialized => {
        simResults.strategy_selections.dry = JSON.parse(serialized);
        renderPitPlanSelectionResults(simResults);
      }, frozenWeighted);
      for (const width of [390, 1440]) {
        await page.setViewportSize({width, height: 900});
        await page.evaluate(() => document.querySelectorAll('#pitPlanSelectionResults details')
          .forEach(node => { node.open = true; }));
        await page.waitForFunction(() => getComputedStyle(document.getElementById('panel-scenarios')).opacity === '1');
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `Weighted results overflow at ${width}px`);
        if (process.env.F1SIM_SCREENSHOTS) await page.screenshot({
          path: path.join(process.env.F1SIM_SCREENSHOTS, `rival-selection-results-${width}.png`), fullPage: true,
        });
        await page.locator('#tab-race').click();
        await page.waitForFunction(() => getComputedStyle(document.getElementById('panel-race')).opacity === '1');
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `Rival editor overflows at ${width}px`);
        if (process.env.F1SIM_SCREENSHOTS) await page.screenshot({
          path: path.join(process.env.F1SIM_SCREENSHOTS, `rival-selection-editor-${width}.png`), fullPage: true,
        });
        await page.locator('#tab-scenarios').click();
      }
      await page.locator('#tab-race').click();
      await page.locator('#pitRivalEnabled').uncheck();
      await page.locator('#pitRivalEditor').evaluate(node => { node.open = false; });
      const allCandidateRows = page.locator('#pitPlanSelectionCandidates .pit-selection-candidate');
      while (await allCandidateRows.count() < 10) await page.locator('#addPitPlanCandidateBtn').click();
      for (const width of [390, 1440]) {
        await page.setViewportSize({width, height: width === 390 ? 700 : 900});
        await page.locator('#tab-scenarios').click();
        await page.locator('#pitPlanSelectionEnabled').uncheck();
        await page.locator('#pitPlanSelectionEnabled').check();
        await page.locator('#pitPlanSelectionPanel').waitFor({state: 'visible'});
        await page.waitForFunction(() => {
          const target = document.getElementById('pitPlanSelectionTargetMode');
          const header = document.querySelector('.header');
          const targetRect = target.getBoundingClientRect();
          const headerRect = header.getBoundingClientRect();
          const sticky = getComputedStyle(header).position === 'sticky'
            && headerRect.top <= 1 && headerRect.bottom > 0;
          return document.activeElement === target && targetRect.top >= (sticky ? headerRect.bottom : 0)
            && targetRect.top < innerHeight;
        });
        assert.equal(await page.locator('#tab-race').getAttribute('aria-selected'), 'true');
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          `Ten candidate rows must not overflow at ${width}px`);
        if (process.env.F1SIM_SCREENSHOTS) {
          await page.screenshot({
            path: path.join(process.env.F1SIM_SCREENSHOTS, `pit-plan-selection-editor-${width}.png`),
            fullPage: true,
          });
        }
      }
      await page.locator('#tab-race').click();
      assert(!(await page.locator('#panel-race').innerText()).includes('Selected and frozen'),
        'Source race charts must not be relabeled as the selected strategy');
      await selectionCandidateRows.nth(0).locator('.pit-selection-label').fill('automatic');
      await selectionCandidateRows.nth(0).locator('.pit-selection-plan-mode').selectOption('automatic');
      await selectionCandidateRows.nth(1).locator('.pit-selection-label').fill('alternative');
      await page.locator('#pitPlanSelectionTargetMode').selectOption('driver');
      await page.locator('#pitPlanSelectionTargetId').selectOption('');
      await page.locator('#pitPlanTrainingTrials').fill('50');
      await page.locator('#pitPlanValidationTrials').fill('50');
      await page.locator('#pitPlanSelectionEnabled').uncheck();
      await page.locator('#simCount').fill('10');
      assert.equal(await compareAutomaticInput.isChecked(), false,
        'The selection test leaves automatic comparison off for the independent cancellation check');
      await page.locator('#tab-scenarios').click();
    }
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
