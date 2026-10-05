// Exercise recorded and missing scoring evidence in the real dashboard renderer.
const assert = require('node:assert/strict');

async function checkScoringEvidence(page) {
  await page.locator('#tab-race').click();
  const native = await page.evaluate(() => getScenarioEntry().data.sample_race_scoring_context);
  assert(native && native.policy === 'race_distance_points_2026_v1',
    'The native representative race must carry its actual scoring inputs');
  assert((await page.locator('.race-scoring-note').innerText()).includes(native.description));
  await page.evaluate(() => {
    const scenario = getScenarioEntry().data;
    window.savedScoringSample = {
      race: scenario.sample_race, context: scenario.sample_race_scoring_context,
      stats: scenario.race_scoring_statistics,
    };
    scenario.sample_race = savedScoringSample.race.slice(0, 1).map(row => ({...row,
      points_awarded: 6, points_reason: 'reduced_distance',
      points_explanation: 'Reduced points schedule'}));
    scenario.sample_race_scoring_context = {...savedScoringSample.context,
      scheduled_laps: 100, winner_laps: 24, winner_points: 6,
      description: 'Reduced points schedule (6 for first place): winner completed 24/100 laps; two consecutive complete green laps recorded.'};
    scenario.race_scoring_statistics = {
      recorded_races: 4, races_with_scoring_evidence: 3, races_without_scoring_evidence: 1,
      full_points_races: 1, reduced_points_races: 1, zero_points_races: 1,
    };
    renderRace();
  });
  assert((await page.locator('.race-scoring-note').innerText()).includes('24/100 laps'));
  assert((await page.locator('.race-scoring-note').innerText()).includes('3 of 4 recorded races; 1 unknown'));
  assert.equal(await page.locator('.points-awarded').getAttribute('title'), 'Reduced points schedule');
  await page.evaluate(() => {
    const scenario = getScenarioEntry().data;
    scenario.sample_race[0].points_awarded = 0;
    scenario.sample_race[0].points_reason = 'no_green_pair';
    scenario.sample_race[0].points_explanation = 'Two consecutive complete green laps were not recorded';
    scenario.sample_race_scoring_context.description = 'No points: two consecutive complete green laps were not recorded.';
    renderRace();
  });
  assert((await page.locator('.race-scoring-note').innerText()).includes('No points:'));
  assert((await page.locator('.points-explanation').innerText()).includes('green laps were not recorded'));
  await page.evaluate(() => {
    const scenario = getScenarioEntry().data;
    scenario.sample_race_scoring_context = null;
    scenario.sample_race[0].points_explanation = null;
    renderRace();
  });
  assert((await page.locator('.race-scoring-note').innerText()).includes('Not recorded'));
  assert.equal(await page.locator('.points-awarded').innerText(), '0 pts');
  assert.equal(await page.locator('.points-explanation').count(), 0);
  await page.evaluate(() => {
    getScenarioEntry().data.sample_race_scoring_context = {
      description: 'Full points schedule', scheduled_laps: true,
    };
    renderRace();
  });
  assert((await page.locator('.race-scoring-note').innerText()).includes('Not recorded'));
  await page.evaluate(() => {
    const scenario = getScenarioEntry().data;
    const hostile = '<img src=x onerror="globalThis.scoringInjected=true">';
    scenario.sample_race_scoring_context = {...savedScoringSample.context, description: hostile};
    scenario.sample_race[0].points_explanation = hostile;
    renderRace();
  });
  assert.equal(await page.locator('.race-scoring-note img, .points-explanation img').count(), 0);
  assert((await page.locator('.race-scoring-note').innerText()).includes('<img src=x'));
  assert.equal(await page.evaluate(() => Boolean(globalThis.scoringInjected)), false);
  for (const width of [320, 390, 768, 1440]) {
    await page.setViewportSize({width, height: 900});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      `Scoring evidence must fit at ${width}px`);
  }
  await page.evaluate(() => {
    const scenario = getScenarioEntry().data;
    scenario.sample_race = [];
    scenario.sample_race_scoring_context = {...savedScoringSample.context,
      winner_laps: null, has_two_green_laps: false, points_eligible: false,
      description: 'No points: no finishing winner.'};
    renderRace();
  });
  assert((await page.locator('.race-scoring-note').innerText()).includes('no finishing winner'));
  await page.evaluate(() => {
    const scenario = getScenarioEntry().data;
    scenario.sample_race = savedScoringSample.race;
    scenario.sample_race_scoring_context = savedScoringSample.context;
    scenario.race_scoring_statistics = savedScoringSample.stats;
    delete window.savedScoringSample;
    renderRace();
  });
}

module.exports = {checkScoringEvidence};
