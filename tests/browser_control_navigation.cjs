// Exercise real pointer and keyboard navigation when settings occupy a tall header.
const assert = require('node:assert/strict');

async function checkControlNavigation(page) {
  const viewport = page.viewportSize();
  const original = await page.evaluate(() => ({
    fontSize: document.documentElement.style.fontSize,
    editorHeight: document.getElementById('pitPlansInput').style.height,
    activeTab: document.querySelector('[role="tab"][aria-selected="true"]').dataset.tab,
    focusId: document.activeElement?.id,
    scrollX: window.scrollX,
    scrollY: window.scrollY,
  }));
  const controlValues = () => page.locator('.header input, .header textarea, .header select')
    .evaluateAll(controls => controls.map(control => ({
      id: control.id, value: control.value, checked: control.checked,
    })));
  const values = await controlValues();
  const tabNames = ['race', 'qualifying', 'stats', 'scenarios'];
  const layouts = [
    {width: 1440, height: 900, fontSize: 15, editorHeight: 0},
    {width: 1440, height: 900, fontSize: 21, editorHeight: 180},
    {width: 1440, height: 600, fontSize: 30, editorHeight: 280},
    {width: 860, height: 600, fontSize: 18, editorHeight: 180},
    {width: 390, height: 700, fontSize: 15, editorHeight: 0},
    {width: 390, height: 700, fontSize: 21, editorHeight: 180},
    {width: 320, height: 600, fontSize: 21, editorHeight: 280},
  ];
  const assertSelected = async (name, context) => {
    assert.equal(await page.locator('#tab-' + name).getAttribute('aria-selected'),
      'true', context + ': selected tab');
    assert.equal(await page.locator('[role="tab"][aria-selected="true"]').count(),
      1, context + ': exactly one selected tab');
    assert(await page.locator('#panel-' + name).isVisible(), context + ': active panel');
  };
  const assertPointerTarget = async (name, context) => {
    const target = await page.locator('#tab-' + name).evaluate(tab => {
      const rect = tab.getBoundingClientRect();
      const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
      return {reachable: tab === hit || tab.contains(hit), hit: hit?.id || hit?.className,
        bounds: rect.toJSON()};
    });
    assert(target.reachable, context + ': tab covered by ' + target.hit +
      ' at ' + JSON.stringify(target.bounds));
  };
  try {
    for (const layout of layouts) {
      const context = `${layout.width}x${layout.height}, ${layout.fontSize}px text, ` +
        `${layout.editorHeight}px editor`;
      await page.setViewportSize({width: layout.width, height: layout.height});
      await page.evaluate(({fontSize, editorHeight}) => {
        document.documentElement.style.fontSize = `${fontSize}px`;
        document.getElementById('pitPlansInput').style.height = editorHeight ? `${editorHeight}px` : '';
      }, layout);
      for (const name of tabNames) {
        // Return from the bottom of the results before every native pointer click.
        await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
        await page.locator('#tab-' + name).scrollIntoViewIfNeeded();
        await assertPointerTarget(name, context);
        await page.locator('#tab-' + name).click({timeout: 10000});
        await assertSelected(name, context);
      }
      // Focus must remain visible while the roving tab index wraps in both directions.
      for (const [key, name] of [
        ['Home', 'race'], ['ArrowLeft', 'scenarios'], ['ArrowRight', 'race'],
        ['End', 'scenarios'], ['ArrowLeft', 'stats'], ['ArrowRight', 'scenarios'],
      ]) {
        await page.keyboard.press(key);
        assert.equal(await page.evaluate(() => document.activeElement?.id),
          'tab-' + name, context + ': keyboard focus after ' + key);
        await assertSelected(name, context);
        await assertPointerTarget(name, context + ': keyboard focus after ' + key);
      }
      assert.deepEqual(await controlValues(), values, context + ': settings preserved');
    }
  } finally {
    if (viewport) await page.setViewportSize(viewport);
    await page.evaluate(saved => {
      document.documentElement.style.fontSize = saved.fontSize;
      document.getElementById('pitPlansInput').style.height = saved.editorHeight;
      switchTab(saved.activeTab);
      if (saved.focusId) document.getElementById(saved.focusId)?.focus({preventScroll: true});
      window.scrollTo(saved.scrollX, saved.scrollY);
    }, original);
  }
  console.log('Dashboard tabs remain reachable with resized settings, large text and keyboard navigation.');
}

module.exports = {checkControlNavigation};
