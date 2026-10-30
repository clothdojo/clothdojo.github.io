const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const out = process.env.REVIEW_OUTPUT;
fs.mkdirSync(out, { recursive: true });
const report = { commit: process.env.REVIEW_SHA, screenshots: [], layouts: [], checks: [], errors: [] };

(async () => {
  const browser = await chromium.launch({ args: ['--disable-gpu'] });
  report.browser = browser.version();
  async function open(viewport, reducedMotion = 'reduce') {
    const context = await browser.newContext({ viewport, deviceScaleFactor: 1, reducedMotion });
    const page = await context.newPage();
    page.on('pageerror', e => report.errors.push(e.message));
    page.on('response', r => { if (r.status() >= 400) report.errors.push(r.status() + ' ' + r.url()); });
    await page.goto('http://127.0.0.1:8787/', { waitUntil: 'networkidle' });
    return page;
  }
  async function capture(page, name, selector) {
    if (selector) await page.locator(selector).first().scrollIntoViewIfNeeded();
    await page.evaluate(() => document.querySelectorAll('video').forEach(v => v.pause()));
    await page.evaluate(async () => {
      await document.fonts.ready;
      await Promise.all([...document.images].filter(i => i.complete).map(i => i.decode().catch(() => {})));
    });
    await page.screenshot({ path: path.join(out, name + '.png'), fullPage: false, animations: 'disabled' });
    report.screenshots.push(name);
  }
  try {
    const desktop = await open({ width: 1440, height: 1000 });
    await capture(desktop, 'desktop-opening');
    for (const [name, selector] of [['generation', '#generation'], ['appearance', '.appearance-figure'], ['transfer', '.robot-comparison'], ['results', '#results'], ['use', '#use']]) {
      await capture(desktop, 'desktop-' + name, selector);
    }
    await desktop.locator('[data-task="fold"]').click();
    assert.equal(await desktop.locator('#hero-video').getAttribute('src'), 'assets/curated/hero-fold.mp4');
    await capture(desktop, 'desktop-fold', '.showcase');
    await desktop.locator('#hero-hd').click();
    assert(await desktop.locator('#film-dialog').evaluate(d => d.open));
    await desktop.keyboard.press('Escape');
    await desktop.waitForFunction(() => !document.getElementById('film-video').hasAttribute('src'));
    assert.equal(await desktop.locator('#film-video').getAttribute('src'), null);
    for (const target of ['aloha2', 'x5', 'fr3_duo', 'ur5e_2f85']) {
      await desktop.locator('[data-compare="' + target + '"]').click();
      assert.equal(await desktop.locator('#embodiment-video').getAttribute('src'), 'assets/curated/compare-' + target + '.mp4');
    }
    report.checks.push('task selection, HD dialog/Escape/source cleanup, four target robots');
    for (const width of [390, 320]) {
      const mobile = await open({ width, height: 844 });
      await capture(mobile, 'mobile-' + width + '-opening');
      for (const [name, selector] of [['hero', '.showcase'], ['appearance', '.appearance-figure'], ['transfer', '.transfer-evidence'], ['results', '.experiment-charts'], ['use', '.use-start']]) {
        await capture(mobile, 'mobile-' + width + '-' + name, selector);
      }
      const layout = await mobile.evaluate(() => ({ width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
        overflow: [...document.querySelectorAll('body *')].filter(e => { const r = e.getBoundingClientRect(); return r.width && r.right > innerWidth + 1 && getComputedStyle(e).position !== 'fixed'; }).slice(0, 20).map(e => e.tagName + '.' + e.className) }));
      report.layouts.push(layout);
      assert(layout.scrollWidth <= width + 1, 'Horizontal page overflow at ' + width);
    }
    const active = await open({ width: 1440, height: 1000 }, 'no-preference');
    await active.locator('#hero-video').evaluate(async v => { await v.play(); });
    await active.locator('[data-task="fold"]').click();
    await active.waitForFunction(() => !document.getElementById('hero-video').paused);
    assert.equal(await active.locator('video').evaluateAll(vs => vs.filter(v => !v.paused).length), 1);
    await active.locator('#hero-hd').click();
    await active.waitForFunction(() => !document.getElementById('film-video').paused);
    assert.equal(await active.locator('video').evaluateAll(vs => vs.filter(v => !v.paused).length), 1);
    report.checks.push('real Chromium playback: one active video after task switch and HD open');
    assert.equal(report.errors.length, 0, 'Browser or HTTP errors');
    report.passed = true;
  } catch (error) {
    report.failure = error.stack;
    report.passed = false;
    process.exitCode = 1;
  } finally {
    fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
    await browser.close();
  }
})();
