const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const assert = require('node:assert/strict');
const out = process.env.REVIEW_OUTPUT;
const pagePath = process.env.REVIEW_PAGE_PATH || 'index.html';
const origin = 'http://127.0.0.1:8787/';
assert(!pagePath.startsWith('/') && !pagePath.includes('..') && !pagePath.includes(':'), 'Expected a relative website page path');
fs.mkdirSync(out, { recursive: true });
const report = {
  ref: process.env.REVIEW_REF,
  commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  pagePath, passed: false, status: 'running', screenshots: [], layouts: [], pages: [], checks: [], errors: []
};
const saveReport = () => fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
saveReport();

(async () => {
  let browser;
  const progress = message => console.log(new Date().toISOString() + ' ' + message);
  async function open(viewport, reducedMotion = 'reduce', target = pagePath) {
    progress('Open ' + target + ' ' + viewport.width + 'x' + viewport.height + ' motion=' + reducedMotion);
    const context = await browser.newContext({ viewport, deviceScaleFactor: 1, reducedMotion });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push({ page: target, error: error.message }));
    page.on('response', response => {
      if (response.status() >= 400) report.errors.push({ page: target, status: response.status(), url: response.url() });
    });
    await page.goto(new URL(target, origin).href, { waitUntil: 'networkidle' });
    report.codec = await page.evaluate(() => document.createElement('video').canPlayType('video/mp4; codecs="avc1.640028"'));
    return page;
  }
  async function position(page, selector) {
    if (!selector) { await page.evaluate(() => scrollTo({ top: 0, behavior: 'instant' })); return; }
    await page.locator(selector).first().evaluate(element => {
      const targetTop = element.getBoundingClientRect().top + window.scrollY;
      const headers = [...document.querySelectorAll('header')].filter(header => ['sticky', 'fixed'].includes(getComputedStyle(header).position));
      const offset = Math.max(0, ...headers.map(header => header.getBoundingClientRect().height));
      window.scrollTo({ top: targetTop - offset - 16, behavior: 'instant' });
    });
  }
  async function stableMedia(page) {
    return page.evaluate(async () => {
      const visible = element => {
        if (typeof element.checkVisibility === 'function' && !element.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) return false;
        for (let ancestor = element; ancestor; ancestor = ancestor.parentElement) {
          const style = getComputedStyle(ancestor);
          if (ancestor.hidden || style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse' || Number(style.opacity) === 0) return false;
          if (ancestor.tagName === 'DETAILS' && !ancestor.open) {
            const summary = [...ancestor.children].find(child => child.tagName === 'SUMMARY');
            if (!summary?.contains(element)) return false;
          }
        }
        const box = element.getBoundingClientRect();
        return box.width > 0 && box.height > 0 && box.top < innerHeight && box.bottom > 0 && box.left < innerWidth && box.right > 0;
      };
      await document.fonts.ready;
      await Promise.all([...document.images].filter(visible).map(async image => {
        await new Promise((resolve, reject) => {
          const source = image.currentSrc || image.src || '<missing image source>';
          const timer = setTimeout(() => reject(new Error('Visible image decode timed out after 15s: ' + source)), 15000);
          image.decode().then(() => {
            clearTimeout(timer);
            resolve();
          }, error => {
            clearTimeout(timer);
            reject(new Error('Visible image decode failed: ' + source + ': ' + error.message));
          });
        });
        if (!image.naturalWidth) throw new Error('Visible image failed to decode: ' + image.src);
      }));
      document.querySelectorAll('video').forEach(video => video.pause());
      const videos = [...document.querySelectorAll('video')].filter(visible);
      await Promise.all(videos.map(async video => {
        // Screenshot preparation requests a real decoded frame, without changing
        // controls, layout, playback policy, or any visible page styling.
        video.preload = 'auto';
        const wait = (event, ready) => new Promise((resolve, reject) => {
          if (ready()) return resolve();
          const timer = setTimeout(() => { cleanup(); reject(new Error('Video timed out waiting for ' + event + ': ' + video.currentSrc)); }, 15000);
          const done = () => { cleanup(); resolve(); };
          const fail = () => { cleanup(); reject(new Error('Video media error: ' + video.currentSrc)); };
          const cleanup = () => { clearTimeout(timer); video.removeEventListener(event, done); video.removeEventListener('error', fail); };
          video.addEventListener(event, done, { once: true });
          video.addEventListener('error', fail, { once: true });
        });
        if (!video.getAttribute('src') && video.dataset.src) video.src = video.dataset.src;
        if (!video.getAttribute('src') && !video.querySelector('source')) return;
        if (video.readyState < 2) {
          const loaded = wait('loadeddata', () => video.readyState >= 2);
          video.load();
          await loaded;
        }
        const time = Math.min(0.5, Number.isFinite(video.duration) ? video.duration / 2 : 0.5);
        if (Math.abs(video.currentTime - time) > 0.01) {
          const seeked = wait('seeked', () => false);
          video.currentTime = time;
          await seeked;
        }
        await new Promise((resolve, reject) => {
          const ready = () => video.readyState >= 3 && !video.seeking;
          if (ready()) return resolve();
          const cleanup = () => {
            clearTimeout(timer);
            video.removeEventListener('canplay', check);
            video.removeEventListener('seeked', check);
            video.removeEventListener('error', fail);
          };
          const check = () => { if (ready()) { cleanup(); resolve(); } };
          const fail = () => { cleanup(); reject(new Error('Video failed before stable canplay: ' + video.currentSrc)); };
          const timer = setTimeout(() => {
            cleanup();
            reject(new Error('Video did not reach stable canplay after 15s: ' + video.currentSrc + ' readyState=' + video.readyState + ' seeking=' + video.seeking));
          }, 15000);
          video.addEventListener('canplay', check);
          video.addEventListener('seeked', check);
          video.addEventListener('error', fail, { once: true });
        });
        video.pause();
      }));
      // Give native controls time to finish their own loading-state transition.
      // Nothing is hidden, restyled, or removed from the page for the capture.
      await new Promise(resolve => setTimeout(resolve, 500));
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('Two animation frames did not settle within 5s; visibility=' + document.visibilityState)), 5000);
        requestAnimationFrame(() => requestAnimationFrame(() => { clearTimeout(timer); resolve(); }));
      });
      return videos.map(video => ({ src: video.currentSrc || video.getAttribute('src'), time: video.currentTime, readyState: video.readyState, seeking: video.seeking }));
    });
  }
  async function layout(page, label) {
    const result = await page.evaluate(() => ({
      width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
      overflow: [...document.querySelectorAll('body *')].filter(element => {
        const box = element.getBoundingClientRect();
        return box.width && (box.right > innerWidth + 1 || box.left < -1) && getComputedStyle(element).position !== 'fixed';
      }).slice(0, 20).map(element => element.tagName + '.' + element.className)
    }));
    report.layouts.push({ page: page.url(), label, ...result });
    assert(result.scrollWidth <= result.width + 1, 'Horizontal page overflow: ' + label + ' at ' + result.width);
  }
  async function capture(page, name, selector) {
    progress('Capture start: ' + name + ' selector=' + (selector || 'page top'));
    await page.bringToFront();
    await position(page, selector);
    const visibleVideos = await stableMedia(page);
    await layout(page, name);
    const file = name + '.jpg';
    await page.screenshot({ path: path.join(out, file), type: 'jpeg', quality: 75, fullPage: false, animations: 'disabled' });
    report.screenshots.push({ file, page: page.url(), viewport: page.viewportSize(), selector: selector || 'page top', scrollY: await page.evaluate(() => scrollY), visibleVideos });
    saveReport();
    progress('Capture done: ' + name);
  }
  async function captureSections(page, prefix) {
    const sections = await page.locator('main section[id]').evaluateAll(elements => elements.map(element => element.id));
    for (const id of sections) {
      const selector = '[id=' + JSON.stringify(id) + ']';
      await capture(page, prefix + '-' + id.replace(/[^a-zA-Z0-9_-]/g, '-'), selector);
    }
  }
  async function captureEvidenceDetails(page, prefix) {
    const table = await page.locator('.transfer-evidence .table-block').count() ? '.transfer-evidence .table-block' : '.transfer-evidence .table-wrap';
    if (await page.locator(table).count()) await capture(page, prefix + '-transfer-table', table);
    const studies = page.locator('.research-study');
    if (await studies.count()) {
      const index = await studies.count() - 1;
      await capture(page, prefix + '-policy-comparison', '.research-study >> nth=' + index);
    }
  }
  async function mediaSnapshot(locator) {
    return locator.evaluate(video => {
      const ranges = list => Array.from({ length: list.length }, (_, index) => [list.start(index), list.end(index)]);
      return { source: video.currentSrc, time: video.currentTime, duration: video.duration, readyState: video.readyState, seeking: video.seeking, seekable: ranges(video.seekable), buffered: ranges(video.buffered) };
    });
  }
  async function interactions(page) {
    progress('Interactions start: media selections');
    const options = page.locator('[data-select]');
    for (let index = 0; index < await options.count(); index++) {
      const button = options.nth(index);
      const option = await button.evaluate(element => ({ ...element.dataset }));
      progress('Select ' + option.select + ': ' + option.src);
      await page.locator('#' + option.select).evaluate(video => { video.preload = 'auto'; });
      await button.click();
      await page.waitForFunction(({ select, src }) => {
        const video = document.getElementById(select);
        return video.getAttribute('src') === src && video.readyState >= 2;
      }, option);
      assert.equal(await button.getAttribute('aria-pressed'), 'true');
      assert.equal(await page.locator('#' + option.select).getAttribute('poster'), option.poster);
      assert.equal(await page.locator('#' + option.select + '-caption').textContent(), option.caption);
      if (option.layout) assert.equal(await page.locator('#' + option.select).getAttribute('data-layout'), option.layout);
      assert(await page.locator('#' + option.select).evaluate(video => video.paused), 'Reduced-motion selection must remain paused');
    }
    if (await options.count()) report.checks.push('all media selections update source/poster/caption/pressed state and remain paused under reduced motion');
    const appearance = page.locator('#appearance-film[data-sync-time]');
    if (await appearance.count()) {
      progress('Interactions: material timestamp preservation');
      const button = page.locator('[data-select="appearance-film"]').first();
      const source = await button.getAttribute('data-src');
      const expectedSource = new URL(source, page.url()).href;
      report.materialTiming = { expectedTime: 1.25, expectedSource, tolerance: 0.1, events: [] };
      await appearance.evaluate(video => {
        const ranges = list => Array.from({ length: list.length }, (_, index) => [list.start(index), list.end(index)]);
        const state = () => ({ source: video.currentSrc, time: video.currentTime, duration: video.duration, readyState: video.readyState, seeking: video.seeking, seekable: ranges(video.seekable), buffered: ranges(video.buffered) });
        window.__reviewMaterialEvents = [];
        for (const type of ['loadedmetadata', 'loadeddata', 'seeked']) {
          video.addEventListener(type, () => window.__reviewMaterialEvents.push({ type, timestamp: performance.now(), ...state() }));
        }
        video.currentTime = 1.25;
      });
      report.materialTiming.before = await mediaSnapshot(appearance);
      saveReport();
      try {
        await page.waitForFunction(() => {
          const video = document.getElementById('appearance-film');
          return Math.abs(video.currentTime - 1.25) < 0.1 && !video.seeking;
        }, null, { timeout: 5000 });
        report.materialTiming.before = await mediaSnapshot(appearance);
        saveReport();
        await button.click();
        await page.waitForFunction(src => {
          const video = document.getElementById('appearance-film');
          return video.currentSrc === src && video.readyState >= 2 && !video.seeking && Math.abs(video.currentTime - 1.25) < 0.1;
        }, expectedSource, { timeout: 5000 });
        assert(Math.abs(await appearance.evaluate(video => video.currentTime) - 1.25) < 0.1, 'Material selection must preserve replay time');
      } finally {
        report.materialTiming.after = await mediaSnapshot(appearance);
        report.materialTiming.events = await page.evaluate(() => window.__reviewMaterialEvents);
        saveReport();
      }
      report.checks.push('material switch preserves current physical timestamp');
    }
    const expand = page.locator('[data-expand]');
    for (let index = 0; index < await expand.count(); index++) {
      const button = expand.nth(index);
      const id = await button.getAttribute('data-expand');
      progress('Interactions: expand and close ' + id);
      const source = await page.locator('#' + id).evaluate(video => video.currentSrc || video.dataset.src);
      await button.click();
      assert(await page.locator('#media-dialog').evaluate(dialog => dialog.open));
      await page.waitForFunction(() => document.getElementById('expanded-film').readyState >= 2);
      assert.equal(await page.locator('#expanded-film').evaluate(video => video.currentSrc), new URL(source, origin).href);
      if (index % 2) await page.locator('#close-media').click();
      else await page.keyboard.press('Escape');
      await page.waitForFunction(() => !document.getElementById('expanded-film').hasAttribute('src'));
      assert(await page.locator('#expanded-film').evaluate(video => video.paused));
    }
    if (await expand.count()) report.checks.push('all expanded replays, close button/Escape and media-source cleanup');
    if (await page.locator('#copy-commands').count()) {
      progress('Interactions: clipboard copy');
      await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
      await page.locator('#copy-commands').click();
      await page.waitForFunction(async () => await navigator.clipboard.readText() === document.getElementById('commands').textContent);
      assert.equal(await page.evaluate(() => navigator.clipboard.readText()), await page.locator('#commands').textContent());
      report.checks.push('CPU quickstart clipboard copy');
    }
    progress('Interactions done');
  }

  async function tableStyle(page) {
    assert.equal(await page.locator('table:not(.data-table)').count(), 0, 'All research tables must use shared data-table styling');
    return page.locator('.data-table th').first().evaluate(header => {
      const style = getComputedStyle(header);
      return Object.fromEntries(['fontFamily', 'fontSize', 'color', 'paddingTop', 'paddingBottom', 'borderBottomColor'].map(key => [key, style[key]]));
    });
  }
  try {
    browser = await chromium.launch({ channel: 'chrome', args: ['--disable-gpu'] });
    report.browser = browser.version();
    progress('Verify HTTP byte-range serving');
    const probe = await browser.newContext();
    try {
      const range = await probe.request.get(new URL('assets/launch/appearance-checkered.mp4', origin).href, { headers: { Range: 'bytes=0-1023' } });
      report.httpRange = { status: range.status(), contentRange: range.headers()['content-range'], acceptRanges: range.headers()['accept-ranges'], bytes: (await range.body()).length };
      saveReport();
      assert.equal(report.httpRange.status, 206, 'Media server must serve byte-range requests with HTTP 206');
      assert.match(report.httpRange.contentRange || '', /^bytes 0-1023\/\d+$/, 'Media server returned an invalid Content-Range');
      assert.equal(report.httpRange.bytes, 1024, 'Media server returned the wrong byte-range length');
      report.checks.push('HTTP 206 byte-range media serving');
      progress('Observe deployed GitHub Pages byte-range response');
      const hostedUrl = 'https://clothdojo.github.io/assets/hero.mp4';
      try {
        const hosted = await probe.request.get(hostedUrl, { headers: { Range: 'bytes=0-1023' }, timeout: 15000 });
        report.hostedRange = { url: hostedUrl, status: hosted.status(), contentRange: hosted.headers()['content-range'], acceptRanges: hosted.headers()['accept-ranges'] };
      } catch (error) {
        report.hostedRange = { url: hostedUrl, error: error.message };
      }
      saveReport();
    } finally {
      await probe.close();
    }
    const desktop = await open({ width: 1440, height: 1000 });
    await capture(desktop, 'desktop-opening');
    await captureSections(desktop, 'desktop');
    await captureEvidenceDetails(desktop, 'desktop');
    await interactions(desktop);
    const mainTable = pagePath === 'index.html' ? await tableStyle(desktop) : null;
    report.tableStyles = { homepage: mainTable };
    for (const width of [390, 320]) {
      const mobile = await open({ width, height: 844 });
      if (await mobile.locator('#hero-film').count()) {
        assert((await mobile.locator('#hero-film').getAttribute('data-src')).endsWith('flatten-wide.mp4'), 'Mobile hero must default to one complete workstation');
        assert.equal(await mobile.locator('[data-select="hero-film"][aria-pressed="true"]').getAttribute('data-src'), 'assets/launch/flatten-wide.mp4');
        assert.equal(await mobile.locator('[data-select="hero-film"].desktop-overview').isVisible(), false, 'Batch hero overview must be hidden on mobile');
      }
      if (await mobile.locator('#collection-film').count()) {
        assert((await mobile.locator('#collection-film').getAttribute('data-src')).endsWith('sample-01.mp4'), 'Mobile collection must default to a readable single replay');
        assert.equal(await mobile.locator('[data-select="collection-film"][aria-pressed="true"]').getAttribute('data-src'), 'assets/launch/sample-01.mp4');
        assert.equal(await mobile.locator('[data-select="collection-film"].desktop-overview').isVisible(), false, 'Collection overview must be hidden on mobile');
      }
      if (await mobile.locator('#transfer-film').count()) {
        assert((await mobile.locator('#transfer-film').getAttribute('data-src')).endsWith('transfer-source.mp4'), 'Mobile transfer must default to a readable single source replay');
        assert.equal(await mobile.locator('#transfer-film').getAttribute('data-layout'), 'single');
        assert.equal(await mobile.locator('[data-select="transfer-film"][aria-pressed="true"]').getAttribute('data-src'), 'assets/launch/transfer-source.mp4');
        assert.equal(await mobile.locator('[data-select="transfer-film"].desktop-overview').isVisible(), false, 'Side-by-side transfer must be hidden on mobile');
      }
      await capture(mobile, 'mobile-' + width + '-opening');
      await captureSections(mobile, 'mobile-' + width);
      await captureEvidenceDetails(mobile, 'mobile-' + width);
      await mobile.context().close();
    }
    const active = await open({ width: 1440, height: 1000 }, 'no-preference');
    progress('Playback checks start');
    if (await active.locator('#hero-film').count()) {
      await position(active, '#hero-film');
      await active.waitForFunction(() => document.getElementById('hero-film').hasAttribute('src'));
      await active.locator('#hero-film').evaluate(async video => { await video.play(); });
      await active.locator('[data-select="hero-film"]').last().click();
      await active.waitForFunction(() => !document.getElementById('hero-film').paused);
      assert.equal(await active.locator('video').evaluateAll(videos => videos.filter(video => !video.paused).length), 1);
      await active.locator('[data-expand="hero-film"]').click();
      await active.waitForFunction(() => !document.getElementById('expanded-film').paused);
      assert.equal(await active.locator('video').evaluateAll(videos => videos.filter(video => !video.paused).length), 1);
      report.checks.push('real Chromium playback: one active video after task switch and HD open');
    }
    await active.context().close();
    progress('Playback checks done');
    for (const width of [1440, 390, 320]) {
      const results = await open({ width, height: width === 1440 ? 1000 : 844 }, 'reduce', 'results.html');
      const anchors = ['generation-results', 'generalization', 'policy-models', 'outcome-quality', 'task-composition'];
      for (const id of anchors) assert.equal(await results.locator('#' + id).count(), 1, 'Missing results anchor ' + id);
      const style = await tableStyle(results);
      if (width === 1440) {
        report.tableStyles.results = style;
        if (mainTable) assert.deepEqual(style, mainTable, 'Homepage and results table styles differ');
      }
      await capture(results, 'results-' + width + '-opening');
      await captureSections(results, 'results-' + width);
      const tableSelector = await results.locator('.table-block').count() ? '.table-block' : '.table-wrap';
      await capture(results, 'results-' + width + '-table', tableSelector);
      await results.context().close();
    }
    report.checks.push('five results anchors, shared rendered table typography, desktop/mobile results screenshots');
    const pages = fs.readdirSync('website').filter(file => file.endsWith('.html'));
    for (const target of pages) {
      for (const width of [1440, 390]) {
        progress('Smoke page: ' + target + ' width=' + width);
        const page = await open({ width, height: 1000 }, 'reduce', target);
        const css = await page.evaluate(() => [...document.querySelectorAll('link[rel="stylesheet"]')].map(link => ({ href: link.href, loaded: !!link.sheet })));
        assert(css.every(sheet => sheet.loaded), 'A stylesheet failed to load: ' + target);
        if (!target.includes('preview')) await layout(page, 'smoke-' + target);
        report.pages.push({ path: target, viewport: page.viewportSize(), stylesheets: css });
        await page.context().close();
      }
    }
    assert.equal(report.errors.length, 0, 'Browser or HTTP errors');
    report.passed = true;
    progress('Visual review passed');
  } catch (error) {
    progress('Visual review failed: ' + error.message);
    report.failure = error.stack;
    report.passed = false;
    process.exitCode = 1;
  } finally {
    report.status = 'completed';
    saveReport();
    progress('Report written: ' + path.join(out, 'report.json'));
    if (browser) await browser.close();
  }
})();
