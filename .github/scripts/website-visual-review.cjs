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
    // Ambient video can keep range requests active. Media readiness is checked
    // explicitly below instead of treating an idle network as page readiness.
    await page.goto(new URL(target, origin).href, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => document.fonts.ready);
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
  async function stableMedia(page, frameTime = null, frameSelector = null) {
    return page.evaluate(async ({ frameTime, frameSelector }) => {
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
      const sampleVideo = frameTime === null ? null : document.querySelector(frameSelector);
      if (frameTime !== null && (!(sampleVideo instanceof HTMLVideoElement) || !videos.includes(sampleVideo))) {
        throw new Error('Timed frame capture requires a visible video: ' + frameSelector);
      }
      for (const video of videos) {
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
        if (!video.getAttribute('src') && !video.querySelector('source')) continue;
        if (video.readyState < 2) {
          const loaded = wait('loadeddata', () => video.readyState >= 2);
          video.load();
          await loaded;
        }
        // Present an actual playback frame before seeking a paused screenshot.
        // Native loading controls are never hidden or restyled for the review.
        const presented = new Promise((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('No presented video frame after explicit playback: ' + video.currentSrc)), 5000);
          video.requestVideoFrameCallback(() => { clearTimeout(timer); resolve(); });
        });
        await video.play();
        await presented;
        video.pause();
        const time = video === sampleVideo ? frameTime : Math.min(0.5, Number.isFinite(video.duration) ? video.duration / 2 : 0.5);
        if (video === sampleVideo && (!Number.isFinite(video.duration) || time >= video.duration)) {
          throw new Error('Requested frame ' + time + 's is outside video duration ' + video.duration + 's: ' + video.currentSrc);
        }
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
      }
      // Give native controls time to finish their own loading-state transition.
      // Nothing is hidden, restyled, or removed from the page for the capture.
      await new Promise(resolve => setTimeout(resolve, 500));
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('Two animation frames did not settle within 5s; visibility=' + document.visibilityState)), 5000);
        requestAnimationFrame(() => requestAnimationFrame(() => { clearTimeout(timer); resolve(); }));
      });
      return videos.map(video => ({ id: video.id, src: video.currentSrc || video.getAttribute('src'), currentSrc: video.currentSrc,
        time: video.currentTime, duration: video.duration, videoWidth: video.videoWidth, videoHeight: video.videoHeight,
        readyState: video.readyState, seeking: video.seeking, controls: video.controls, paused: video.paused }));
    }, { frameTime, frameSelector });
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
  async function capture(page, name, selector, frameTime = null) {
    assert(frameTime === null || Number.isFinite(frameTime) && frameTime >= 0, 'Frame capture time must be finite and nonnegative');
    progress('Capture start: ' + name + ' selector=' + (selector || 'page top') + (frameTime === null ? '' : ' time=' + frameTime));
    await page.bringToFront();
    await position(page, selector);
    const visibleVideos = await stableMedia(page, frameTime, selector);
    await layout(page, name);
    const file = name + '.jpg';
    await page.screenshot({ path: path.join(out, file), type: 'jpeg', quality: 75, fullPage: false, animations: 'disabled' });
    const evidence = { file, page: page.url(), viewport: page.viewportSize(), selector: selector || 'page top', frameTime, scrollY: await page.evaluate(() => scrollY), visibleVideos };
    report.screenshots.push(evidence);
    saveReport();
    progress('Capture done: ' + name);
    return evidence;
  }
  async function heroReveal(page, prefix, captureFrames) {
    if (pagePath !== 'index.html') return;
    const video = page.locator('#hero-film');
    assert.equal(await page.locator('[data-select="hero-film"]').count(),0,'Both tasks must be in the automatic film');
    assert.equal(await page.locator('[data-seek]').count(),3);
    const mobile=page.viewportSize().width<=700;
    assert.equal(await video.getAttribute('data-src'),'assets/studio26/experience-loop'+(mobile?'-portrait':'')+'.mp4');
    report.checks.push(prefix + ': one automatic flattening/folding/scale film, with chapter seeking');
    if (!captureFrames) return;
    for (const time of [0,8.1,13,19.2,26,31.8]) {
      const shot=await capture(page,prefix+'-hero-'+String(time).replace('.','p')+'s','#hero-film',time);
      const state=shot.visibleVideos.find(item=>item.id==='hero-film');
      assert(state && Math.abs(state.duration-32)<.05 && Math.abs(state.time-time)<.05);
      assert.equal(state.videoWidth,mobile?1080:2560);assert.equal(state.videoHeight,mobile?1350:1440);
      assert(state.readyState>=3 && !state.seeking && state.paused);
      const index=time<9.25?0:time<19.5?1:2;
      assert.equal(await page.locator('[data-seek]').nth(index).getAttribute('aria-pressed'),'true');
    }
    for (const index of [1,0,2]) {
      const button=page.locator('[data-seek]').nth(index);
      const expected=Number(await button.getAttribute('data-seek'));
      await button.click();
      await page.waitForFunction(t=>Math.abs(document.getElementById('hero-film').currentTime-t)<.3,expected);
      await video.evaluate(v=>v.pause());
    }
  }
  async function captureSections(page, prefix) {
    const sections = await page.locator('main section[id]').evaluateAll(elements => elements.map(element => element.id));
    for (const id of sections) {
      const selector = '[id=' + JSON.stringify(id) + ']';
      await capture(page, prefix + '-' + id.replace(/[^a-zA-Z0-9_-]/g, '-'), selector);
    }
  }
  async function rerenderComparison(page,prefix) {
    if(pagePath!=='index.html')return;
    assert.equal(await page.locator('#render video').count(),1,'Randomization and retargeting share one scene and decoder');
    const mobile=page.viewportSize().width<=700;
    for(const time of [2.5,14.5,23.8]) {
      const shot=await capture(page,prefix+'-paper-appearance-'+String(time).replace('.','p'),'#transfer-film',time);
      const state=shot.visibleVideos.find(item=>item.id==='transfer-film');
      assert(state && state.currentSrc.endsWith(mobile?'/domains-robots-mobile.mp4':'/domains-robots.mp4'));
      assert.equal(state.videoWidth,mobile?1440:2560);
      assert.equal(state.videoHeight,mobile?810:1440);
      assert(Math.abs(state.duration-24)<.05 && Math.abs(state.time-time)<.05 && state.readyState>=3 && !state.seeking);
      assert.match(await page.locator('#domain-scene-caption').textContent(),time<12?/Office/:/Warehouse/);
    }
    report.checks.push(prefix+': both paper-sampled appearances and the final home return decoded');
  }
  async function responsivePairs(page) {
    if(pagePath!=='index.html')return;
    const viewport=page.viewportSize();
    await position(page,'#transfer-film');
    await stableMedia(page,2.5,'#transfer-film');
    for(const width of [1440,viewport.width]) {
      await page.setViewportSize({width,height:viewport.height});
      const source=new URL('assets/studio26/domains-robots'+(width<=700?'-mobile':'')+'.mp4',origin).href;
      await page.waitForFunction(source=>{
        const v=document.getElementById('transfer-film');
        return v.currentSrc===source && v.readyState>=3 && !v.seeking && Math.abs(v.currentTime-2.5)<.1;
      },source);
      assert(await page.locator('#transfer-film').evaluate(v=>v.paused));
    }
    report.checks.push('Responsive domain/robot scenes preserve paused timestamp');
  }
  async function formalTitle(page, prefix) {
    if (pagePath !== 'index.html') return;
    const expected = 'ClothDojo: Learned Data Generation and Benchmarking for Bimanual Garment Flattening and Folding';
    assert.equal(await page.locator('#title').count(), 1, 'Homepage must have one formal project title');
    await capture(page, prefix + '-title', '#title');
    const title = await page.locator('#title').evaluate(element => {
      const normalize = text => text.replace(/\s+/g, ' ').trim();
      const bounds = box => ({ top: box.top, right: box.right, bottom: box.bottom, left: box.left, width: box.width, height: box.height });
      const headerBottom = Math.max(0, ...[...document.querySelectorAll('header')]
        .filter(header => ['sticky', 'fixed'].includes(getComputedStyle(header).position))
        .map(header => header.getBoundingClientRect().bottom));
      const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
      const textRuns = [];
      while (walker.nextNode()) {
        const node = walker.currentNode;
        if (!node.textContent.trim()) continue;
        const range = document.createRange();
        range.selectNodeContents(node);
        textRuns.push({ text: normalize(node.textContent), visible: node.parentElement.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }), rects: [...range.getClientRects()].map(bounds) });
      }
      return { text: normalize(element.textContent), renderedText: normalize(element.innerText), tag: element.tagName,
        insideDetails: !!element.closest('details'), visible: element.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }),
        bounds: bounds(element.getBoundingClientRect()), width: innerWidth, height: innerHeight, headerBottom,
        clipped: element.scrollWidth > element.clientWidth + 1 || element.scrollHeight > element.clientHeight + 1, textRuns };
    });
    (report.titles ||= []).push({ viewport: page.viewportSize(), screenshot: prefix + '-title.jpg', ...title });
    saveReport();
    assert.equal(title.tag, 'H1', 'The formal project title must be the main heading');
    assert.equal(title.text, expected, 'The main heading must contain the full formal project title');
    assert.equal(title.renderedText.toLowerCase(), expected.toLowerCase(), 'The full formal project title must be visibly rendered');
    assert(title.visible && !title.insideDetails && !title.clipped, 'The full project title must be visible outside details, without clipping');
    assert(title.textRuns.every(run => run.visible && run.rects.length && run.rects.every(box =>
      box.width > 0 && box.height > 0 && box.left >= -1 && box.right <= title.width + 1 && box.top >= title.headerBottom - 1 && box.bottom <= title.height + 1
    )), 'Every title line must fit fully in the screenshot below the sticky header: ' + prefix);
    report.checks.push(prefix + ': full formal H1 title visible outside details and fully covered by the title screenshot');
    saveReport();
  }
  async function captureEvidenceDetails(page, prefix) {
    const table = await page.locator('.transfer-evidence .table-block').count() ? '.transfer-evidence .table-block' : '.transfer-evidence .table-wrap';
    if (await page.locator(table).count()) await capture(page, prefix + '-transfer-table', table);
    const method = page.locator('#method details.explanation');
    if (await method.count()) {
      const wasOpen = await method.evaluate(details => details.open);
      if (!wasOpen) await method.locator(':scope > summary').click();
      try {
        assert(await method.evaluate(details => details.open), 'Method details must be open before checking the table');
        const methodTable = await method.locator('.table-block').count() ? '#method .table-block' : '#method .table-wrap';
        assert(await page.locator(methodTable).isVisible(), 'Method table must be rendered for the screenshot');
        await capture(page, prefix + '-method-table', methodTable);
      } finally {
        if (!wasOpen) await method.locator(':scope > summary').click();
      }
    }
    const studies = page.locator('.research-study');
    if (await studies.count()) {
      const index = await studies.count() - 1;
      await capture(page, prefix + '-transfer-evaluation', '.research-study >> nth=' + index);
    }
  }
  async function explanations(page, prefix, exercise) {
    if (pagePath !== 'index.html') return;
    progress('Explanation defaults: ' + prefix);
    const navigation = await page.locator('.nav-links').evaluate(container => {
      const bounds = container.getBoundingClientRect();
      return [...container.querySelectorAll('a')].map(link => {
        const box = link.getBoundingClientRect();
        return { text: link.textContent, visible: box.left >= bounds.left - 1 && box.right <= bounds.right + 1 };
      });
    });
    assert(navigation.every(link => link.visible), 'All main navigation entries must fit without clipping: ' + prefix);
    report.checks.push(prefix + ': all main navigation entries fit without clipping');
    const details = page.locator('.explanation');
    const initial = await details.evaluateAll(elements => elements.map(element => ({
      tag: element.tagName, open: element.open,
      summary: element.querySelector(':scope > summary')?.textContent.trim()
    })));
    const entry = { viewport: page.viewportSize(), initial, interactions: [] };
    (report.explanations ||= []).push(entry);
    saveReport();
    assert.equal(initial.length, 7, 'Homepage must retain author, benchmark, transfer, method, generation, randomization and usage explanations');
    assert(initial.every(item => item.tag === 'DETAILS' && item.open === false && item.summary), 'Explanations must be native details, initially closed, with a summary');
    report.checks.push(prefix + ': all ' + initial.length + ' native explanation details default closed');
    if (!exercise) return;
    for (let index = 0; index < initial.length; index++) {
      const item = details.nth(index);
      const summary = item.locator(':scope > summary');
      const body = item.locator(':scope > .explanation-body');
      const label = prefix + '-explanation-' + (index + 1);
      const interaction = { summary: initial[index].summary, actions: [] };
      entry.interactions.push(interaction);
      progress('Explanation interactions: ' + label + ' ' + initial[index].summary);
      assert.equal(await body.isVisible(), false, 'Closed explanation content must be hidden');
      for (const [action, open] of [['click-open', true], ['click-close', false], ['keyboard-enter-open', true], ['keyboard-space-close', false]]) {
        if (action.startsWith('click')) await summary.click();
        else {
          await summary.focus();
          assert(await summary.evaluate(element => document.activeElement === element), 'Native summary must receive keyboard focus');
          await summary.press(open ? 'Enter' : 'Space');
        }
        const state = { action, open: await item.evaluate(element => element.open), bodyVisible: await body.isVisible() };
        interaction.actions.push(state);
        saveReport();
        assert.equal(state.open, open, label + ' ' + action + ' must toggle the native details');
        assert.equal(state.bodyVisible, open, label + ' ' + action + ' must toggle the explanation content');
        await layout(page, label + '-' + action);
        if (action === 'click-open') {
          if (await item.getAttribute('id') === 'about') await capture(page, prefix + '-hero-about-expanded', '#about');
        }
      }
    }
    assert(await details.evaluateAll(elements => elements.every(element => !element.open)), 'Explanation checks must leave all details closed');
    report.checks.push(prefix + ': all ' + initial.length + ' explanations support pointer, Enter and Space toggles without page overflow');
    saveReport();
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
    if (pagePath === 'index.html') {
      assert.deepEqual(await desktop.locator('main > section[id]').evaluateAll(nodes => nodes.map(node => node.id)),
        ['overview','results','method','generation','render','use'], 'Model evaluations must precede method and generation scale');
      assert.match(await desktop.locator('#results').textContent(), /Four models/);
      report.checks.push('Model benchmark → method → generation scale reading order');
    }
    await formalTitle(desktop, 'desktop');
    await explanations(desktop, 'desktop', true);
    await capture(desktop, 'desktop-opening');
    await captureSections(desktop, 'desktop');
    await captureEvidenceDetails(desktop, 'desktop');
    await heroReveal(desktop, 'desktop', true);
    await rerenderComparison(desktop, 'desktop');
    await interactions(desktop);
    const mainTable = pagePath === 'index.html' ? await tableStyle(desktop) : null;
    report.tableStyles = { homepage: mainTable };
    if (pagePath === 'index.html') {
      const tablet = await open({ width: 768, height: 1000 });
      await formalTitle(tablet, 'tablet-768');
      await heroReveal(tablet, 'tablet-768', false);
      await capture(tablet, 'tablet-768-opening');
      await tablet.context().close();
    }
    for (const width of [390, 320]) {
      const mobile = await open({ width, height: 844 });
      await formalTitle(mobile, 'mobile-' + width);
      await explanations(mobile, 'mobile-' + width, width === 390);
      await heroReveal(mobile, 'mobile-' + width, width === 390);
      if (await mobile.locator('#transfer-film').count()) {
        assert((await mobile.locator('#transfer-film').getAttribute('data-src')).endsWith('/domains-robots-mobile.mp4'), 'Mobile must use a readable vertical synchronized comparison');
        assert.match(await mobile.locator('#transfer figcaption').textContent(), /PIPER.*UR5e/, 'Both robots need visible source/target labels');
        assert(await mobile.locator('[data-expand="transfer-film"]').isVisible(), 'The mobile pair needs a larger-view entry');
      }
      await capture(mobile, 'mobile-' + width + '-opening');
      await captureSections(mobile, 'mobile-' + width);
      await captureEvidenceDetails(mobile, 'mobile-' + width);
      await rerenderComparison(mobile, 'mobile-' + width);
      if (width === 390) await responsivePairs(mobile);
      await mobile.context().close();
    }
    const active = await open({ width: 1440, height: 1000 }, 'no-preference');
    progress('Playback checks start');
    if (await active.locator('#hero-film').count()) {
      await position(active, '#hero-film');
      await active.waitForFunction(() => document.getElementById('hero-film').hasAttribute('src'));
      await active.locator('#hero-film').evaluate(async video => { await video.play(); });
      await active.locator('#hero-film').evaluate(video => { video.currentTime = 8.95; });
      await active.waitForFunction(() => document.querySelector('[data-seek="9.25"]').getAttribute('aria-pressed') === 'true' && document.getElementById('hero-film').currentTime > 9.5);
      report.checks.push('Automatic playback reaches the folding chapter without a task switch');
      const expectedSource = new URL('assets/studio26/experience-loop.mp4', origin).href;
      await active.waitForFunction(source => {
        const video = document.getElementById('hero-film');
        return video.currentSrc === source && video.readyState >= 3 && !video.paused && video.currentTime > 0.1;
      }, expectedSource);
      assert.equal(await active.locator('video').evaluateAll(videos => videos.filter(video => !video.paused).length), 1);
      await active.locator('[data-expand="hero-film"]').click();
      await active.waitForFunction(source => {
        const video = document.getElementById('expanded-film');
        return video.currentSrc === source && video.readyState >= 3 && !video.paused && video.currentTime > 0.1;
      }, expectedSource);
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
    if(pagePath==='index.html') {
      const manifest=await (await desktop.request.get(new URL('assets/studio26/manifest.json',origin).href)).json();
      const conditions=manifest.films['domains-robots'].evidence.conditions;
      assert.equal(conditions.length,2);
      assert.deepEqual(conditions.map(c=>c.layout.background),['isaac_office','isaac_warehouse']);
      assert(conditions.every(c=>c.layout.visual_profile==='train' && c.base_seed===20260911 && c.protocol_identity.protocol_sha256));
      report.checks.push('Paper train-profile scene identities, seeds and protocol hashes recorded');
      for(const width of [1440,390]) {
        const atlas=await open({width,height:1000},'reduce','explore.html#atlas-panel');
        await atlas.waitForFunction(()=>document.querySelectorAll('.atlas-hit').length===500 && document.getElementById('atlas-image').naturalWidth===6400);
        const catalog=await (await atlas.request.get(new URL('assets/catalog/catalog.json?v=26',origin).href)).json();
        for(const index of [0,499]) {
          await atlas.locator('.atlas-hit').nth(index).click();
          assert.equal((await atlas.locator('#atlas-asset-id').textContent()).trim(),catalog.atlas.assets[index].asset_id);
          assert.equal(await atlas.locator('.atlas-hit').nth(index).getAttribute('aria-pressed'),'true');
        }
        await capture(atlas,'atlas-'+width+'-neutral','#atlas-panel');
        await atlas.locator('#atlas-zoom-in').click();
        assert.match(await atlas.locator('#atlas-zoom-label').textContent(),/1.5/);
        await atlas.locator('#atlas-reset').click();
        await atlas.context().close();
      }
      report.checks.push('Neutral 500-asset atlas: first/last pointer selections and zoom work on desktop/mobile');
    }
    if (pagePath === 'index.html') {
      for (const width of [1440, 390]) {
        const study = await open({width, height:1000}, 'reduce', 'index.html?theme=light');
        const title = await study.locator('h1').innerText();
        await capture(study, 'theme-light-' + width + '-title');
        await capture(study, 'theme-light-' + width + '-scale', '#hero-film', 26);
        const source = await study.locator('#hero-film').evaluate(video => video.currentSrc);
        await study.locator('[data-theme-choice="dark"]').click();
        assert.equal(await study.locator('html').getAttribute('data-theme'), 'dark');
        assert.equal(await study.locator('[data-theme-choice="dark"]').getAttribute('aria-pressed'), 'true');
        assert.equal(await study.locator('h1').innerText(), title);
        const media = await study.locator('#hero-film').evaluate(video => ({source:video.currentSrc,time:video.currentTime,paused:video.paused}));
        assert.equal(media.source, source);
        assert(Math.abs(media.time - 26) < 0.15 && media.paused, 'Theme choice must preserve paused media');
        await capture(study, 'theme-dark-' + width + '-scale', '#hero-film', 26);
        await capture(study, 'theme-dark-' + width + '-title');
        await capture(study, 'theme-dark-' + width + '-results', '#results');
        const resultColors = await study.locator('.metric-name').first().evaluate(element => ({text:getComputedStyle(element).color,panel:getComputedStyle(element.closest('.chart-pair')).backgroundColor}));
        assert.equal(resultColors.text, 'rgb(237, 242, 238)');
        assert.equal(resultColors.panel, 'rgb(29, 39, 33)');
        await capture(study, 'theme-dark-' + width + '-method', '#method');
        await capture(study, 'theme-dark-' + width + '-code', '#use');
        await study.locator('[data-theme-choice="light"]').click();
        assert.equal(await study.locator('html').getAttribute('data-theme'), 'light');
        await study.context().close();
      }
      report.checks.push('Same-content light/dark study on desktop/mobile: media time preserved, charts styled, themes reversible');
    }
    if (pagePath === 'index.html') {
      const previewManifest = await (await desktop.request.get(new URL('assets/studio27/manifest.json', origin).href)).json();
      assert.equal(previewManifest.environments.dark.global_lights, 0);
      assert.equal(previewManifest.environments.dark.overhead_rect_lights, 384);
      for (const width of [1440, 390]) {
        const scene = await open({width, height:1000}, 'reduce', 'scene-preview-v27.html?scene=dark&theme=dark');
        assert.equal(await scene.locator('main > section[id]').count(), 6);
        for (const environment of ['white', 'dark']) {
          await scene.locator('[data-scene-choice="' + environment + '"]').click();
          await scene.locator('[data-theme-choice="' + (environment === 'white' ? 'light' : 'dark') + '"]').click();
          for (const view of ['near', 'bridge', 'scale']) {
            await scene.locator('[data-scene-view="' + view + '"]').click();
            await scene.waitForFunction(expected => {
              const image = document.getElementById('scene-frame');
              return image.currentSrc.endsWith(expected) && image.complete && image.naturalWidth === 2560;
            }, '/assets/studio27/' + environment + '-' + view + '.jpg');
            await capture(scene, 'scene-' + width + '-' + environment + '-' + view, '.scene-study');
            if (view === 'near') await capture(scene, 'scene-' + width + '-' + environment + '-opening');
          }
        }
        await scene.locator('[data-scene-view="motion"]').click();
        await scene.waitForFunction(() => {const v=document.getElementById('hero-film');return v.readyState>=3 && !v.paused && v.currentTime>.15;});
        await capture(scene, 'scene-' + width + '-dark-motion', '#hero-film', 1.1);
        await scene.locator('[data-scene-choice="white"]').click();
        await scene.waitForFunction(() => {const v=document.getElementById('hero-film');return v.currentSrc.endsWith('/white-handoff.mp4') && v.readyState>=3 && Math.abs(v.currentTime-1.1)<.1 && v.paused;});
        const excerpt = await scene.locator('#hero-film').evaluate(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration}));
        assert.equal(excerpt.width,1920);assert.equal(excerpt.height,1080);assert(Math.abs(excerpt.duration-2.4)<.05);
        await capture(scene, 'scene-' + width + '-white-motion', '#hero-film', 1.1);
        await scene.locator('[data-scene-view="near"]').click();
        assert(await scene.locator('#hero-film').isHidden());
        assert(await scene.locator('#scene-frame').isVisible());
        await scene.context().close();
      }
      report.checks.push('In-page scene study: full homepage context, three uncropped views, top lights only, real 25fps excerpt, matched paused time and desktop/mobile layout');
    }
    if (pagePath === 'index.html') {
      const fitManifest=await (await desktop.request.get(new URL('assets/studio28/manifest.json',origin).href)).json();
      for (const [width,height] of [[1920,1080],[1440,900],[1366,768],[1280,720],[768,1024],[390,844],[320,640]]) {
        const fit=await open({width,height},'reduce','scene-preview-v28.html');
        await fit.locator('#scene-frame').evaluate(image=>image.decode());
        const bounds=await fit.evaluate(()=>{
          const box=selector=>{const r=document.querySelector(selector).getBoundingClientRect();return {top:r.top,bottom:r.bottom,width:r.width,height:r.height};};
          return {viewportWidth:document.documentElement.clientWidth,header:box('.site-header'),hero:box('#overview'),title:box('h1'),caption:box('.hero-figure figcaption'),authors:box('#authors>summary'),frame:box('#scene-frame'),background:getComputedStyle(document.body).backgroundColor,fit:getComputedStyle(document.getElementById('scene-frame')).objectFit};
        });
        assert(bounds.hero.bottom<=height+1 && bounds.authors.bottom<=height+1 && bounds.caption.bottom<=height+1,'Whole hero must fit one viewport at '+width+'x'+height);
        assert(bounds.title.bottom<bounds.frame.top && bounds.frame.height>100,'Headline and media must remain separate and visible');
        assert(Math.abs(bounds.frame.width-bounds.viewportWidth)<1,'Banner spans full available viewport width');
        assert.equal(bounds.fit,'contain');assert.equal(bounds.background,'rgb(255, 255, 255)');
        const whitePatch=fitManifest.images[(width<=700?'mobile':'wide')+'-near'].white_patch;
        const pixel=await fit.locator('#scene-frame').evaluate((image,point)=>{const c=document.createElement('canvas');c.width=image.naturalWidth;c.height=image.naturalHeight;const context=c.getContext('2d');context.drawImage(image,0,0);return [...context.getImageData(point[0],point[1],1,1).data].slice(0,3);},whitePatch);
        assert(pixel.every(channel=>channel>=253),'Unshadowed image background must match the white page');
        report.layouts.push({label:'one-screen-white-banner',viewport:{width,height},...bounds});
        await capture(fit,'fit-'+width+'x'+height+'-opening');
        if (width===1280 || width===390) {
          await fit.locator('[data-scene-view="scale"]').click();
          await fit.waitForFunction(()=>document.getElementById('scene-frame').currentSrc.endsWith('/wide-scale.jpg') && document.getElementById('scene-frame').complete);
          await capture(fit,'fit-'+width+'-scale');
          await fit.locator('[data-scene-view="motion"]').click();
          await fit.waitForFunction(()=>{const v=document.getElementById('hero-film');return v.readyState>=3 && !v.paused && v.currentTime>.15;});
          await capture(fit,'fit-'+width+'-motion','#hero-film',1);
          const size=await fit.locator('#hero-film').evaluate(v=>[v.videoWidth,v.videoHeight]);
          assert.deepEqual(size,width<=700?[1080,810]:[2400,720]);
          if(width===1280){
            await fit.setViewportSize({width:390,height:844});
            await fit.waitForFunction(()=>{const v=document.getElementById('hero-film');return v.currentSrc.endsWith('/mobile-handoff.mp4') && v.readyState>=3 && v.paused && Math.abs(v.currentTime-1)<.1;});
            await fit.setViewportSize({width,height});
            await fit.waitForFunction(()=>{const v=document.getElementById('hero-film');return v.currentSrc.endsWith('/wide-handoff.mp4') && v.readyState>=3 && v.paused && Math.abs(v.currentTime-1)<.1;});
          }
        }
        await fit.context().close();
      }
      report.checks.push('White panoramic banner: entire closed hero fits seven viewport sizes, full width, uncropped source, pure-white page and real desktop/mobile motion');
    }
    if(pagePath==='index.html'){
      const islands=await(await desktop.request.get(new URL('assets/studio29/manifest.json',origin).href)).json();
      assert.deepEqual(islands.stages,[2,8,48,384]);
      assert.equal(islands.audit.subject_border_intersections,0);
      assert.equal(islands.images.pair.visible_trajectories,2);
      for(const [width,height] of [[1440,900],[1280,720],[390,844],[320,640]]){
        const scene=await open({width,height},'reduce','scene-preview-v29.html');
        await scene.locator('#scene-frame').evaluate(image=>image.decode());
        assert.match(await scene.locator('#hero-film-caption').textContent(),/Left: Flattening.*Right: Folding/);
        const box=await scene.locator('#overview').boundingBox();assert(box.y+box.height<=height+1);
        await capture(scene,'islands-'+width+'-opening');
        for(const [view,file] of [['bridge','broader.jpg'],['scale','scale.jpg']]){
          await scene.locator('[data-scene-view="'+view+'"]').click();
          await scene.waitForFunction(expected=>{const image=document.getElementById('scene-frame');return image.currentSrc.endsWith(expected)&&image.complete;},'/assets/studio29/'+file);
          await capture(scene,'islands-'+width+'-'+view);
        }
        if(width===1440 || width===390){
          await scene.locator('[data-scene-view="motion"]').click();
          await scene.waitForFunction(()=>{const v=document.getElementById('hero-film');return v.readyState>=3&&!v.paused&&v.currentTime>.1;});
          for(const time of [.1,1.8])await capture(scene,'islands-'+width+'-reveal-'+String(time).replace('.','p'),'#hero-film',time);
          const media=await scene.locator('#hero-film').evaluate(v=>({duration:v.duration,loop:v.loop}));
          assert(Math.abs(media.duration-2.8)<.01);assert.equal(media.loop,false,'Partial reveal sample must not advertise a seamless full loop');
        }
        await scene.context().close();
      }
      report.checks.push('Finite-workcell study: centred two-task opening, 48/384 complete-group views, one-screen desktop/mobile and a real whole-group reveal excerpt');
    }
    if(pagePath==='index.html'){
      for(const [width,height] of [[1440,900],[1280,720],[390,844],[320,640]]){
        const page=await open({width,height},'reduce','research-preview-v30.html');
        await page.locator('#scene-frame').evaluate(image=>image.decode());
        const hero=await page.locator('#overview').boundingBox();assert(hero.y+hero.height<=height+1,'Centred title and hero must fit the viewport');
        await capture(page,'research30-'+width+'-opening');
        await capture(page,'research30-'+width+'-results','#results');
        await capture(page,'research30-'+width+'-rgb','#rgb-film',24);
        const video=page.locator('#rgb-film');
        const state=await video.evaluate(v=>({src:v.currentSrc,width:v.videoWidth,height:v.videoHeight,duration:v.duration,loop:v.loop,paused:v.paused}));
        assert.equal(state.width,width<=700?960:1920);assert.equal(state.height,720);assert.equal(state.duration,72);assert(state.loop&&state.paused);
        if(width===1440){
          await capture(page,'research30-rgb-first','#rgb-film',0);
          await capture(page,'research30-rgb-last','#rgb-film',71.96);
          await capture(page,'research30-method','#method');
          await page.locator('[data-expand="rgb-film"]').click();
          await page.waitForFunction(()=>document.getElementById('expanded-film').readyState>=3);
          assert.match(await page.locator('#media-title').textContent(),/RGB dataset cam_high/);
          await page.locator('#close-media').click();
        }
        await page.context().close();
      }
      const moving=await open({width:1440,height:900},'no-preference','research-preview-v30.html');
      await position(moving,'#rgb-film');
      await moving.waitForFunction(()=>{const v=document.getElementById('rgb-film');return !v.paused&&v.currentTime>.2;});
      await moving.locator('#rgb-film').evaluate(v=>v.currentTime=10);
      await moving.setViewportSize({width:390,height:844});
      await moving.waitForFunction(()=>{const v=document.getElementById('rgb-film');return v.videoWidth===960&&!v.paused&&v.currentTime>=9.9;});
      await moving.context().close();
      report.checks.push('Research UI and original cam_high ribbon: one-screen title, consistent evidence sections, matched mobile clock, autoplay/reduced-motion, dialog and 72-second media loop');
    }
    if(pagePath==='index.html'){
      for(const [width,height] of [[1440,900],[1280,720],[390,844],[320,640]]){
        const page=await open({width,height},'reduce','research-preview-v31.html');
        const hero=await page.locator('#overview').boundingBox();assert(hero.y+hero.height<=height+1);
        if(width===1440){
          const source=await(await page.request.get(new URL('assets/studio35/manifest.json',origin).href)).json();
          assert.equal(source.workspaces,384);assert.equal(source.unique_trajectories,386);assert.equal(source.cases_per_workspace,2);
          assert(source.audit.foreground_does_not_reset_within_a_case&&source.audit.every_workspace_changes_episode&&source.audit.matched_layer_camera_and_state_hashes);
        }
        assert.deepEqual(await page.locator('main > section').evaluateAll(nodes=>nodes.slice(0,5).map(n=>n.id)),['overview','rgb','diversity','results','method']);
        await capture(page,'generation31-'+width+'-opening');
        const times=width===1440?[0,7,9,11.5,15,19,22.5,23.8,24.5,31,34,36,46,47.96]:width===390?[0,9,22.5,24.5,34,46,47.96]:[0];
        for(const time of times)await capture(page,'generation31-'+width+'-'+String(time).replace('.','p'),'#hero-film',time);
        const state=await page.locator('#hero-film').evaluate(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration,loop:v.loop,paused:v.paused}));
        assert.equal(state.width,width<=700?1200:2400);assert.equal(state.height,width<=700?900:840);assert.equal(state.duration,48);assert(state.loop&&state.paused);
        assert.equal(await page.locator('#hero-film').evaluate(v=>v.controls),false);
        assert.equal(await page.locator('[data-gen-seek], [data-expand="hero-film"], #overview input[type="range"]').count(),0);
        await page.locator('#hero-toggle').click();
        await page.waitForFunction(()=>!document.getElementById('hero-film').paused);
        await page.locator('#hero-toggle').click();
        assert(await page.locator('#hero-film').evaluate(v=>v.paused));
        await page.context().close();
      }
      for(const [width,height] of [[1920,1080],[1366,768]]){
        const page=await open({width,height},'reduce','research-preview-v31.html');
        const hero=await page.locator('#overview').boundingBox();assert(hero.y+hero.height<=height+1);
        await capture(page,'generation35-'+width+'-opening');await page.context().close();
      }
      const live=await open({width:1440,height:900},'no-preference','research-preview-v31.html');
      await live.waitForFunction(()=>{const v=document.getElementById('hero-film');return !v.paused&&v.currentTime>.3;});
      await live.locator('#hero-film').evaluate(v=>v.currentTime=47.8);
      await live.waitForFunction(()=>{const v=document.getElementById('hero-film');return !v.paused&&v.currentTime<1;});
      await position(live,'#rgb-film');
      await live.waitForFunction(()=>{const v=document.getElementById('rgb-film');return !v.paused&&v.currentTime>.3&&document.getElementById('hero-film').paused;});
      const rgbStart=await live.locator('#rgb-film').evaluate(v=>v.currentTime);
      await live.waitForFunction(start=>document.getElementById('rgb-film').currentTime>start+.5,rgbStart);
      await position(live,'#hero-film');
      await live.waitForFunction(()=>!document.getElementById('hero-film').paused&&document.getElementById('rgb-film').paused);
      await live.setViewportSize({width:390,height:844});
      await live.waitForFunction(()=>{const v=document.getElementById('hero-film');return v.videoWidth===1200&&!v.paused;});
      await position(live,'#rgb-film');
      await live.waitForFunction(()=>!document.getElementById('rgb-film').paused&&document.getElementById('hero-film').paused);
      const mobileRgbStart=await live.locator('#rgb-film').evaluate(v=>v.currentTime);
      await live.waitForFunction(start=>document.getElementById('rgb-film').currentTime>start+.5,mobileRgbStart);
      await live.context().close();
      report.checks.push('Two-case ambient hero: 48s loop, 20:7 desktop framing, direct case transitions, no seek controls, play/pause and immediate handoff to RGB');
    }
    if(pagePath==='index.html'){
      const atlas=await(await desktop.request.get(new URL('assets/atlas32/manifest.json',origin).href)).json();
      assert.equal(atlas.asset_count,500);assert(atlas.audit.subject_pixels_unchanged&&atlas.audit.pure_white_border);
      for(const [width,height] of [[1440,900],[390,844]]){
        const page=await open({width,height},'reduce','research-preview-v31.html');
        const image=page.locator('.atlas-overview img');
        assert.equal(await page.locator('#diversity a').count(),0,'Asset visualization is presentation-only');
        await image.scrollIntoViewIfNeeded();await image.evaluate(img=>img.decode());
        const style=await image.evaluate(img=>{
          const box=img.getBoundingClientRect(),parent=img.parentElement.getBoundingClientRect(),s=getComputedStyle(img.parentElement);
          const canvas=document.createElement('canvas');canvas.width=img.naturalWidth;canvas.height=1;
          const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0);
          return {src:img.currentSrc,naturalWidth:img.naturalWidth,width:box.width,height:box.height,parentWidth:parent.width,parentHeight:parent.height,background:s.backgroundColor,radius:s.borderRadius,whiteEdge:[...ctx.getImageData(0,0,canvas.width,1).data].every(n=>n===255)};
        });
        assert(style.src.endsWith('/assets/atlas32/garment-atlas.webp'));assert.equal(style.naturalWidth,3200);assert(style.whiteEdge);
        assert.equal(style.background,'rgb(255, 255, 255)');assert.equal(style.radius,'0px');assert(Math.abs(style.width/style.height-16/9)<.001);
        assert(Math.abs(style.width-style.parentWidth)<1&&Math.abs(style.height-style.parentHeight)<1);
        await capture(page,'atlas32-'+width+'-section','#diversity');
        await capture(page,'atlas32-'+width+'-image','.atlas-overview');
        await page.context().close();
      }
      report.checks.push('White asset atlas: 500 real assets, retained subject pixels/contact shadows, decoded pure-white edge, no card border or mobile crop');
    }
    if(pagePath==='index.html'){
      for(const width of [1440,390]){
        const page=await open({width,height:900},'reduce','research-preview-v31.html');
        await capture(page,'rgb34-'+width+'-collection','#rgb-film',16);
        const state=await page.locator('#rgb-film').evaluate(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration,controls:v.controls,loop:v.loop}));
        const rgbBounds=await page.locator('#rgb-film').boundingBox();
        assert(rgbBounds.width<=1120&&rgbBounds.x>0,'RGB collection has a bounded width and side margins');
        assert(Math.abs(rgbBounds.x-(width-rgbBounds.width)/2)<1,'RGB collection is centered');
        assert.equal(state.width,width<=700?640:1920);assert.equal(state.height,720);assert.equal(state.duration,64);assert(state.loop&&!state.controls);
        assert.equal(await page.locator('[data-expand]').count(),0,'Showcase media has no expand/scrub interface');
        await page.locator('[data-ambient-toggle="rgb-film"]').click();
        await page.waitForFunction(()=>!document.getElementById('rgb-film').paused);
        await page.locator('[data-ambient-toggle="rgb-film"]').click();
        assert(await page.locator('#rgb-film').evaluate(v=>v.paused));
        await page.context().close();
      }
      report.checks.push('Larger RGB collection: 96 source episodes, 18 desktop/6 mobile panels, 64s loop, pause-only controls and noninteractive asset display before results');
    }
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
