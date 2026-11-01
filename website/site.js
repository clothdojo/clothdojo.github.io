(() => {
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const narrow = matchMedia('(max-width: 700px)');
  const films = [...document.querySelectorAll('video[data-src]')];
  const visible = new Map();
  let active;
  const versions = new WeakMap();
  const variantLoading = new WeakMap();
  const resizePaused = new WeakSet();
  const comparisons = films.filter(video => video.dataset.mobileSrc).map(video => ({
    video, desktopSrc: video.dataset.src, desktopPoster: video.poster
  }));
  function attach(video) {
    if (!video.hasAttribute('src')) { video.src = video.dataset.src; video.load(); }
  }
  function play(video) {
    attach(video);
    document.querySelectorAll('video').forEach(other => { if (other !== video) other.pause(); });
    active = video;
    video.play().catch(() => {});
  }
  function ambient() {
    if (document.hidden || motion.matches || navigator.connection?.saveData || document.getElementById('media-dialog').open) return;
    if (active && !active.paused && !active.hasAttribute('data-auto') && visible.has(active)) return;
    const selected = [...visible].filter(([v]) => v.hasAttribute('data-auto') && !variantLoading.has(v) && !resizePaused.has(v)).sort((a,b) => b[1]-a[1])[0]?.[0];
    if (selected && (active !== selected || selected.paused)) play(selected);
  }
  function choose(button) {
    const video = document.getElementById(button.dataset.select);
    if (!video) return;
    const version = (versions.get(video) || 0) + 1;
    versions.set(video, version);
    video.pause();
    document.querySelectorAll('[data-select="' + video.id + '"]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    video.dataset.src = button.dataset.src;
    video.poster = button.dataset.poster;
    video.setAttribute('aria-label', button.textContent.trim() + ': ' + button.dataset.caption);
    document.getElementById(video.id + '-caption').textContent = button.dataset.caption;
    video.addEventListener('loadedmetadata', () => {
      if (versions.get(video) !== version) return;
      if (visible.has(video) && !motion.matches && !navigator.connection?.saveData && !document.hidden && !document.getElementById('media-dialog').open) play(video);
    }, { once: true });
    video.src = video.dataset.src;
    video.load();
  }
  document.querySelectorAll('[data-select]').forEach(button => button.addEventListener('click', () => choose(button)));
  const hero = document.getElementById('hero-film');
  const chapters = [...document.querySelectorAll('[data-seek]')];
  const domainFilm = document.getElementById('transfer-film');
  const domainCaption = document.getElementById('domain-scene-caption');
  if (domainFilm && domainCaption) domainFilm.addEventListener('timeupdate', () => {
    domainCaption.textContent = (domainFilm.currentTime < 12 ? 'Office' : 'Warehouse') + ' · PIPER ↔ UR5e · 2× replay';
  });
  if (hero && chapters.length) {
    function chapterLabel() {
      const time = hero.currentTime;
      const chapter = time < 9.25 ? 0 : time < 19.5 ? 1 : 2;
      chapters.forEach((button, index) => button.setAttribute('aria-pressed', String(index === chapter)));
      const label = time < 6.76 ? 'Flattening · 2× task replay' : time < 8.25 ? 'Return to home · Kinematic presentation' :
        time < 9.25 ? 'Moving to folding' : time < 18.01 ? 'Folding · 2× task replay' : time < 19.5 ?
        'Return to home · Kinematic presentation' : '96 independent trajectories · 63 garments · 2× replay';
      document.getElementById('hero-film-caption').textContent = label;
    }
    hero.addEventListener('timeupdate', chapterLabel);
    chapters.forEach(button => button.addEventListener('click', () => {
      const seek = () => { hero.currentTime = Number(button.dataset.seek); chapterLabel(); play(hero); };
      if (hero.readyState) seek();
      else { hero.addEventListener('loadedmetadata', seek, { once: true }); attach(hero); }
    }));
  }
  function comparisonVariants() {
    for (const { video, desktopSrc, desktopPoster } of comparisons) {
      const src = narrow.matches ? video.dataset.mobileSrc : desktopSrc;
      const poster = narrow.matches ? video.dataset.mobilePoster : desktopPoster;
      if (video.dataset.src === src) continue;
      const pending = variantLoading.get(video);
      const wasPlaying = pending?.wasPlaying ?? !video.paused;
      const position = pending?.position ?? video.currentTime;
      const attached = video.hasAttribute('src');
      video.dataset.src = src;
      video.poster = poster;
      if (!attached) continue;
      const version = (versions.get(video) || 0) + 1;
      versions.set(video, version);
      variantLoading.set(video, { position, wasPlaying });
      if (!wasPlaying && active === video) resizePaused.add(video);
      video.pause();
      video.addEventListener('loadedmetadata', () => {
        if (versions.get(video) !== version) return;
        variantLoading.delete(video);
        if (Number.isFinite(video.duration)) video.currentTime = Math.min(position, Math.max(0, video.duration - .05));
        if (wasPlaying && visible.has(video) && !motion.matches && !navigator.connection?.saveData &&
            !document.hidden && !document.getElementById('media-dialog').open && (!active || active === video)) play(video);
        else ambient();
      }, { once: true });
      video.src = src;
      video.load();
    }
  }
  comparisonVariants();
  narrow.addEventListener('change', comparisonVariants);
  document.addEventListener('play', event => {
    if (event.target.tagName !== 'VIDEO' || event.target.paused) return;
    active = event.target;
    resizePaused.delete(active);
    document.querySelectorAll('video').forEach(v => { if (v !== active) v.pause(); });
  }, true);
  const observer = new IntersectionObserver(entries => {
    for (const { target, isIntersecting, intersectionRatio } of entries) {
      if (isIntersecting && intersectionRatio >= .2) { visible.set(target, intersectionRatio); attach(target); }
      else { visible.delete(target); resizePaused.delete(target); target.pause(); if (active === target) active = undefined; }
    }
    ambient();
  }, { threshold: [0,.2,.5,1] });
  films.forEach(v => observer.observe(v));
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { document.querySelectorAll('video').forEach(v => v.pause()); active = undefined; }
    else ambient();
  });
  motion.addEventListener('change', () => {
    if (motion.matches) { films.forEach(v => v.pause()); active = undefined; }
    else ambient();
  });
  const dialog = document.getElementById('media-dialog');
  const expanded = document.getElementById('expanded-film');
  document.querySelectorAll('[data-expand]').forEach(button => button.addEventListener('click', () => {
    const source = document.getElementById(button.dataset.expand);
    document.getElementById('media-title').textContent = source.getAttribute('aria-label');
    expanded.dataset.src = source.dataset.src;
    expanded.src = source.dataset.src;
    expanded.load();
    dialog.showModal();
    play(expanded);
  }));
  document.getElementById('close-media').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    expanded.pause(); expanded.removeAttribute('src'); expanded.load();
    if (active === expanded) active = undefined;
  });
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();
  });
  document.getElementById('copy-commands').addEventListener('click', async () => {
    const status = document.getElementById('copy-status');
    try { await navigator.clipboard.writeText(document.getElementById('commands').textContent); status.textContent = 'Copied'; }
    catch { status.textContent = 'Select the commands above to copy.'; }
  });
})();
