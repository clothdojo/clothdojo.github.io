(() => {
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const narrow = matchMedia('(max-width: 700px)');
  const films = [...document.querySelectorAll('video[data-src]')];
  const visible = new Map();
  let active;
  const versions = new WeakMap();
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
    if (active || document.hidden || motion.matches || navigator.connection?.saveData || document.getElementById('media-dialog').open) return;
    const selected = [...visible].filter(([v]) => v.hasAttribute('data-auto')).sort((a,b) => b[1]-a[1])[0]?.[0];
    if (selected) play(selected);
  }
  function choose(button, autoplay = true) {
    const video = document.getElementById(button.dataset.select);
    if (!video) return;
    const position = video.hasAttribute('data-sync-time') ? video.currentTime : 0;
    const version = (versions.get(video) || 0) + 1;
    versions.set(video, version);
    video.pause();
    document.querySelectorAll('[data-select="' + video.id + '"]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    video.dataset.src = button.dataset.src;
    video.poster = button.dataset.poster;
    if (button.dataset.layout) video.dataset.layout = button.dataset.layout;
    video.setAttribute('aria-label', button.dataset.caption);
    document.getElementById(video.id + '-caption').textContent = button.dataset.caption;
    video.addEventListener('loadedmetadata', () => {
      if (versions.get(video) !== version) return;
      if (position && Number.isFinite(video.duration)) video.currentTime = Math.min(position, video.duration - .05);
      if (autoplay && visible.has(video) && !motion.matches && !navigator.connection?.saveData && !document.hidden && !document.getElementById('media-dialog').open) play(video);
    }, { once: true });
    video.src = video.dataset.src;
    video.load();
  }
  document.querySelectorAll('[data-select]').forEach(button => button.addEventListener('click', () => choose(button)));
  function mobileCollection() {
    if (!narrow.matches) return;
    for (const id of ['collection-film', 'transfer-film']) {
      const options = [...document.querySelectorAll('[data-select="' + id + '"]')];
      const overview = options.find(b => b.classList.contains('desktop-overview'));
      if (overview?.getAttribute('aria-pressed') === 'true') {
        const first = options.find(b => !b.classList.contains('desktop-overview'));
        if (first) choose(first, false);
      }
    }
  }
  mobileCollection();
  narrow.addEventListener('change', mobileCollection);
  document.addEventListener('play', event => {
    if (event.target.tagName !== 'VIDEO' || event.target.paused) return;
    active = event.target;
    document.querySelectorAll('video').forEach(v => { if (v !== active) v.pause(); });
  }, true);
  const observer = new IntersectionObserver(entries => {
    for (const { target, isIntersecting, intersectionRatio } of entries) {
      if (isIntersecting && intersectionRatio >= .2) { visible.set(target, intersectionRatio); attach(target); }
      else { visible.delete(target); target.pause(); if (active === target) active = undefined; }
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
