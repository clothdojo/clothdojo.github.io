// Reuse the shared player policy for task and robot selection.
(() => {
  const hero = document.getElementById('hero-video');
  const heroHD = document.getElementById('hero-hd');
  function selectVideo(video, src, poster, label) {
    video.pause();
    video.dataset.src = src;
    video.src = src;
    video.poster = poster;
    video.setAttribute('aria-label', label);
    video.load();
    // The selection is explicit; respect motion and data-saving preferences.
    if (!reducedMotion.matches && !connection?.saveData) playExclusive(video);
  }
  document.querySelectorAll('[data-task]').forEach(button => button.addEventListener('click', () => {
    const task = button.dataset.task;
    document.querySelectorAll('[data-task]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    selectVideo(hero, 'assets/curated/hero-' + task + '.mp4', 'assets/curated/hero-' + task + '-poster.jpg',
      'External-camera ' + task + ' excerpt with three independent generated trajectories');
    document.getElementById('hero-caption').textContent = 'One detailed ' + (task === 'fold' ? 'folding' : 'flattening') +
      ' excerpt, alongside three different garments. Each panel replays an independent recorded trajectory at 1×.';
    heroHD.dataset.film = 'assets/' + task + '-hq.mp4';
    heroHD.dataset.title = (task === 'fold' ? 'Folding · Source 12–22 s' : 'Flattening · Source 10–20 s') + ' · External view · 1×';
  }));
  document.querySelectorAll('[data-compare]').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('[data-compare]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    const stem = 'assets/curated/compare-' + button.dataset.compare;
    selectVideo(document.getElementById('embodiment-video'), stem + '.mp4', stem + '-poster.jpg',
      'PIPER versus ' + button.dataset.label + ': flattening above, folding below');
    document.getElementById('target-robot-label').textContent = button.dataset.label + ' · target';
  }));
})();
