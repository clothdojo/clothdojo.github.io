const collection = document.getElementById('collection-video');
const detail = document.getElementById('detail-video');
const players = [collection, detail];
const motion = matchMedia('(prefers-reduced-motion: reduce)');
const dialog = document.getElementById('replay-dialog');
const replay = document.getElementById('replay-video');
let selected = collection;
let visible = false;
let pendingTime = 0;
function load(video) {
  if (!video.hasAttribute('src')) { video.src = video.dataset.src; video.load(); }
}
function play(video) { load(video); video.play().catch(() => {}); }
function ambient() {
  if (visible && !document.hidden && !dialog.open && !motion.matches && !navigator.connection?.saveData) play(selected);
}
document.addEventListener('play', event => {
  if (event.target.tagName === 'VIDEO') {
    document.querySelectorAll('video').forEach(video => { if (video !== event.target) video.pause(); });
  }
}, true);
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
  players.forEach(video => { video.pause(); video.hidden = true; });
  selected = button.dataset.view === 'collection' ? collection : detail;
  selected.hidden = false;
  document.querySelectorAll('[data-view]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  document.getElementById('view-caption').textContent = selected === collection
    ? '200 distinct saved trajectories, spanning 169 garments. Flattening in the top half; folding in the bottom half.'
    : 'Flattening / Folding. Separate autonomous rollout excerpts with matching external camera and appearance; not full-task success evidence.';
  document.getElementById('viewer-status').textContent = selected === collection ? 'Recorded excerpts · 1×' : 'External-camera excerpts · 1×';
  load(selected);
  ambient();
}));
new IntersectionObserver(entries => {
  visible = entries[0].isIntersecting && entries[0].intersectionRatio >= .3;
  if (visible) { load(selected); ambient(); } else players.forEach(video => video.pause());
}, {threshold:[0,.3]}).observe(document.querySelector('.film-surface'));
document.addEventListener('visibilitychange', () => { if (document.hidden) document.querySelectorAll('video').forEach(video => video.pause()); });
motion.addEventListener('change', event => { if (event.matches) players.forEach(video => video.pause()); });
document.querySelectorAll('[data-time]').forEach(button => button.addEventListener('click', () => {
  pendingTime = Number(button.dataset.time);
  players.forEach(video => video.pause());
  dialog.showModal();
  if (!replay.hasAttribute('src')) { replay.src = 'assets/catalog/case-00.mp4'; replay.load(); }
  if (replay.readyState >= 1) { replay.currentTime = pendingTime; replay.play().catch(() => {}); }
}));
replay.addEventListener('loadedmetadata', () => {
  if (dialog.open) { replay.currentTime = pendingTime; replay.play().catch(() => {}); }
});
document.getElementById('close-replay').addEventListener('click', () => dialog.close());
dialog.addEventListener('close', () => { replay.pause(); replay.removeAttribute('src'); replay.load(); });
dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
});
