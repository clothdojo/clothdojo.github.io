const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const videos = [...document.querySelectorAll('video[data-src]')];
const visibleVideos = new Map();
const connection = navigator.connection;
let activeVideo;

function loadVideo(video) {
  if (!video.getAttribute('src')) { video.src = video.dataset.src; video.load(); }
}
function pauseOthers(selected) {
  document.querySelectorAll('video').forEach(video => {
    if (video !== selected) video.pause();
  });
}
function playExclusive(video) {
  loadVideo(video);
  pauseOthers(video);
  activeVideo = video;
  video.play().catch(() => {});
}
function playVisiblePreview() {
  if (document.hidden || reducedMotion.matches || connection?.saveData || activeVideo) return;
  const selected = [...visibleVideos.entries()]
    .filter(([video, ratio]) => ratio >= .25 && video.hasAttribute('data-autoplay'))
    .sort((a, b) => b[1] - a[1])[0]?.[0];
  if (selected) playExclusive(selected);
}
document.addEventListener('play', event => {
  if (event.target.tagName !== 'VIDEO' || event.target.paused) return;
  activeVideo = event.target;
  pauseOthers(activeVideo);
}, true);
const observer = new IntersectionObserver(entries => {
  entries.forEach(({target: video, isIntersecting, intersectionRatio}) => {
    if (isIntersecting && intersectionRatio >= .25) {
      visibleVideos.set(video, intersectionRatio);
      loadVideo(video);
    } else {
      visibleVideos.delete(video);
      video.pause();
      if (activeVideo === video) activeVideo = undefined;
    }
  });
  playVisiblePreview();
}, { threshold: [0, .25, .5, .75, 1] });
videos.forEach(video => observer.observe(video));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    document.querySelectorAll('video').forEach(video => video.pause());
    activeVideo = undefined;
  } else playVisiblePreview();
});
reducedMotion.addEventListener('change', event => {
  if (event.matches) {
    videos.forEach(video => video.pause());
    activeVideo = undefined;
  } else playVisiblePreview();
});
const filmDialog = document.getElementById('film-dialog');
const film = document.getElementById('film-video');
document.querySelectorAll('[data-film]').forEach(button => button.addEventListener('click', () => {
  pauseOthers(film);
  document.getElementById('film-title').textContent = button.dataset.title;
  film.src = button.dataset.film;
  film.load();
  filmDialog.showModal();
  playExclusive(film);
}));
filmDialog.addEventListener('close', () => {
  film.pause(); film.removeAttribute('src'); film.load();
  if (activeVideo === film) activeVideo = undefined;
});
const figureDialog = document.getElementById('figure-dialog');
document.querySelectorAll('[data-image]').forEach(button => button.addEventListener('click', () => {
  const image = figureDialog.querySelector('img');
  image.src = button.dataset.image;
  image.alt = button.querySelector('img').alt;
  figureDialog.showModal();
}));
document.querySelectorAll('dialog').forEach(dialog => {
  dialog.querySelector('[data-close]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }});
});
document.getElementById('copy')?.addEventListener('click', async () => {
  const status = document.getElementById('copy-status');
  try { await navigator.clipboard.writeText(document.getElementById('commands').textContent); status.textContent = 'Copied'; }
  catch { status.textContent = 'Select the commands above to copy.'; }
});

document.querySelectorAll('details').forEach(detail => detail.addEventListener('toggle', () => {
  detail.querySelectorAll('video[data-src]').forEach(video => {
    if (detail.open) loadVideo(video);
    else video.pause();
  });
}));
const embodimentVideo = document.getElementById('embodiment-video');
const robotButtons = [...document.querySelectorAll('[data-robot-film]')];
robotButtons.forEach(button => button.addEventListener('click', () => {
  robotButtons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  embodimentVideo.pause();
  embodimentVideo.dataset.src = button.dataset.robotFilm;
  embodimentVideo.src = button.dataset.robotFilm;
  embodimentVideo.poster = button.dataset.robotPoster;
  embodimentVideo.setAttribute('aria-label', button.dataset.robotTitle + ' flattening on the left and folding on the right');
  embodimentVideo.load();
  document.getElementById('embodiment-caption').textContent = button.dataset.robotTitle + ' · ' + (button.dataset.robotTitle === 'PIPER' ? 'Recorded trajectory replay' : 'Retargeted kinematic reference');
  const hdButton = document.getElementById('embodiment-hd');
  hdButton.dataset.film = button.dataset.robotHd;
  hdButton.dataset.title = button.dataset.robotTitle + ' · HD flattening / folding';
  playExclusive(embodimentVideo);
}));
