(() => {
  const params = new URL(location.href).searchParams;
  let environment = params.get('scene') === 'white' ? 'white' : 'dark';
  let view = 'near';
  let revision = 0;
  const image = document.getElementById('scene-frame');
  const film = document.getElementById('hero-film');
  const original = document.getElementById('scene-original');
  const expand = document.getElementById('scene-expand');
  const caption = document.getElementById('hero-film-caption');
  function update(explicit = false) {
    const version = ++revision;
    const moving = view === 'motion';
    const wasPlaying = !film.paused;
    const position = film.currentTime;
    film.pause();
    film.hidden = !moving;
    image.hidden = moving;
    expand.hidden = !moving;
    original.hidden = moving;
    document.querySelectorAll('[data-scene-choice]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.sceneChoice === environment)));
    document.querySelectorAll('[data-scene-view]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.sceneView === view)));
    const prefix = 'assets/studio27/' + environment;
    if (moving) {
      const src = prefix + '-handoff.mp4';
      film.dataset.src = src;
      film.poster = prefix + '-bridge.jpg';
      const play = () => { if (explicit || wasPlaying) film.play().catch(() => {}); };
      if (film.getAttribute('src') !== src) {
        film.addEventListener('loadedmetadata', () => {
          if (version !== revision) return;
          film.currentTime = Math.min(position, Math.max(0, film.duration - .05));
          play();
        }, {once:true});
        film.src = src;
        film.load();
      } else play();
    } else {
      image.src = prefix + '-' + view + '.jpg';
      original.href = image.src;
      image.alt = (environment === 'dark' ? 'Workcell overhead lights only, without global light' : 'Pure-white studio') + ', matte cotton garments, ' + view + ' view';
    }
    caption.textContent = (environment === 'dark' ? 'Workstation lights only' : 'White studio') + ' · Matte cotton · ' + (moving ? '2.4 s excerpt / 25 fps / 2× task speed · Folding begins before camera arrival' : '384 saved trajectories / 202 assets');
    const url = new URL(location.href);
    url.searchParams.set('scene', environment);
    history.replaceState(null, '', url);
  }
  document.querySelectorAll('[data-scene-choice]').forEach(button => button.addEventListener('click', () => {environment=button.dataset.sceneChoice;update();}));
  document.querySelectorAll('[data-scene-view]').forEach(button => button.addEventListener('click', () => {view=button.dataset.sceneView;update(view === 'motion');}));
  update();
})();
