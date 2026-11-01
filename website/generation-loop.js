(() => {
  const film=document.getElementById('hero-film');
  const button=document.getElementById('hero-toggle');
  const caption=document.getElementById('hero-film-caption');
  function state(){
    button.textContent=film.paused?'Play':'Pause';
    button.setAttribute('aria-label',film.paused?'Play background animation':'Pause background animation');
  }
  film.addEventListener('play',state);film.addEventListener('pause',state);
  film.addEventListener('timeupdate',()=>{
    caption.textContent=film.currentTime>=22.56&&film.currentTime<24?'384 saved trajectories · 202 garments · 2× replay':'Left: Flattening · Right: Folding · 2× saved task replay';
  });
  button.addEventListener('click',()=>{
    if(film.paused){
      film.setAttribute('data-auto','');
      if(!film.hasAttribute('src')){film.src=film.dataset.src;film.load();}
      film.play().catch(()=>{});
    }else{film.removeAttribute('data-auto');film.pause();}
  });
  state();
})();
