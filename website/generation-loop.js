(() => {
  document.querySelectorAll('[data-ambient-toggle]').forEach(button=>{
    const film=document.getElementById(button.dataset.ambientToggle);
    function state(){
      button.textContent=film.paused?'Play':'Pause';
      button.setAttribute('aria-label',(film.paused?'Play ':'Pause ')+button.dataset.motionName);
    }
    film.addEventListener('play',state);film.addEventListener('pause',state);
    button.addEventListener('click',()=>{
      if(film.paused){
        film.setAttribute('data-auto','');
        if(!film.hasAttribute('src')){film.src=film.dataset.src;film.load();}
        film.play().catch(()=>{});
      }else{film.removeAttribute('data-auto');film.pause();}
    });
    state();
  });
})();
