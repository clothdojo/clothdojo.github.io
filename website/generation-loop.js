(() => {
  const film=document.getElementById('hero-film');
  const caption=document.getElementById('hero-film-caption');
  const buttons=[...document.querySelectorAll('[data-gen-seek]')];
  function label(){
    const t=film.currentTime;
    const stage=t<9.64||t>30.36?0:t<21.56||t>25.32?1:2;
    buttons.forEach((b,i)=>b.setAttribute('aria-pressed',String(i===stage)));
    caption.textContent=t>=22.56&&t<=24.32?'384 saved trajectories · 202 garments · 2× replay':'Left: Flattening · Right: Folding · 2× saved task replay';
  }
  film.addEventListener('timeupdate',label);
  buttons.forEach(button=>button.addEventListener('click',()=>{
    const seek=()=>{film.currentTime=Number(button.dataset.genSeek);label();film.play().catch(()=>{});};
    if(film.readyState)seek();
    else{film.addEventListener('loadedmetadata',seek,{once:true});film.src=film.dataset.src;film.load();}
  }));
})();
