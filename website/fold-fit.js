(() => {
  const narrow=matchMedia('(max-width:700px)');
  const image=document.getElementById('scene-frame');
  const film=document.getElementById('hero-film');
  const original=document.getElementById('scene-original');
  const expand=document.getElementById('scene-expand');
  const caption=document.getElementById('hero-film-caption');
  let view='near', revision=0;
  function update(explicit=false){
    const version=++revision, moving=view==='motion', wasPlaying=!film.paused, position=film.currentTime;
    film.pause();film.hidden=!moving;image.hidden=moving;expand.hidden=!moving;original.hidden=moving;
    const format=narrow.matches && view!=='scale'?'mobile':'wide';
    const prefix='assets/studio28/'+format;
    document.querySelectorAll('[data-scene-view]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.sceneView===view)));
    if(moving){
      const src=prefix+'-handoff.mp4';film.dataset.src=src;film.poster=prefix+'-bridge.jpg';
      const play=()=>{if(explicit||wasPlaying)film.play().catch(()=>{});};
      if(film.getAttribute('src')!==src){
        film.addEventListener('loadedmetadata',()=>{if(version!==revision)return;film.currentTime=Math.min(position,Math.max(0,film.duration-.05));play();},{once:true});
        film.src=src;film.load();
      }else play();
    }else{image.src=prefix+'-'+view+'.jpg';original.href=image.src;}
    caption.textContent=moving?'2.4 s real motion excerpt · 25 fps · 2× task replay':'Flattening + Folding · 384 trajectories / 202 garments';
  }
  document.querySelectorAll('[data-scene-view]').forEach(button=>button.addEventListener('click',()=>{view=button.dataset.sceneView;update(view==='motion');}));
  narrow.addEventListener('change',()=>update());
  update();
})();
