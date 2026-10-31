(() => {
  const narrow=matchMedia('(max-width:700px)');
  const image=document.getElementById('scene-frame');
  const film=document.getElementById('hero-film');
  const original=document.getElementById('scene-original');
  const expand=document.getElementById('scene-expand');
  const caption=document.getElementById('hero-film-caption');
  const labels={near:'Left: Flattening · Right: Folding · One garment asset',bridge:'48 saved trajectories · More garment assets and initial states',scale:'384 saved trajectories · 202 garment assets',motion:'2 → 8 workcells · Whole-group presentation reveal · 2× saved task replay'};
  let view='near';
  function update(explicit=false){
    const moving=view==='motion';film.pause();film.hidden=!moving;image.hidden=moving;expand.hidden=!moving;original.hidden=moving;
    document.querySelectorAll('[data-scene-view]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.sceneView===view)));
    if(moving){
      if(!film.hasAttribute('src')){film.src=film.dataset.src;film.load();}
      if(explicit){film.currentTime=0;film.play().catch(()=>{});}
    }else{
      const name=view==='near'?(narrow.matches?'pair-mobile':'pair'):view==='bridge'?'broader':'scale';
      image.src='assets/studio29/'+name+'.jpg';original.href=image.src;
    }
    caption.textContent=labels[view];
  }
  document.querySelectorAll('[data-scene-view]').forEach(button=>button.addEventListener('click',()=>{view=button.dataset.sceneView;update(view==='motion');}));
  narrow.addEventListener('change',()=>{if(view!=='motion')update();});
  update();
})();
