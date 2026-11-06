(() => {
 const track=document.querySelector('.story-track');if(!track)return;
 const cards=[...track.children],dots=document.querySelector('.story-dots'),prev=document.querySelector('[data-story-prev]'),next=document.querySelector('[data-story-next]'),announcement=document.querySelector('.story-announcement');
 const reduced=matchMedia('(prefers-reduced-motion: reduce)'),automatic=new WeakSet();let active=-1,requested=0,programmatic=false,frame=0,settle,drag;
 for(const card of cards)for(const video of card.querySelectorAll('video[data-auto]'))automatic.add(video);
 const buttons=cards.map((card,index)=>{const b=document.createElement('button');b.type='button';b.setAttribute('aria-label',card.querySelector('h2').textContent.trim());b.addEventListener('click',()=>go(index));dots.append(b);return b;});
 function activate(index){
  if(active===index)return;active=index;
  for(let i=0;i<cards.length;i++){
   buttons[i].setAttribute('aria-current',String(i===index));cards[i].dataset.active=String(i===index);
   for(const video of cards[i].querySelectorAll('video')){
    if(i!==index){video.removeAttribute('data-auto');video.pause();}
    else if(automatic.has(video))video.setAttribute('data-auto','');
   }
  }
  prev.disabled=index===0;next.disabled=index===cards.length-1;
  announcement.textContent=(index+1)+' of '+cards.length+': '+cards[index].querySelector('h2').textContent.trim();
  document.dispatchEvent(new Event('clothdojo:ambient'));
 }
 function nearest(){const center=track.scrollLeft+track.clientWidth/2;return cards.reduce((best,c,i)=>Math.abs(c.offsetLeft+c.offsetWidth/2-center)<Math.abs(cards[best].offsetLeft+cards[best].offsetWidth/2-center)?i:best,0);}
 function sync(){frame=0;const index=nearest();if(!programmatic)requested=index;activate(index);}
 function go(index,behavior=reduced.matches?'auto':'smooth'){
  requested=Math.max(0,Math.min(cards.length-1,index));programmatic=true;
  const c=cards[requested];track.scrollTo({left:c.offsetLeft+c.offsetWidth/2-track.clientWidth/2,behavior});
  if(behavior==='auto'){sync();programmatic=false;}
 }
 track.addEventListener('scroll',()=>{if(!frame)frame=requestAnimationFrame(sync);clearTimeout(settle);settle=setTimeout(()=>{programmatic=false;requested=nearest();sync();},160);},{passive:true});
 prev.addEventListener('click',()=>go(requested-1));next.addEventListener('click',()=>go(requested+1));
 const interactive='a,button,input,textarea,select,summary,pre,.table-wrap';
 track.addEventListener('keydown',e=>{if(e.target.closest(interactive))return;let target;if(e.key==='ArrowRight')target=requested+1;else if(e.key==='ArrowLeft')target=requested-1;else if(e.key==='Home')target=0;else if(e.key==='End')target=cards.length-1;else return;e.preventDefault();go(target);});
 track.addEventListener('pointerdown',e=>{if(e.pointerType!=='mouse'||e.button!==0||e.target.closest(interactive))return;drag={id:e.pointerId,x:e.clientX,y:e.clientY,left:track.scrollLeft,moved:false};programmatic=false;});
 track.addEventListener('pointermove',e=>{if(!drag||e.pointerId!==drag.id)return;const dx=e.clientX-drag.x;if(!drag.moved){if(Math.abs(dx)<8||Math.abs(dx)<Math.abs(e.clientY-drag.y))return;drag.moved=true;track.setPointerCapture(e.pointerId);track.classList.add('is-dragging');}e.preventDefault();track.scrollLeft=drag.left-dx;});
 function release(e){if(!drag||e.pointerId!==drag.id)return;const moved=drag.moved;drag=undefined;track.classList.remove('is-dragging');if(track.hasPointerCapture(e.pointerId))track.releasePointerCapture(e.pointerId);if(moved)go(nearest());}
 track.addEventListener('pointerup',release);track.addEventListener('pointercancel',release);track.addEventListener('dragstart',e=>{if(e.target.tagName==='IMG')e.preventDefault();});
 track.addEventListener('focusin',e=>{const card=e.target.closest('.story-card');if(card&&cards.indexOf(card)!==active)go(cards.indexOf(card),'auto');});
 function fromHash(scroll){const target=document.getElementById(decodeURIComponent(location.hash.slice(1))),card=target?.closest('.story-card');if(!card)return;go(cards.indexOf(card),'auto');if(scroll)document.getElementById('highlights').scrollIntoView({behavior:reduced.matches?'auto':'smooth',block:'start'});}
 document.addEventListener('click',e=>{const link=e.target.closest('a[href^="#"]');if(!link)return;const id=link.getAttribute('href').slice(1),target=document.getElementById(id);if(!target?.closest('.story-card'))return;e.preventDefault();history.pushState(null,'','#'+id);fromHash(true);});
 addEventListener('hashchange',()=>fromHash(true));addEventListener('resize',()=>go(active<0?0:active,'auto'));
 Object.defineProperty(window,'storyCards',{value:{go,get state(){return {active,requested,count:cards.length,ids:cards.map(c=>c.id),scrollLeft:track.scrollLeft};}}});
 go(1,'auto');fromHash(false);
})();
