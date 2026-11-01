(() => {
  const root=document.getElementById('camera-viewer');
  const slots=[document.getElementById('camera-slot-a'),document.getElementById('camera-slot-b')];
  const slider=document.getElementById('camera-distance');
  const label=document.getElementById('camera-label');
  const status=document.getElementById('camera-status');
  const resume=document.getElementById('resume-auto');
  const toggle=document.getElementById('motion-toggle');
  const views=['near','middle','far'];const names={near:'近景',middle:'中景',far:'远景',auto:'自动运镜'};
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  let manifest,active=0,initialized=false,displayed='auto',mode='auto';
  let running=!reduced.matches&&!navigator.connection?.saveData;
  let version=0,pending=null,wanted='auto',lastError='',lastChange=null;
  for(const video of slots){
    if(!video.requestVideoFrameCallback)continue;
    const presented=(_now,frame)=>{
      video.dataset.presentedTime=String(frame.mediaTime);
      if(video===slots[active]&&lastChange&&video.dataset.view===lastChange.view&&lastChange.firstPresentedSkew===undefined){
        const expected=lastChange.masterPhase+(running&&!video.paused?(performance.now()-lastChange.switchedAt)/1000:0);
        lastChange.firstPresentedSkew=Math.abs(difference(frame.mediaTime,expected));
      }
      video.requestVideoFrameCallback(presented);
    };
    video.requestVideoFrameCallback(presented);
  }
  const phase=()=>initialized?slots[active].currentTime:0;
  const difference=(a,b)=>{const d=manifest?.duration_seconds||10;return ((a-b+d*1.5)%d)-d*.5;};
  function text(){
    label.textContent=mode==='auto'?'自动运镜':'手动 · '+names[wanted];
    resume.disabled=mode==='auto';resume.textContent=mode==='auto'?'自动运镜中':'恢复自动';
    toggle.textContent=running?'暂停动作':'继续动作';
    root.dataset.mode=mode;root.dataset.view=displayed;
    root.dataset.running=String(running);root.dataset.pending=pending?pending.view:'';
    root.setAttribute('aria-busy',String(Boolean(pending)||!initialized));
    slider.setAttribute('aria-valuetext',mode==='auto'?'自动运镜':names[wanted]);
  }
  function waitFor(video,event,ready,signal){
    if(ready())return Promise.resolve();
    return new Promise((resolve,reject)=>{
      const cleanup=()=>{clearTimeout(timer);video.removeEventListener(event,done);video.removeEventListener('error',fail);signal?.removeEventListener('abort',abort);};
      const done=()=>{if(ready()){cleanup();resolve();}};
      const fail=()=>{cleanup();reject(new Error('视频暂时无法加载'));};
      const abort=()=>{cleanup();reject(new DOMException('Cancelled','AbortError'));};
      const timer=setTimeout(()=>{cleanup();reject(new Error('换景等待超时'));},15000);
      video.addEventListener(event,done);video.addEventListener('error',fail);signal?.addEventListener('abort',abort,{once:true});
    });
  }
  async function seek(video,time,signal){
    const position=(time+video.duration)%video.duration;
    if(Math.abs(difference(video.currentTime,position))<.015&&video.readyState>=2&&!video.seeking)return;
    const done=waitFor(video,'seeked',()=>!video.seeking&&video.readyState>=2&&Math.abs(difference(video.currentTime,position))<.08,signal);
    video.currentTime=position;
    await done;
  }
  async function change(view){
    wanted=view;const ticket=++version;
    pending?.controller.abort();pending=null;
    if(initialized&&displayed===view){
      slots[1-active].pause();slots[1-active].removeAttribute('src');slots[1-active].load();
      status.textContent='';text();return;
    }
    const controller=new AbortController();const signal=controller.signal;
    pending={view,controller};lastError='';text();
    status.textContent=initialized?'正在准备'+names[view]+'，当前动作继续…':'正在准备自动展示…';
    const index=initialized?1-active:0;const next=slots[index];const before=phase();const started=performance.now();
    try{
      next.pause();next.removeAttribute('data-active');next.dataset.view=view;
      delete next.dataset.presentedTime;
      next.poster='assets/camera45/'+manifest.views[view].poster;
      next.src='assets/camera45/'+manifest.views[view].src;next.load();
      await waitFor(next,'canplay',()=>next.readyState>=3,signal);
      if(ticket!==version)return;
      // Sample the still-playing master after loading, not at drag start.
      await seek(next,phase(),signal);
      if(ticket!==version)return;
      if(running&&!document.hidden)await next.play();
      if(ticket!==version)return;
      if(!running||document.hidden)next.pause();
      // Correct load/seek drift before exposing the new view. No crossfade:
      // differently projected tables must never be superimposed.
      if(initialized&&Math.abs(difference(next.currentTime,phase()))>.06){
        await seek(next,phase(),signal);
        if(ticket!==version)return;
      }
      const master=phase();
      slots[active].removeAttribute('data-active');slots[active].setAttribute('aria-hidden','true');
      if(initialized)slots[active].pause();
      active=index;initialized=true;displayed=view;
      next.dataset.active='true';next.removeAttribute('aria-hidden');
      lastChange={view,fromPhase:before,masterPhase:master,presentedPhase:next.currentTime,skew:Math.abs(difference(next.currentTime,master)),latencyMs:performance.now()-started,switchedAt:performance.now()};
      pending=null;status.textContent='';text();
      slider.disabled=false;toggle.disabled=false;
      // Keep at most two sources, one playing after a committed change.
      const old=slots[1-active];old.pause();old.removeAttribute('src');old.load();
      root.dispatchEvent(new CustomEvent('camerachange',{detail:lastChange}));
    }catch(error){
      if(ticket!==version||error.name==='AbortError')return;
      pending=null;lastError=error.message;
      next.pause();status.textContent='换景暂未完成，保留当前画面。请再次拖动或点击继续动作。';text();
    }
  }
  slider.addEventListener('pointerdown',()=>{
    mode='manual';const view=views[Math.round(Number(slider.value))];
    if(view!==wanted||(!pending&&displayed!==view))change(view);
    else text();
  });
  slider.addEventListener('input',()=>{
    mode='manual';const view=views[Math.round(Number(slider.value))];
    if(view===wanted&&pending)return;
    change(view);
  });
  slider.addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key))return;
    event.preventDefault();let value=Math.round(Number(slider.value));
    value=event.key==='Home'?0:event.key==='End'?2:Math.max(0,Math.min(2,value+(['ArrowRight','ArrowUp'].includes(event.key)?1:-1)));
    slider.value=String(value);slider.dispatchEvent(new Event('input',{bubbles:true}));
  });
  slider.addEventListener('change',()=>{if(mode==='manual')slider.value=String(views.indexOf(wanted));});
  resume.addEventListener('click',()=>{mode='auto';change('auto');});
  toggle.addEventListener('click',async()=>{
    if(!manifest){location.reload();return;}
    running=!running;text();
    if(!initialized||lastError){await change(wanted);return;}
    if(running){try{await slots[active].play();}catch{running=false;status.textContent='点击继续动作以播放';text();}}
    else slots.forEach(video=>video.pause());
  });
  document.addEventListener('visibilitychange',()=>{
    if(document.hidden)slots.forEach(video=>video.pause());
    else if(initialized&&running)slots[active].play().catch(()=>{running=false;text();});
  });
  function autoPosition(t){
    const source=t+.48;
    if(source>=10){const u=Math.min(1,(source-10)/.44);return 2*(1-u*u*(3-2*u));}
    if(source<2)return 0;
    if(source<=3.8)return 1-(1-(source-2)/1.8)**3;
    if(source<4.3)return 1;
    if(source<=6.1)return 2-(1-(source-4.3)/1.8)**3;
    return 2;
  }
  function tick(){
    if(initialized&&mode==='auto')slider.value=String(autoPosition(phase()));
    requestAnimationFrame(tick);
  }
  Object.defineProperty(window,'cameraStudy',{value:{get state(){return {ready:initialized,mode,view:displayed,wanted,running,pending:pending?.view||null,phase:phase(),lastChange,lastError,playing:slots.filter(v=>!v.paused).length,loadedSources:slots.filter(v=>v.hasAttribute('src')).length};}}});
  text();tick();
  fetch('assets/camera45/manifest.json').then(response=>{if(!response.ok)throw new Error('素材暂未就绪');return response.json();}).then(data=>{manifest=data;toggle.disabled=false;return change('auto');}).catch(error=>{lastError=error.message;status.textContent=error.message;running=false;text();toggle.disabled=false;toggle.textContent='重新加载';});
})();
