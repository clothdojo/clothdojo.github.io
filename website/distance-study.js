(() => {
 const root=document.getElementById('distance-viewer'),status=document.getElementById('distance-status'),label=document.getElementById('distance-label');
 const loops={near:document.getElementById('distance-near'),middle:document.getElementById('distance-middle')},bridge=document.getElementById('distance-bridge');
 const all=[loops.near,loops.middle,bridge],playButton=document.getElementById('distance-play'),autoButton=document.getElementById('distance-auto');
 const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches,names={near:'近景',middle:'中景'};
 let manifest,ready=false,current='near',wanted='near',mode=reduced?'manual':'auto',running=!reduced&&!navigator.connection?.saveData;
 let active=loops.near,stage='loop',pending=null,version=0,error='',elapsed=0,lastTime=0,inFlight=null,lastTransition=null;
 const transitions=[],cache=new Map(),asset=path=>new URL(path,new URL('assets/camera52/',location.href)).href;
 const wrap=t=>((t%10)+10)%10,diff=(a,b)=>wrap(a-b+5)-5;
 function ui(){
  root.setAttribute('aria-busy',String(!ready));root.dataset.stage=stage;root.dataset.distance=current;
  document.querySelectorAll('[data-distance]').forEach(b=>{b.disabled=!ready;b.setAttribute('aria-pressed',String(b.dataset.distance===wanted));});
  autoButton.disabled=!ready;playButton.disabled=!ready;autoButton.setAttribute('aria-pressed',String(mode==='auto'));
  playButton.textContent=running?'暂停动作':'继续动作';
  label.textContent=stage==='bridge'?names[inFlight.from]+' → '+names[inFlight.to]:(mode==='manual'?'已固定':'自动 · ')+names[current]+' · 动作循环';
  status.textContent=error?'运镜暂未完成，当前距离继续播放。':!ready?'正在载入近景影片…':pending?running?'正在准备运镜，当前动作继续…':'继续播放后运镜':'';
 }
 function show(video){for(const v of all){v.dataset.active=String(v===video);v.setAttribute('aria-hidden',String(v!==video));}active=video;}
 function waitReady(video,signal){
  if(video.readyState>=3&&!video.seeking)return Promise.resolve();
  return new Promise((resolve,reject)=>{
   const cleanup=()=>{clearTimeout(timer);for(const e of ['canplay','loadeddata','seeked'])video.removeEventListener(e,check);video.removeEventListener('error',fail);signal?.removeEventListener('abort',abort);};
   const check=()=>{if(video.readyState>=3&&!video.seeking){cleanup();resolve();}};
   const fail=()=>{cleanup();reject(new Error('Media loading failed'));},abort=()=>{cleanup();reject(new DOMException('Cancelled','AbortError'));};
   const timer=setTimeout(()=>{cleanup();reject(new Error('Media preparation timed out'));},20000);
   for(const e of ['canplay','loadeddata','seeked'])video.addEventListener(e,check);video.addEventListener('error',fail);signal?.addEventListener('abort',abort,{once:true});
   if(signal?.aborted)abort();
  });
 }
 async function seek(video,time,signal){
  if(video.readyState>=3&&!video.seeking&&Math.abs(video.currentTime-time)<.005)return;
  video.currentTime=time;await waitReady(video,signal);
 }
 async function ensureLoop(view,signal){
  const video=loops[view];if(!video.hasAttribute('src')){video.src=asset(manifest.views[view].src);video.poster=asset(manifest.views[view].poster);video.preload='auto';video.load();}
  await waitReady(video,signal);return video;
 }
 function clipURL(record){
  if(!cache.has(record.src))cache.set(record.src,fetch(asset(record.src)).then(r=>{if(!r.ok)throw new Error('Bridge loading failed');return r.blob();}).then(blob=>URL.createObjectURL(blob)).catch(e=>{cache.delete(record.src);throw e;}));
  return cache.get(record.src);
 }
 function cancelPending(){version++;pending?.controller.abort();pending=null;if(stage==='loop'){bridge.pause();bridge.removeAttribute('src');bridge.load();}}
 async function prepare(target){
  cancelPending();if(target===current){ui();return;}
  const ticket=version,controller=new AbortController(),signal=controller.signal;
  pending={target,controller,armed:false,anchor:null};error='';ui();
  try{
   const destination=await ensureLoop(target,signal);
   while(ticket===version&&stage==='loop'){
    const anchor=Math.ceil(loops[current].currentTime+.22)%10,record=manifest.bridges[current+'-'+target][anchor];
    const url=await clipURL(record);if(ticket!==version)return;
    if(wrap(anchor-loops[current].currentTime)>1.4)continue;
    bridge.pause();bridge.src=url;bridge.load();await waitReady(bridge,signal);await seek(bridge,0,signal);
    destination.pause();await seek(destination,record.end_phase,signal);
    if(ticket!==version)return;
    if(wrap(anchor-loops[current].currentTime)>1.4)continue;
    pending={target,controller,armed:true,anchor,record,ticket};ui();return;
   }
  }catch(e){if(ticket!==version||signal.aborted)return;pending=null;error=e.message;ui();}
 }
 function request(view,manual){
  if(!ready)return;
  if(manual)mode='manual';wanted=view;error='';
  if(stage==='loop')prepare(view);else ui();
 }
 async function startBridge(){
  const item=pending;if(!item?.armed||stage!=='loop'||!running||document.hidden)return;
  const source=loops[current],late=wrap(source.currentTime-item.anchor);
  if(late>.09){prepare(wanted);return;}
  pending=null;source.pause();stage='bridge';
  inFlight={from:current,to:item.target,anchor:item.anchor,record:item.record,requestedPhase:item.anchor,startPhase:source.currentTime,startedAt:performance.now()};
  lastTransition={from:current,to:item.target,anchor:item.anchor,startPhase:source.currentTime,startSkew:Math.abs(diff(source.currentTime,item.anchor)),startedAt:performance.now()};
  show(bridge);ui();
  try{await bridge.play();if(!running||document.hidden)bridge.pause();}catch(e){running=false;error=e.message;ui();}
 }
 bridge.addEventListener('ended',async()=>{
  if(stage!=='bridge'||!inFlight)return;
  const flight=inFlight;stage='landing';const destination=loops[flight.to];
  try{
   await seek(destination,flight.record.end_phase);
   bridge.pause();show(destination);current=flight.to;stage='loop';elapsed=0;lastTime=destination.currentTime;
   lastTransition.landedPhase=destination.currentTime;lastTransition.expectedPhase=flight.record.end_phase;lastTransition.landingSkew=Math.abs(diff(destination.currentTime,flight.record.end_phase));lastTransition.landedAt=performance.now();
   transitions.push({...lastTransition});if(transitions.length>30)transitions.shift();inFlight=null;
   if(running&&!document.hidden)await destination.play();
   bridge.removeAttribute('src');bridge.load();ui();
   if(wanted!==current)prepare(wanted);
  }catch(e){error=e.message;running=false;ui();}
 });
 for(const video of all){
  const presented=(_now,frame)=>{
   video.dataset.presentedTime=String(frame.mediaTime);
   if(video===active&&lastTransition){
    if(stage==='bridge'&&lastTransition.bridgeFirstFrame===undefined)lastTransition.bridgeFirstFrame=frame.mediaTime;
    if(stage==='loop'&&lastTransition.landedAt&&lastTransition.firstPresentedSkew===undefined){
     lastTransition.firstPresentedSkew=Math.abs(diff(frame.mediaTime,lastTransition.expectedPhase+(running?(performance.now()-lastTransition.landedAt)/1000:0)));
     const saved=transitions[transitions.length-1];if(saved)saved.firstPresentedSkew=lastTransition.firstPresentedSkew;
    }
   }
   video.requestVideoFrameCallback(presented);
  };video.requestVideoFrameCallback(presented);
 }
 document.querySelectorAll('[data-distance]').forEach(b=>b.addEventListener('click',()=>request(b.dataset.distance,true)));
 autoButton.addEventListener('click',()=>{mode=mode==='auto'?'manual':'auto';elapsed=0;wanted=current;if(stage==='loop')cancelPending();ui();});
 playButton.addEventListener('click',async()=>{
  running=!running;ui();if(!running){all.forEach(v=>v.pause());return;}
  try{await active.play();lastTime=active.currentTime;}catch(e){running=false;error=e.message;ui();}
 });
 document.addEventListener('visibilitychange',()=>{if(document.hidden)all.forEach(v=>v.pause());else if(ready&&running)active.play().catch(e=>{error=e.message;running=false;ui();});});
 function tick(){
  requestAnimationFrame(tick);if(!ready||!running||document.hidden||stage!=='loop')return;
  const t=loops[current].currentTime,delta=wrap(t-lastTime);if(delta<.5)elapsed+=delta;lastTime=t;
  if(pending?.armed){const late=wrap(t-pending.anchor);if(late<.09)startBridge();else if(wrap(pending.anchor-t)>1.5)prepare(wanted);}
  else if(mode==='auto'&&!pending&&elapsed>=8)request(current==='near'?'middle':'near',false);
 }
 async function prefetch(){
  // Start only after the first fixed view is available. Small files stay encoded.
  const list=[...manifest.bridges['near-middle'],...manifest.bridges['middle-near']];let cursor=0;
  await Promise.all(Array.from({length:2},async()=>{while(cursor<list.length){const record=list[cursor++];try{await clipURL(record);}catch{/* A later explicit selection can retry. */}}}));
 }
 Object.defineProperty(window,'distanceStudy',{value:{get state(){return {ready,current,wanted,mode,running,stage,phase:stage==='bridge'?wrap(inFlight.anchor+bridge.currentTime):active.currentTime,pending:pending?{target:pending.target,armed:pending.armed,anchor:pending.anchor}:null,playing:all.filter(v=>!v.paused&&!v.ended).length,lastTransition,transitions:transitions.map(x=>({...x})),error,cacheEntries:cache.size};}}});
 ui();requestAnimationFrame(tick);
 fetch('assets/camera52/manifest.json').then(r=>{if(!r.ok)throw new Error('Manifest loading failed');return r.json();}).then(async data=>{
  manifest=data;await ensureLoop('near');ready=true;show(loops.near);ui();
  if(running&&!document.hidden)await active.play();lastTime=active.currentTime;
  ensureLoop('middle').catch(()=>{});if(!navigator.connection?.saveData)prefetch();
 }).catch(e=>{error=e.message;running=false;ui();});
 addEventListener('pagehide',()=>{all.forEach(v=>v.pause());for(const promise of cache.values())promise.then(url=>URL.revokeObjectURL(url)).catch(()=>{});});
})();
