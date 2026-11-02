(() => {
 const video=document.getElementById('distance-near'),root=document.getElementById('distance-viewer'),status=document.getElementById('distance-status'),label=document.getElementById('distance-label');
 const auto=document.getElementById('distance-auto'),play=document.getElementById('distance-play'),buttons=[...document.querySelectorAll('button[data-distance]')];
 const names={near:'近景',middle:'中景'},reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 let manifest,source,buffer,ready=false,running=!reduced&&!navigator.connection?.saveData,mode=reduced?'manual':'auto',wanted='near',current='near',stage='loop',error='',busy=false;
 let tailView='near',tailFrame=0,end=0,landedAt=0,timer,transition=null;
 const cache=new Map(),timeline=[],transitions=[],presentations=[],waits=[];
 const url=path=>new URL(path,new URL('assets/camera54/',location.href)).href;
 function ui(){
  root.dataset.stage=stage;root.dataset.distance=current;root.setAttribute('aria-busy',String(!ready));
  for(const b of buttons){b.disabled=!ready;b.setAttribute('aria-pressed',String(b.dataset.distance===wanted));}
  auto.disabled=play.disabled=!ready;auto.setAttribute('aria-pressed',String(mode==='auto'));play.textContent=running?'暂停动作':'继续动作';
  label.textContent=stage==='bridge'?names[transition.from]+' → '+names[transition.to]:(mode==='auto'?'自动 · ':'已固定')+names[current]+' · 动作循环';
  status.textContent=error?'影片暂未就绪：'+error:!ready?'正在载入近景影片…':wanted!==current&&stage==='loop'?'当前动作继续，即将运镜…':'';
 }
 function load(record){
  if(!cache.has(record.src))cache.set(record.src,fetch(url(record.src)).then(r=>{if(!r.ok)throw new Error('素材载入失败');return r.arrayBuffer();}).then(data=>({record,data})).catch(e=>{cache.delete(record.src);throw e;}));
  return cache.get(record.src);
 }
 function update(operation){return new Promise((resolve,reject)=>{
  const clean=()=>{buffer.removeEventListener('updateend',done);buffer.removeEventListener('error',fail);};
  const done=()=>{clean();resolve();},fail=()=>{clean();reject(new Error('视频片段解码失败'));};
  buffer.addEventListener('updateend',done,{once:true});buffer.addEventListener('error',fail,{once:true});
  try{operation();}catch(e){clean();reject(e);}
 });}
 async function append(data,entry){
  await update(()=>buffer.appendBuffer(data));timeline.push({...entry,start:end,end:end+entry.frames/25});end+=entry.frames/25;
 }
 function at(time){return timeline.find(e=>time>=e.start-.0001&&time<e.end-.0001)||timeline[timeline.length-1];}
 async function fill(){
  if(busy||!buffer||error)return;busy=true;
  try{
   while(end-video.currentTime<.8){
    const edge=tailView+'-'+wanted;
    if(wanted!==tailView&&tailFrame%25===0){
     const record=manifest.bridges[edge][tailFrame/25];
     if(!record.loaded){load(record).then(()=>{record.loaded=true;}).catch(fail);}
     if(record.loaded){
      const {data}=await load(record),from=tailView,to=wanted,phase=tailFrame/25;
      await append(data.slice(record.init_bytes),{kind:'bridge',from,to,phase,frames:40});tailView=to;tailFrame=record.end_frame;continue;
     }
    }
    const {record,data}=await load(manifest.views[tailView]),segment=record.segments[tailFrame/5];
    await append(data.slice(segment.offset,segment.offset+segment.bytes),{kind:'loop',view:tailView,phase:tailFrame/25,frames:5});tailFrame=(tailFrame+5)%250;
   }
   if(video.currentTime>20&&buffer.buffered.length&&buffer.buffered.start(0)<video.currentTime-16){
    await update(()=>buffer.remove(0,video.currentTime-15));while(timeline.length>1&&timeline[0].end<video.currentTime-15)timeline.shift();
   }
  }catch(e){fail(e);}finally{busy=false;}
 }
 function fail(e){error=e.message||String(e);running=false;video.pause();ui();}
 function request(view){mode='manual';wanted=view;ui();fill();}
 for(const b of buttons)b.addEventListener('click',()=>request(b.dataset.distance));
 auto.addEventListener('click',()=>{mode=mode==='auto'?'manual':'auto';wanted=current;landedAt=video.currentTime;ui();});
 play.addEventListener('click',async()=>{running=!running;ui();if(!running)video.pause();else if(!document.hidden)try{await video.play();}catch(e){fail(e);}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)video.pause();else if(ready&&running)video.play().catch(fail);});
 video.addEventListener('error',()=>fail(new Error(video.error?.message||'视频播放错误')));
 video.addEventListener('waiting',()=>{if(ready&&running&&!document.hidden){waits.push(video.currentTime);if(waits.length>30)waits.shift();}});
 video.requestVideoFrameCallback(function presented(now,frame){
  const e=at(frame.mediaTime);
  if(e){
   if(e.kind==='bridge'){
    if(transition!==e){transition=e;e.presentedAt=frame.mediaTime;e.presentedFrames=0;}stage='bridge';e.presentedFrames++;
   }else{
    if(stage==='bridge'){
     transitions.push({from:transition.from,to:transition.to,start:transition.start,end:transition.end,firstFrame:transition.presentedAt,presentedFrames:transition.presentedFrames,landingFrame:frame.mediaTime,expectedLanding:transition.end});if(transitions.length>30)transitions.shift();landedAt=frame.mediaTime;
    }current=e.view;stage='loop';
   }
   presentations.push({time:frame.mediaTime,wall:now,kind:e.kind});if(presentations.length>160)presentations.shift();
   if(mode==='auto'&&stage==='loop'&&video.currentTime-landedAt>=8)wanted=current==='near'?'middle':'near';ui();
  }
  fill();video.requestVideoFrameCallback(presented);
 });
 Object.defineProperty(window,'distanceStudy',{value:{get state(){const e=at(video.currentTime);return {ready,current,wanted,mode,running,stage,error,phase:e?((e.phase+video.currentTime-e.start)%10+10)%10:0,playing:video.paused?0:1,bufferAhead:end-video.currentTime,sourceChanges:video.dataset.sourceChanges||0,transitions:transitions.map(x=>({...x})),presentations:presentations.map(x=>({...x})),waits:[...waits],lastTransition:transitions.at(-1),stream:true};}}});
 async function prefetch(){const list=[manifest.views.middle,...Object.values(manifest.bridges).flat()];let cursor=0;await Promise.all([0,1].map(async()=>{while(cursor<list.length){const r=list[cursor++];try{await load(r);r.loaded=true;}catch{/* Explicit requests retry. */}}}));}
 async function start(){
  manifest=await fetch('assets/camera54/manifest.json').then(r=>{if(!r.ok)throw new Error('素材清单未就绪');return r.json();});
  const mime='video/mp4; codecs="'+manifest.codec+'"';if(!window.MediaSource||!MediaSource.isTypeSupported(mime))throw new Error('请使用支持此视频格式的 Chrome 浏览器');
  const near=await load(manifest.views.near);source=new MediaSource();const opened=new Promise(resolve=>source.addEventListener('sourceopen',resolve,{once:true}));video.src=URL.createObjectURL(source);video.dataset.sourceChanges='1';await opened;
  buffer=source.addSourceBuffer(mime);buffer.mode='sequence';await update(()=>buffer.appendBuffer(near.data.slice(0,near.record.init_bytes)));await fill();
  if(video.readyState<2)await new Promise(resolve=>video.addEventListener('loadeddata',resolve,{once:true}));ready=true;ui();if(running&&!document.hidden)await video.play();timer=setInterval(fill,100);prefetch();
 }
 ui();start().catch(fail);addEventListener('pagehide',()=>{clearInterval(timer);video.pause();if(video.src.startsWith('blob:'))URL.revokeObjectURL(video.src);});
})();
