const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process'),out=process.env.REVIEW_OUTPUT;fs.mkdirSync(out,{recursive:true});
const manifest=JSON.parse(fs.readFileSync('website/assets/camera58/manifest.json','utf8')),order=['near','middle','far','scale'].filter(v=>manifest.views[v]);
const report={commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),passed:false,errors:[],screenshots:[],checks:[],recordings:[],draft:manifest.draft,distances:order};
const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));let browser;
(async()=>{try{
 if(!manifest.draft){assert.deepEqual(order,['near','middle','far','scale']);assert.equal(manifest.sources.length,384);assert.equal(new Set(manifest.sources.map(s=>s.trajectory)).size,384);assert.equal(new Set(manifest.sources.map(s=>s.asset_id)).size,202);}
 browser=await chromium.launch({channel:'chrome'});
 for(const width of [1440,390]){
  const context=await browser.newContext({viewport:{width,height:900},reducedMotion:'reduce',...(width===1440?{recordVideo:{dir:path.join(out,'recordings'),size:{width,height:900}}}:{})}),page=await context.newPage();
  page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto('http://127.0.0.1:8787/distance-study.html');
  await page.waitForFunction(()=>window.distanceStudy?.state.ready||window.distanceStudy?.state.error,null,{timeout:45000});
  const state=()=>page.evaluate(()=>window.distanceStudy.state);
  assert(!(await state()).error,JSON.stringify(await state()));
  const originalSource=await page.locator('video').evaluate(v=>v.currentSrc);
  const shot=async name=>{const file=name+'-'+width+'.jpg';await page.screenshot({path:path.join(out,file),type:'jpeg',quality:92});report.screenshots.push({file,state:await state()});save();};
  assert.equal(await page.locator('video').count(),1);assert.equal(await page.locator('video[controls]').count(),0);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.evaluate(()=>{window.unexpectedSeeks=0;document.querySelector('video').addEventListener('seeking',()=>window.unexpectedSeeks++);});
  const hold=async view=>{
   await page.evaluate(()=>{
    window.loopSeams=[];window.loopProbeActive=true;const v=document.querySelector('video'),c=document.createElement('canvas');c.width=160;c.height=56;const ctx=c.getContext('2d',{willReadFrequently:true});let last,phase;
    v.requestVideoFrameCallback(function sample(_now,frame){
     if(!window.loopProbeActive)return;ctx.drawImage(v,0,0,160,56);const pixels=ctx.getImageData(0,0,160,56).data,next=window.distanceStudy.phaseAt(frame.mediaTime);
     if(last&&phase>9.5&&next<.5){let sum=0,cloth=0,count=0;
      for(let i=0;i<pixels.length;i+=4){const d=Math.abs(pixels[i]-last[i])+Math.abs(pixels[i+1]-last[i+1])+Math.abs(pixels[i+2]-last[i+2]);sum+=d;
       if(Math.max(...pixels.slice(i,i+3))-Math.min(...pixels.slice(i,i+3))>12||Math.max(...last.slice(i,i+3))-Math.min(...last.slice(i,i+3))>12){cloth+=d;count++;}
      }window.loopSeams.push({time:frame.mediaTime,meanRgbDifference:sum/(160*56*3),clothDifference:count?cloth/(count*3):0,coloredPixels:count});
     }last=pixels;phase=next;v.requestVideoFrameCallback(sample);
    });
   });
   await page.waitForTimeout(10500);const seams=await page.evaluate(()=>{window.loopProbeActive=false;return window.loopSeams;});const s=await state();assert.equal(s.current,view);assert.equal(s.stage,'loop');assert(seams.length>=1,JSON.stringify({view,seams}));assert(seams.every(x=>x.meanRgbDifference<8&&x.clothDifference<28),JSON.stringify(seams));assert.equal(s.waits.length,0,JSON.stringify(s.waits));report.checks.push({width,heldLoop:view,seams});save();
  };
  const move=async(target,pause=false)=>{
   if((await state()).current===target)return;
   await page.locator('button[data-distance="'+target+'"]').click();
   await page.waitForFunction(()=>window.distanceStudy.state.stage==='bridge',null,{timeout:20000});
   const response=(await state()).requests.at(-1);assert(response.latencyMs<1600,JSON.stringify(response));report.checks.push({width,target,clickToCameraMs:response.latencyMs});
   if(pause){await page.locator('#distance-play').click();const held=await page.locator('video').evaluate(v=>v.currentTime);await page.waitForTimeout(250);assert.equal(await page.locator('video').evaluate(v=>v.currentTime),held);await shot('stream-paused-bridge');await page.locator('#distance-play').click();}
   await page.waitForTimeout(320);await shot('early-reveal-to-'+target);
   await page.waitForTimeout(650);await shot('stream-to-'+target);
   await page.waitForFunction(target=>window.distanceStudy.state.stage==='loop'&&window.distanceStudy.state.current===target,target,{timeout:18000});
   await page.waitForTimeout(150);const s=await state(),t=s.lastTransition;assert(!s.error,s.error);assert.equal(s.playing,1);assert.equal(await page.locator('video').evaluate(v=>v.currentSrc),originalSource);
   assert(Math.abs(t.firstFrame-t.start)<.081,JSON.stringify(t));assert(Math.abs(t.landingFrame-t.expectedLanding)<.081,JSON.stringify(t));assert(t.presentedFrames>=Math.ceil(t.expectedFrames*.85),JSON.stringify(t));assert.equal(s.waits.length,0,JSON.stringify(s.waits));
   const p=s.presentations.filter(x=>x.time>=t.start-.08&&x.time<=t.end+.08),gaps=p.slice(1).map((x,i)=>x.time-p[i].time);assert(gaps.every(x=>x<=.121),JSON.stringify(gaps));report.checks.push({width,target,transition:t,maxFrameGap:Math.max(0,...gaps),cachedBytes:s.cachedBytes});await shot('stream-'+target+'-landed');
  };
  await shot('stream-near');await page.locator('#distance-play').click();if(width===1440)await hold('near');
  for(const target of order.slice(1)){await move(target);if(width===1440||target==='middle'||target==='scale')await hold(target);}
  if(width===1440){
   const expected=order[order.length-2];await page.locator('#distance-auto').click();
   await page.waitForFunction(expected=>window.distanceStudy.state.current===expected&&window.distanceStudy.state.stage==='loop',expected,{timeout:18000});report.checks.push({automaticReturn:await state()});await page.locator('button[data-distance="'+expected+'"]').click();
  }
  for(const target of [...order].reverse().slice(1))await move(target,width===1440&&target==='middle');
  await page.locator('button[data-distance="'+order.at(-1)+'"]').click();await page.locator('button[data-distance="middle"]').click();
  await page.waitForFunction(()=>window.distanceStudy.state.current==='middle'&&window.distanceStudy.state.stage==='loop',null,{timeout:15000});assert.equal((await state()).wanted,'middle');await move('near');
  if(width===1440&&order.length===4){
   const before=(await state()).transitions.length;await page.locator('button[data-distance="scale"]').click();
   await page.waitForFunction(()=>window.distanceStudy.state.current==='scale'&&window.distanceStudy.state.stage==='loop',null,{timeout:25000});const s=await state();assert.deepEqual(s.transitions.slice(before).map(t=>t.to),['middle','far','scale']);assert.equal(s.wanted,'scale');assert.equal(s.waits.length,0);report.checks.push({directSelectionRoute:s.transitions.slice(before)});await shot('direct-scale');
  }
  assert.equal(await page.evaluate(()=>window.unexpectedSeeks),0);assert.equal(await page.locator('video').evaluate(v=>v.currentSrc),originalSource);
  const recording=page.video();await context.close();if(recording){const file='stream-switch-1440.webm';await recording.saveAs(path.join(out,file));report.recordings.push(file);}save();
 }
 {
  const context=await browser.newContext({viewport:{width:1440,height:900},reducedMotion:'reduce'}),page=await context.newPage();let ranged=0,fullFinished=false;
  await page.route('**/assets/camera58/middle.mp4',async route=>{
   if(route.request().headers().range){ranged++;await page.waitForTimeout(60);}else{await page.waitForTimeout(6000);fullFinished=true;}
   await route.continue();
  });
  await page.goto('http://127.0.0.1:8787/distance-study.html');await page.waitForFunction(()=>window.distanceStudy?.state.ready,null,{timeout:45000});
  await page.locator('#distance-play').click();const clicked=Date.now();await page.locator('button[data-distance="middle"]').click();
  await page.waitForFunction(()=>window.distanceStudy.state.current==='middle'&&window.distanceStudy.state.stage==='loop',null,{timeout:5000});
  const state=await page.evaluate(()=>window.distanceStudy.state),elapsed=Date.now()-clicked;assert(ranged>0);assert(!fullFinished,'Camera should land before the entire destination movie finishes loading');assert.equal(state.waits.length,0);assert(state.requests.at(-1).latencyMs<1600);report.checks.push({slowDestinationFullDownload:{landingMs:elapsed,rangeRequests:ranged,clickToCameraMs:state.requests.at(-1).latencyMs,waits:state.waits}});save();
  await page.waitForTimeout(Math.max(0,6500-elapsed));await context.close();
 }
 if(!manifest.draft){
  const context=await browser.newContext({viewport:{width:1440,height:900},reducedMotion:'no-preference',recordVideo:{dir:path.join(out,'recordings-auto'),size:{width:1440,height:900}}}),page=await context.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto('http://127.0.0.1:8787/distance-study.html');await page.waitForFunction(()=>window.distanceStudy?.state.ready,null,{timeout:45000});const src=await page.locator('video').evaluate(v=>v.currentSrc);
  for(const target of ['middle','far','scale'])await page.waitForFunction(target=>window.distanceStudy.state.current===target&&window.distanceStudy.state.stage==='loop',target,{timeout:30000});
  const arrival=await page.locator('video').evaluate(v=>v.currentTime);assert(arrival<24,'Default tour should reach scale promptly: '+arrival);await page.screenshot({path:path.join(out,'automatic-scale-1440.jpg'),type:'jpeg',quality:92});report.screenshots.push({file:'automatic-scale-1440.jpg',state:await page.evaluate(()=>window.distanceStudy.state)});
  await page.waitForFunction(()=>window.distanceStudy.state.current==='near'&&window.distanceStudy.state.stage==='loop'&&window.distanceStudy.state.transitions.length>=6,null,{timeout:45000});const s=await page.evaluate(()=>window.distanceStudy.state);assert.deepEqual(s.transitions.slice(0,6).map(t=>t.to),['middle','far','scale','far','middle','near']);assert.equal(s.waits.length,0,JSON.stringify(s));assert.equal(await page.locator('video').evaluate(v=>v.currentSrc),src);report.checks.push({defaultTour:s,scaleArrivalSeconds:arrival});const recording=page.video();await context.close();const file='automatic-tour-1440.webm';await recording.saveAs(path.join(out,file));report.recordings.push(file);save();
 }
 assert.deepEqual(report.errors,[]);report.passed=true;
}catch(e){report.errors.push(e.stack||String(e));process.exitCode=1;}finally{save();await browser?.close();}})();
