const {chromium}=require('playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const out=process.env.REVIEW_OUTPUT;fs.mkdirSync(out,{recursive:true});
const report={commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),passed:false,errors:[],screenshots:[],checks:[],performance:[],pagePath:'cloth-study.html'};
const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
let browser;
(async()=>{try{
  browser=await chromium.launch({channel:'chrome',args:['--use-gl=angle','--use-angle=swiftshader-webgl','--enable-unsafe-swiftshader']});
  for(const [width,height] of [[1440,900],[390,844]]){
    const context=await browser.newContext({viewport:{width,height},reducedMotion:'reduce'});
    const page=await context.newPage();let requests=0;
    page.on('request',()=>requests++);page.on('pageerror',e=>report.errors.push(e.message));
    page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
    page.on('response',r=>{if(r.status()>=400)report.errors.push(r.status()+' '+r.url());});
    await page.goto('http://127.0.0.1:8787/cloth-study.html',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>(window.liveCamera?.state.loaded===48&&window.liveCamera?.state.bufferedFrames===250)||window.liveCamera?.state.error,null,{timeout:120000});
    let state=await page.evaluate(()=>window.liveCamera.state);assert(!state.error,state.error);assert(state.ready&&state.fidelityReady);
    assert.deepEqual(state.focusVertices,[9760,6116]);assert.equal(state.phase,0);assert(!state.running);
    const loadedRequests=requests;
    const settle=async()=>{const n=await page.evaluate(()=>window.liveCamera.state.frames);await page.waitForFunction(n=>window.liveCamera.state.frames>n,n);};
    const shot=async name=>{const file=name+'.jpg';await page.screenshot({path:path.join(out,file),type:'jpeg',quality:94});report.screenshots.push({file,viewport:{width,height},state:await page.evaluate(()=>window.liveCamera.state)});save();};
    for(const frame of (width===1440?[0,75,150]:[75])){
      await page.evaluate(f=>window.liveCamera.seekFrame(f),frame);await settle();
      const camera=await page.evaluate(()=>window.liveCamera.state.camera);
      for(const variant of ['full','baseline']){
        await page.locator('[data-cloth-mode="'+variant+'"]').click();await settle();
        state=await page.evaluate(()=>window.liveCamera.state);
        assert.equal(state.phase,frame/25);assert.deepEqual(state.camera,camera);
        assert.deepEqual(state.focusVertices,variant==='full'?[9760,6116]:[1564,1537]);
        await shot('cloth48-'+width+'-'+frame+'-'+variant);
      }
    }
    await page.locator('[data-cloth-mode="full"]').click();await settle();
    // A stationary cloth is not recalculated merely because the camera moves.
    const updates=await page.evaluate(()=>window.liveCamera.state.geometryUpdates);
    for(const value of [.05,.1,.15]){
      await page.locator('#camera-distance').evaluate((el,v)=>{el.value=String(v);el.dispatchEvent(new Event('input',{bubbles:true}));},value);
      await page.waitForFunction(v=>Math.abs(window.liveCamera.state.renderedZoom-v)<1e-6,value);
    }
    assert.equal(await page.evaluate(()=>window.liveCamera.state.geometryUpdates),updates);
    await page.locator('#motion-toggle').click();
    const phase=await page.evaluate(()=>window.liveCamera.state.phase);
    const timing=await page.evaluate(async()=>{
      const times=[],latencies=[],slider=document.getElementById('camera-distance');let last=performance.now();
      for(let i=0;i<24;i++){
        slider.value=String(.05+i*.01);slider.dispatchEvent(new Event('input',{bubbles:true}));
        await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame);
        const now=performance.now();times.push(now-last);last=now;latencies.push(window.liveCamera.state.inputToFrameMs);
      }
      return {twoFrameIntervalsMs:times,inputToRenderCallMs:latencies};
    });
    assert.notEqual(await page.evaluate(()=>window.liveCamera.state.phase),phase);
    assert.equal(requests,loadedRequests,'Camera dragging and same-frame comparison make no new requests');
    report.performance.push({width,renderer:'Chrome SwiftShader software WebGL; not a target-device FPS benchmark',...timing});
    await shot('cloth48-'+width+'-dragging');
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await context.close();
  }
  assert.equal(report.errors.length,0,JSON.stringify(report.errors));
  report.checks.push('Complete source mesh counts; original cotton texture loading; same-frame/camera comparison at0,75,150; no network on switching or continuous drag; no geometry updates during paused camera drag; action clock advances under drag; desktop/mobile/reduced motion');
  report.passed=true;
}catch(e){report.failure=e.stack;process.exitCode=1;}finally{save();if(browser)await browser.close();console.log(JSON.stringify({passed:report.passed,failure:report.failure,errors:report.errors}));}})();
