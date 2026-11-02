const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const out=process.env.REVIEW_OUTPUT;fs.mkdirSync(out,{recursive:true});
const report={commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),passed:false,errors:[],screenshots:[],checks:[],timing:[],pagePath:'lighting-study.html'};
const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));let browser;
(async()=>{try{
 browser=await chromium.launch({channel:'chrome',args:['--use-gl=angle','--use-angle=swiftshader-webgl','--enable-unsafe-swiftshader']});
 for(const [width,height] of [[1440,900],[390,844]]){
  const context=await browser.newContext({viewport:{width,height},reducedMotion:'reduce'}),page=await context.newPage();let requests=0;
  page.on('request',()=>requests++);page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});page.on('response',r=>{if(r.status()>=400)report.errors.push(r.status()+' '+r.url());});
  await page.goto('http://127.0.0.1:8787/lighting-study.html',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.liveCamera?.state.loaded===48||window.liveCamera?.state.error,null,{timeout:150000});
  let s=await page.evaluate(()=>window.liveCamera.state);assert(!s.error,s.error);assert(s.ready);assert.deepEqual(s.focusVertices,[9760,6116]);const baselineRequests=requests;
  const settle=async()=>{const n=await page.evaluate(()=>window.liveCamera.state.frames);await page.waitForFunction(n=>window.liveCamera.state.frames>n,n);};
  const shot=async name=>{const file=name+'.jpg';await page.screenshot({path:path.join(out,file),type:'jpeg',quality:92});report.screenshots.push({file,viewport:{width,height},state:await page.evaluate(()=>window.liveCamera.state)});save();};
  for(const frame of (width===1440?[75,150]:[75])){
   await page.evaluate(f=>window.liveCamera.seekFrame(f),frame);await settle();const camera=await page.evaluate(()=>window.liveCamera.state.camera);
   for(const variant of ['enhanced','baseline']){
    await page.locator('[data-lighting-mode="'+variant+'"]').click();await settle();s=await page.evaluate(()=>window.liveCamera.state);
    assert.equal(s.phase,frame/25);assert.equal(s.lightingMode,variant);assert.deepEqual(s.camera,camera);await shot('light49-'+width+'-'+frame+'-'+variant);
   }
  }
  await page.locator('[data-lighting-mode="enhanced"]').click();await settle();
  const updates=await page.evaluate(()=>window.liveCamera.state.geometryUpdates);
  for(const v of [.1,.101,.2]){await page.locator('#camera-distance').evaluate((el,v)=>{el.value=String(v);el.dispatchEvent(new Event('input',{bubbles:true}));},v);await page.waitForFunction(v=>Math.abs(window.liveCamera.state.renderedZoom-v)<1e-6,v);}
  assert.equal(await page.evaluate(()=>window.liveCamera.state.geometryUpdates),updates);
  await page.locator('#motion-toggle').click();const phase=await page.evaluate(()=>window.liveCamera.state.phase);
  const timing=await page.evaluate(async()=>{const a=[];for(let i=0;i<10;i++){const el=document.getElementById('camera-distance');el.value=String(.1+i*.01);el.dispatchEvent(new Event('input',{bubbles:true}));await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame);a.push(window.liveCamera.state.inputToFrameMs);}return a;});
  assert.notEqual(await page.evaluate(()=>window.liveCamera.state.phase),phase);report.timing.push({width,softwareWebGL:true,inputToRenderCallMs:timing});
  await page.locator('#motion-toggle').click();await page.locator('#camera-distance').evaluate(el=>{el.value='1';el.dispatchEvent(new Event('input',{bubbles:true}));});await page.waitForFunction(()=>window.liveCamera.state.renderedZoom===1);await shot('light49-'+width+'-far');
  assert.equal(requests,baselineRequests,'No additional requests on lighting switch or continuous camera navigation');assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await context.close();
 }
 assert.equal(report.errors.length,0,JSON.stringify(report.errors));report.checks.push('Matched complete mesh/material/frame/camera; lighting switch preserves phase; arbitrary adjacent zoom; paused geometry cached; action continues during drag; no requests on interaction; near AO and far-scale rendering; desktop/mobile/reduced motion');report.passed=true;
}catch(e){report.failure=e.stack;process.exitCode=1;}finally{save();if(browser)await browser.close();console.log(JSON.stringify({passed:report.passed,failure:report.failure,errors:report.errors}));}})();
