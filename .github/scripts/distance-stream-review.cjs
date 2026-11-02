const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process'),out=process.env.REVIEW_OUTPUT;fs.mkdirSync(out,{recursive:true});
const report={commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),passed:false,errors:[],screenshots:[],checks:[],recordings:[]};
const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));let browser;
(async()=>{try{
 browser=await chromium.launch({channel:'chrome'});
 for(const width of [1440,390]){
  const context=await browser.newContext({viewport:{width,height:900},reducedMotion:'reduce',...(width===1440?{recordVideo:{dir:path.join(out,'recordings'),size:{width,height:900}}}:{})}),page=await context.newPage();
  page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto('http://127.0.0.1:8787/distance-study.html');
  await page.waitForFunction(()=>window.distanceStudy?.state.ready||window.distanceStudy?.state.error,null,{timeout:45000});
  const state=()=>page.evaluate(()=>window.distanceStudy.state);
  assert(!(await state()).error,JSON.stringify(await state()));
  const shot=async name=>{const file=name+'-'+width+'.jpg';await page.screenshot({path:path.join(out,file),type:'jpeg',quality:92});report.screenshots.push({file,state:await state()});save();};
  assert.equal(await page.locator('video').count(),1);assert.equal(await page.locator('video[controls]').count(),0);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.evaluate(()=>{window.unexpectedSeeks=0;document.querySelector('video').addEventListener('seeking',()=>window.unexpectedSeeks++);});
  await shot('stream-near');await page.locator('#distance-play').click();
  for(const target of ['middle','near']){
   await page.locator('button[data-distance="'+target+'"]').click();
   await page.waitForFunction(()=>window.distanceStudy.state.stage==='bridge',null,{timeout:20000});
   await page.waitForTimeout(650);await shot('stream-to-'+target);
   await page.waitForFunction(target=>window.distanceStudy.state.stage==='loop'&&window.distanceStudy.state.current===target,target,{timeout:12000});
   await page.waitForTimeout(150);const s=await state();assert(!s.error,s.error);assert.equal(s.playing,1);assert.equal(String(s.sourceChanges),'1');
   assert(Math.abs(s.lastTransition.firstFrame-s.lastTransition.start)<.081,JSON.stringify(s.lastTransition));
   assert(Math.abs(s.lastTransition.landingFrame-s.lastTransition.expectedLanding)<.081,JSON.stringify(s.lastTransition));
   assert(s.lastTransition.presentedFrames>=34,JSON.stringify(s.lastTransition));assert.equal(s.waits.length,0,JSON.stringify(s.waits));
   report.checks.push({width,target,state:s});await shot('stream-'+target+'-landed');
   if(target==='middle'){
    const before=(await state()).phase;await page.waitForTimeout(10500);const after=await state();assert.equal(after.current,'middle');assert.equal(after.stage,'loop');assert(after.presentations.length>100);report.checks.push({width,heldLoop:{before,after:after.phase}});
   }
  }
  await page.locator('button[data-distance="middle"]').click();await page.locator('button[data-distance="near"]').click();
  await page.waitForTimeout(2800);await page.waitForFunction(()=>window.distanceStudy.state.current==='near'&&window.distanceStudy.state.stage==='loop',null,{timeout:10000});
  if(width===1440){
   await page.locator('button[data-distance="middle"]').click();await page.waitForFunction(()=>window.distanceStudy.state.stage==='bridge',null,{timeout:12000});
   await page.locator('#distance-play').click();const held=await page.locator('video').evaluate(v=>v.currentTime);await page.waitForTimeout(250);assert.equal(await page.locator('video').evaluate(v=>v.currentTime),held);await shot('stream-paused-bridge');
   await page.locator('#distance-play').click();await page.waitForFunction(()=>window.distanceStudy.state.current==='middle'&&window.distanceStudy.state.stage==='loop');
   await page.locator('#distance-auto').click();await page.waitForFunction(()=>window.distanceStudy.state.current==='near'&&window.distanceStudy.state.stage==='loop',null,{timeout:16000});
   report.checks.push({automaticReturn:await state()});
  }
  assert.equal(await page.evaluate(()=>window.unexpectedSeeks),0);assert.equal(String((await state()).sourceChanges),'1');
  const recording=page.video();await context.close();if(recording){const file='stream-switch-1440.webm';await recording.saveAs(path.join(out,file));report.recordings.push(file);}save();
 }
 assert.deepEqual(report.errors,[]);report.passed=true;
}catch(e){report.errors.push(e.stack||String(e));process.exitCode=1;}finally{save();await browser?.close();}})();
