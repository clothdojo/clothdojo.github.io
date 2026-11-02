const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process'),out=process.env.REVIEW_OUTPUT;fs.mkdirSync(out,{recursive:true});
const report={commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),passed:false,errors:[],screenshots:[],checks:[]};
const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));let browser;
(async()=>{try{
 browser=await chromium.launch({channel:'chrome'});
 for(const width of [1440,390]){
  const context=await browser.newContext({viewport:{width,height:900},reducedMotion:'reduce'}),page=await context.newPage();
  page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto('http://127.0.0.1:8787/distance-study.html');
  await page.waitForFunction(()=>window.distanceStudy?.state.ready,null,{timeout:45000});
  const state=()=>page.evaluate(()=>window.distanceStudy.state);
  const shot=async name=>{const file=name+'-'+width+'.jpg';await page.screenshot({path:path.join(out,file),type:'jpeg',quality:92});report.screenshots.push({file,state:await state()});save();};
  assert.equal(await page.locator('video[controls]').count(),0);assert.equal(await page.locator('canvas').count(),0);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await shot('near');
  await page.locator('#distance-play').click();
  for(const target of ['middle','near']){
   await page.locator('button[data-distance="'+target+'"]').click();
   await page.waitForFunction(()=>window.distanceStudy.state.stage==='bridge',null,{timeout:20000});
   await page.waitForTimeout(320);await shot('bridge-to-'+target);
   await page.waitForFunction(target=>window.distanceStudy.state.stage==='loop'&&window.distanceStudy.state.current===target,target,{timeout:10000});
   await page.waitForTimeout(160);const s=await state();assert(!s.error,s.error);assert.equal(s.playing,1);assert(s.lastTransition.startSkew<.1);assert(s.lastTransition.landingSkew<.01);assert(s.lastTransition.firstPresentedSkew<.15,JSON.stringify(s.lastTransition));
   report.checks.push({width,target,transition:s.lastTransition});await shot(target+'-landed');
   if(target==='middle'){
    await page.locator('#distance-middle').evaluate(v=>{v.currentTime=9.8;});
    await page.waitForFunction(()=>document.getElementById('distance-middle').currentTime<1,null,{timeout:3000});
    assert.equal((await state()).current,'middle');
   }
  }
  await page.locator('button[data-distance="middle"]').click();await page.locator('button[data-distance="near"]').click();
  await page.waitForTimeout(500);assert.equal((await state()).wanted,'near');assert.equal((await state()).stage,'loop');
  await page.locator('#distance-play').click();const t=await page.locator('#distance-near').evaluate(v=>v.currentTime);await page.waitForTimeout(250);assert.equal(await page.locator('#distance-near').evaluate(v=>v.currentTime),t);
  await context.close();
 }
 assert.deepEqual(report.errors,[]);report.passed=true;
}catch(e){report.errors.push(e.stack||String(e));process.exitCode=1;}finally{save();await browser?.close();}})();
