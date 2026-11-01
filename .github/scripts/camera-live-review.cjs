const {chromium}=require('playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const out=process.env.REVIEW_OUTPUT;fs.mkdirSync(out,{recursive:true});
const report={commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),passed:false,errors:[],screenshots:[],checks:[],pagePath:'camera-study.html'};
const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
let browser;
(async()=>{
  try{
    browser=await chromium.launch({channel:'chrome',args:['--use-gl=angle','--use-angle=swiftshader-webgl','--enable-unsafe-swiftshader']});
    for(const [width,height,reducedMotion] of [[1440,900,'no-preference'],[390,844,'reduce']]){
      const context=await browser.newContext({viewport:{width,height},reducedMotion});
      const page=await context.newPage();let requests=0;
      page.on('pageerror',e=>report.errors.push(e.message));
      page.on('request',()=>requests++);
      page.on('response',r=>{if(r.status()>=400)report.errors.push(r.status()+' '+r.url());});
      page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
      await page.goto('http://127.0.0.1:8787/camera-study.html',{waitUntil:'domcontentloaded'});
      await page.waitForFunction(()=>window.liveCamera&&(window.liveCamera.state.loaded===48||window.liveCamera.state.error),null,{timeout:90000});
      let s=await page.evaluate(()=>window.liveCamera.state);assert(!s.error,s.error);assert(s.ready);assert.equal(s.loaded,48);
      assert.equal(await page.locator('video').count(),0,'The live camera does not switch prerecorded videos');
      const shot=async name=>{
        const file=name+'.jpg';await page.screenshot({path:path.join(out,file),type:'jpeg',quality:85});
        report.screenshots.push({file,viewport:{width,height},state:await page.evaluate(()=>window.liveCamera.state)});save();
      };
      await shot('live46-'+width+'-auto');
      const baseline=requests,phase=s.phase,cameras=[];
      for(const value of [.123,.124,.381,.672,.999]){
        await page.locator('#camera-distance').evaluate((el,v)=>{el.value=String(v);el.dispatchEvent(new Event('input',{bubbles:true}));},value);
        await page.waitForFunction(v=>Math.abs(window.liveCamera.state.renderedZoom-v)<1e-6,value);
        s=await page.evaluate(()=>window.liveCamera.state);assert.equal(s.mode,'manual');assert(Math.abs(s.zoom-value)<1e-6);
        cameras.push(s.camera);await shot('live46-'+width+'-'+String(value).replace('.','p'));
      }
      assert.notDeepEqual(cameras[0],cameras[1],'Neighboring fractional slider values produce different actual camera positions');
      assert.equal(requests,baseline,'Dragging a loaded scene performs zero network requests');
      if(reducedMotion==='reduce')assert.equal(s.phase,phase);
      else assert(s.phase!==phase,'Actions advance while the camera moves');
      if(s.running)await page.locator('#motion-toggle').click();
      const paused=await page.evaluate(()=>window.liveCamera.state.phase);
      await page.locator('#camera-distance').evaluate(el=>{el.value='.401';el.dispatchEvent(new Event('input',{bubbles:true}));});
      await page.waitForFunction(()=>Math.abs(window.liveCamera.state.renderedZoom-.401)<1e-6);
      assert.equal(await page.evaluate(()=>window.liveCamera.state.phase),paused);
      await shot('live46-'+width+'-paused-camera');
      await page.locator('#resume-auto').click();
      assert.equal(await page.evaluate(()=>window.liveCamera.state.mode),'auto');
      assert(Math.abs(await page.evaluate(()=>window.liveCamera.state.zoom)-.401)<1e-6,'Resume starts from current camera distance');
      await page.locator('#motion-toggle').click();
      await page.waitForFunction(p=>window.liveCamera.state.phase!==p,paused);
      await shot('live46-'+width+'-resume');
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      await context.close();
    }
    assert.equal(report.errors.length,0,JSON.stringify(report.errors));
    report.checks.push('Real WebGL renderer;48 actual source episodes; arbitrary neighboring camera positions; no network on drag; independent action/camera clocks; paused camera navigation; continuous auto resume; desktop/mobile/reduced motion');
    report.passed=true;
  }catch(e){report.failure=e.stack;process.exitCode=1;}
  finally{save();if(browser)await browser.close();console.log(JSON.stringify({passed:report.passed,failure:report.failure,errors:report.errors}));}
})();
