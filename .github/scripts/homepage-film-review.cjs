const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process'),out=process.env.REVIEW_OUTPUT;fs.mkdirSync(out,{recursive:true});
const manifest=JSON.parse(fs.readFileSync('website/assets/film62/manifest.json','utf8'));
const report={commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),passed:false,errors:[],screenshots:[],checks:[],recordings:[]};
const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));let browser;
(async()=>{try{
 browser=await chromium.launch({channel:'chrome'});
 for(const target of ['index.html','research-preview-v31.html']){
  for(const [width,height] of [[1440,900],[1280,720],[1920,1080],[1724,756],[390,844],[320,640]]){
   const context=await browser.newContext({viewport:{width,height},reducedMotion:'reduce'}),page=await context.newPage();page.setDefaultTimeout(15000);page.on('pageerror',e=>report.errors.push(e.message));page.on('response',r=>{if(r.status()>=400)report.errors.push(r.status()+' '+r.url());});
   await page.goto('http://127.0.0.1:8787/'+target,{waitUntil:'domcontentloaded'});await page.evaluate(()=>document.fonts.ready);
   const video=page.locator('#hero-film');await page.waitForFunction(()=>document.getElementById('hero-film').readyState>=2,null,{timeout:45000});
   assert.equal(await video.getAttribute('data-src'),'assets/film62/generation-loop.mp4');assert.equal(await video.getAttribute('data-mobile-src'),null);assert.equal(await video.evaluate(v=>v.controls),false);
   assert.equal(await page.locator('#overview button, #overview input').count(),0);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   assert.equal(await video.evaluate(v=>v.paused),true,'Respect reduced motion');
   const bounds=await page.locator('#overview').boundingBox();assert(bounds.y+bounds.height<=height+1,JSON.stringify({target,width,height,bounds}));
   const title=await page.locator('#title').boundingBox();assert(title.x>=0&&title.x+title.width<=width+1);assert.match(await page.locator('#title').innerText(),/Learned Data Generation and Benchmarking for Bimanual Garment Flattening and Folding/);
   const style=await video.evaluate(v=>{const s=getComputedStyle(v);return {fit:s.objectFit,background:s.backgroundColor};});assert.deepEqual(style,{fit:'contain',background:'rgb(255, 255, 255)'});
   const time=width===1440?25.32:3;await video.evaluate(async(v,t)=>{v.currentTime=t;await new Promise(resolve=>v.addEventListener('seeked',resolve,{once:true}));},time);
   const file=target.replace('.html','')+'-'+width+'.jpg';await page.screenshot({path:path.join(out,file),type:'jpeg',quality:92});report.screenshots.push({file,target,width,height,bounds});
   await page.locator('#about summary').click();assert.match(await page.locator('#about').innerText(),/model-generated|generated in simulation/i);
   assert.deepEqual(await page.locator('main > section').evaluateAll(s=>s.map(x=>x.id)),['overview','highlights','use']);
   assert.deepEqual(await page.locator('.story-track > .story-card').evaluateAll(s=>s.map(x=>x.id)),['diversity','rgb','render']);
   assert.equal(await page.locator('#results, #transfer-results, #method, .story-track #use').count(),0);
   assert(await page.locator('body').evaluate(b=>b.classList.contains('research30')),'Keep the approved research design at the official entry');
   assert.equal(await page.locator('#diversity .atlas-overview a, #diversity a.atlas-overview').count(),0,'The garment overview remains presentation-only');
   report.checks.push({target,width,height,oneViewport:true,noHeroControls:true,whiteGround:true,reducedMotion:true});await context.close();save();
  }
 }
 for(const width of [1440,390]){
  const context=await browser.newContext({viewport:{width,height:900},reducedMotion:'reduce',hasTouch:width===390}),page=await context.newPage();page.setDefaultTimeout(15000);page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto('http://127.0.0.1:8787/index.html');await page.locator('#highlights').evaluate(e=>e.scrollIntoView({block:'start'}));
  const next=page.locator('[data-story-next]'),prev=page.locator('[data-story-prev]');
  assert(await prev.isDisabled());console.log('Gallery initial state',width,await page.evaluate(()=>window.storyCards.state));
  for(let i=0;i<3;i++){
   if(i)await next.click();await page.waitForFunction(i=>window.storyCards.state.active===i,i);
   const card=page.locator('.story-card').nth(i);const box=await card.boundingBox();assert(Math.abs(box.x+box.width/2-width/2)<2,'Center the selected card');
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   await page.waitForFunction(i=>[...document.querySelectorAll('.story-card')[i].querySelectorAll('img')].filter(img=>!img.closest('details:not([open])')).every(img=>img.complete&&img.naturalWidth>0),i,{timeout:15000});
   const file='card-'+width+'-'+i+'.jpg';await page.screenshot({path:path.join(out,file),type:'jpeg',quality:92});report.screenshots.push({file,width,card:i});
   assert.equal(await page.locator('.story-dots button[aria-current=true]').count(),1);
   assert.equal(await page.locator('video').evaluateAll(vs=>vs.filter(v=>!v.paused).length),0,'Reduced motion applies to card videos');
  }
  assert(await next.isDisabled());await prev.click();await page.waitForFunction(()=>window.storyCards.state.active===1);
  await page.locator('.story-dots button').nth(0).click();await page.waitForFunction(()=>window.storyCards.state.active===0);
  await page.locator('.story-track').focus();await page.keyboard.press('ArrowRight');await page.waitForFunction(()=>window.storyCards.state.active===1);
  if(width===1440){const box=await page.locator('.story-track').boundingBox();await page.mouse.move(width*.75,box.y+140);await page.mouse.down();await page.mouse.move(width*.2,box.y+140,{steps:12});await page.mouse.up();await page.waitForFunction(()=>window.storyCards.state.active===2);}
  else {const box=await page.locator('.story-track').boundingBox(),session=await context.newCDPSession(page),y=box.y+140;await session.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:width*.85,y}]});for(let i=1;i<=12;i++){await session.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:width*(.85-.7*i/12),y}]});await page.waitForTimeout(16);}await session.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await page.waitForFunction(()=>window.storyCards.state.active===2);await session.detach();}
  await page.locator('header a[href="#use"]').click();await page.waitForFunction(()=>document.querySelector('#use').getBoundingClientRect().top>=0&&document.querySelector('#use').getBoundingClientRect().top<200);
  assert.equal(await page.locator('#use').evaluate(e=>!!e.closest('.story-track')),false);
  assert(await page.locator('#use .code-card').evaluate(e=>Math.abs(parseFloat(getComputedStyle(e).height)-e.scrollHeight)<2),'The code card grows with its content');
  const codeFile='code-'+width+'.jpg';await page.screenshot({path:path.join(out,codeFile),type:'jpeg',quality:92});report.screenshots.push({file:codeFile,width,staticCode:true});
  await page.goto('http://127.0.0.1:8787/index.html#rgb');await page.waitForFunction(()=>window.storyCards.state.active===1);
  report.checks.push({width,horizontalCards:3,staticCode:true,arrows:true,dots:true,keyboard:true,deepLinks:true,noDocumentOverflow:true});await context.close();save();
 }
 for(const width of [1440,390]){
  const context=await browser.newContext({viewport:{width,height:900},recordVideo:{dir:path.join(out,'recordings'),size:{width,height:900}}}),page=await context.newPage();page.setDefaultTimeout(15000);page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto('http://127.0.0.1:8787/index.html');await page.waitForFunction(()=>{const v=document.getElementById('hero-film');return !v.paused&&v.currentTime>.1;},null,{timeout:45000});
  await page.evaluate(()=>{
   window.heroProbe={wrapped:false,gaps:[],seam:null};const v=document.getElementById('hero-film'),c=document.createElement('canvas');c.width=160;c.height=56;const ctx=c.getContext('2d',{willReadFrequently:true});let last,lastPixels;
   v.requestVideoFrameCallback(function sample(_now,f){ctx.drawImage(v,0,0,160,56);const p=ctx.getImageData(0,0,160,56).data;
    if(last!==undefined){if(f.mediaTime<last){let sum=0;for(let i=0;i<p.length;i+=4)for(let j=0;j<3;j++)sum+=Math.abs(p[i+j]-lastPixels[i+j]);window.heroProbe.wrapped=true;window.heroProbe.seam=sum/(160*56*3);}else window.heroProbe.gaps.push(f.mediaTime-last);}
    last=f.mediaTime;lastPixels=p;v.requestVideoFrameCallback(sample);
   });
  });
  const source=await page.locator('#hero-film').evaluate(v=>v.currentSrc);
  for(const c of manifest.cases){await page.waitForFunction(t=>document.getElementById('hero-film').currentTime>=t,c.first_frame/25+3,{timeout:30000});}
  await page.waitForFunction(()=>window.heroProbe.wrapped,null,{timeout:20000});const probe=await page.evaluate(()=>window.heroProbe);assert(probe.seam<8);assert(Math.max(...probe.gaps)<.161);assert.equal(await page.locator('#hero-film').evaluate(v=>v.currentSrc),source);
  await page.locator('header a[href="#rgb"]').click();await page.waitForFunction(()=>{const v=document.getElementById('rgb-film');return !v.paused&&v.currentTime>.2;},null,{timeout:30000});
  await page.locator('[data-story-next]').click();await page.waitForFunction(()=>{const v=document.getElementById('transfer-film');return window.storyCards.state.active===2&&!v.paused&&v.currentTime>.2;},null,{timeout:30000});await page.waitForFunction(()=>{const v=document.getElementById('hero-film');return v.paused;});
  assert((await page.locator('#transfer-film').evaluate(v=>v.currentSrc)).endsWith(width===390?'robots71/robot-transfer-mobile.mp4':'robots71/robot-transfer.mp4'));
  const robotCorner=await page.locator('#transfer-film').evaluate(v=>{const c=document.createElement('canvas');c.width=20;c.height=20;const x=c.getContext('2d');x.drawImage(v,0,0,20,20);return [...x.getImageData(0,0,1,1).data].slice(0,3);});assert(robotCorner.every(x=>x>=245),'The robot replay blends into white');
  await page.locator('#transfer-film').evaluate(v=>{window.robotWrapped=false;let last=v.currentTime;v.addEventListener('timeupdate',()=>{if(v.currentTime<last)window.robotWrapped=true;last=v.currentTime;});});await page.waitForFunction(()=>window.robotWrapped,null,{timeout:18000});
  await page.waitForFunction(()=>document.querySelectorAll('video[data-auto]').length&&[...document.querySelectorAll('video[data-auto]')].filter(v=>!v.paused).length===1);
  await page.locator('#hero-film').scrollIntoViewIfNeeded();await page.waitForFunction(()=>!document.getElementById('hero-film').paused);
  report.checks.push({width,fullLoop:true,seamMeanRgbDifference:probe.seam,maxFrameGap:Math.max(...probe.gaps),exclusiveAmbientPlayback:true});
  const recording=page.video();await context.close();const file='homepage-loop-'+width+'.webm';await recording.saveAs(path.join(out,file));report.recordings.push(file);save();
 }
 assert.deepEqual(report.errors,[]);report.passed=true;
}catch(e){report.errors.push(e.stack||String(e));process.exitCode=1;}finally{save();await browser?.close();}})();
