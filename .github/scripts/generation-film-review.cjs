const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process'),out=process.env.REVIEW_OUTPUT;fs.mkdirSync(out,{recursive:true});
const manifest=JSON.parse(fs.readFileSync('website/assets/film61/manifest.json','utf8'));
const report={commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),passed:false,errors:[],screenshots:[],checks:[],recordings:[]};
const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));let browser;
(async()=>{try{
 browser=await chromium.launch({channel:'chrome'});
 for(const width of [1440,390]){
  const context=await browser.newContext({viewport:{width,height:900},recordVideo:{dir:path.join(out,'recordings'),size:{width,height:900}}}),page=await context.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto('http://127.0.0.1:8787/distance-study.html');
  const video=page.locator('#generation-film');
  await page.waitForFunction(()=>{const v=document.getElementById('generation-film');return v.readyState>=3&&!v.paused&&v.currentTime>.15;},null,{timeout:45000});
  assert.equal(await page.locator('video').count(),1);assert.equal(await page.locator('button, input, video[controls]').count(),0);assert.equal(await page.locator('script').count(),0);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  const src=await video.evaluate(v=>v.currentSrc),metadata=await video.evaluate(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration,loop:v.loop,muted:v.muted}));
  assert.deepEqual(metadata,{width:1600,height:560,duration:manifest.duration_seconds,loop:true,muted:true});
  await page.evaluate(()=>{
   window.filmReview={wrapped:false,gaps:[],seam:null};const v=document.getElementById('generation-film'),c=document.createElement('canvas');c.width=160;c.height=56;const ctx=c.getContext('2d',{willReadFrequently:true});let lastTime,lastPixels;
   v.requestVideoFrameCallback(function sample(_now,frame){
    ctx.drawImage(v,0,0,160,56);const pixels=ctx.getImageData(0,0,160,56).data;
    if(lastTime!==undefined){
     if(frame.mediaTime<lastTime){let total=0;for(let i=0;i<pixels.length;i+=4)for(let j=0;j<3;j++)total+=Math.abs(pixels[i+j]-lastPixels[i+j]);window.filmReview.wrapped=true;window.filmReview.seam=total/(160*56*3);}
     else window.filmReview.gaps.push(frame.mediaTime-lastTime);
    }
    lastTime=frame.mediaTime;lastPixels=pixels;v.requestVideoFrameCallback(sample);
   });
  });
  for(const [name,time] of [['near',1],['middle',12],['far',15.4],['scale',22],['return',30.2]]){
   await page.waitForFunction(t=>document.getElementById('generation-film').currentTime>=t,time,{timeout:40000});
   const file='film-'+name+'-'+width+'.jpg';await page.screenshot({path:path.join(out,file),type:'jpeg',quality:92});report.screenshots.push({file,time:await video.evaluate(v=>v.currentTime)});save();
  }
  await page.waitForFunction(()=>window.filmReview.wrapped,null,{timeout:10000});
  const playback=await page.evaluate(()=>window.filmReview);assert(playback.seam<8,JSON.stringify(playback));assert(Math.max(...playback.gaps)<.161);assert.equal(await video.evaluate(v=>v.currentSrc),src);assert.equal(await video.evaluate(v=>v.paused),false);
  report.checks.push({width,metadata,loopSeamMeanRgbDifference:playback.seam,maxMediaFrameGap:Math.max(...playback.gaps)});
  const recording=page.video();await context.close();const file='generation-film-'+width+'.webm';await recording.saveAs(path.join(out,file));report.recordings.push(file);save();
 }
 assert.deepEqual(report.errors,[]);report.passed=true;
}catch(e){report.errors.push(e.stack||String(e));process.exitCode=1;}finally{save();await browser?.close();}})();
