const fs=require('node:fs'),assert=require('node:assert/strict');

module.exports=async function galleryPlayback(browser,report,save){
 const originalSources={
  'assets/gallery86/rgb.mp4':'assets/rgb76/ribbon.mp4',
  'assets/gallery86/rgb-mobile.mp4':'assets/rgb77/ribbon-mobile.mp4',
  'assets/gallery86/retarget.mp4':'assets/robots85/robot-transfer.mp4',
  'assets/gallery86/retarget-mobile.mp4':'assets/robots82/robot-transfer-mobile.mp4',
 };
 for(const width of [1440,390])for(const baseline of [true,false]){
  const context=await browser.newContext({viewport:{width,height:900},reducedMotion:'no-preference'});
  const page=await context.newPage(),cdp=await context.newCDPSession(page);
  await cdp.send('Network.enable');
  await cdp.send('Network.emulateNetworkConditions',{offline:false,latency:80,downloadThroughput:500000,uploadThroughput:125000});
  await cdp.send('Emulation.setCPUThrottlingRate',{rate:2});
  if(baseline){
   await page.route('**/index.html',route=>{
    let body=fs.readFileSync('website/index.html','utf8');
    for(const [current,original]of Object.entries(originalSources))body=body.replaceAll(current,original);
    return route.fulfill({body,contentType:'text/html'});
   });
   await page.route('**/site.js?*',route=>{
    // Frozen from d0e8827:website/site.js so shallow CI checkouts can compare delivery behavior.
    const body=fs.readFileSync('.github/scripts/fixtures/gallery-before-v86.js','utf8');
    return route.fulfill({body,contentType:'text/javascript'});
   });
  }
  await page.goto('http://127.0.0.1:8787/index.html',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>document.getElementById('hero-film').currentTime>.1,null,{timeout:30000});
  await page.evaluate(()=>{
   window.galleryProbes={};
   for(const id of ['rgb-film','transfer-film']){
    const v=document.getElementById(id),probe={frames:[],waiting:0,waitingAt:null,started:null,quality:null};
    window.galleryProbes[id]=probe;
    v.addEventListener('waiting',()=>{if(probe.started!==null&&!v.paused&&probe.waitingAt===null)probe.waitingAt=performance.now();});
    v.addEventListener('playing',()=>{if(probe.waitingAt!==null){probe.waiting+=performance.now()-probe.waitingAt;probe.waitingAt=null;}});
    v.requestVideoFrameCallback(function frame(now,meta){
     if(probe.started!==null&&!v.paused)probe.frames.push({wall:now,media:meta.mediaTime});
     v.requestVideoFrameCallback(frame);
    });
   }
   window.beginGalleryProbe=id=>{const p=window.galleryProbes[id],v=document.getElementById(id);p.frames=[];p.waiting=0;p.waitingAt=null;p.started=performance.now();p.quality=v.getVideoPlaybackQuality();};
   window.readGalleryProbe=id=>{
    const p=window.galleryProbes[id],now=performance.now(),v=document.getElementById(id),q=v.getVideoPlaybackQuality();
    const gaps=p.frames.slice(1).map((f,i)=>f.wall-p.frames[i].wall);
    if(p.frames.length)gaps.push(now-p.frames.at(-1).wall);
    return {id,startupMs:p.frames.length?p.frames[0].wall-p.started:null,frames:p.frames.length,
     maxWallGapMs:gaps.length?Math.max(...gaps):null,
     waitingMs:p.waiting+(p.waitingAt===null?0:now-p.waitingAt),
     droppedFrames:q.droppedVideoFrames-p.quality.droppedVideoFrames,
     totalFrames:q.totalVideoFrames-p.quality.totalVideoFrames,
     elapsedMs:now-p.started,readyState:v.readyState,paused:v.paused};
   };
  });
  const samples=[];
  for(const [id,action]of [['rgb-film','rgb'],['transfer-film','next'],['rgb-film','previous']]){
   await page.evaluate(id=>window.beginGalleryProbe(id),id);
   if(action==='rgb')await page.locator('header a[href="#rgb"]').click();
   else await page.locator(action==='next'?'[data-story-next]':'[data-story-prev]').click();
   await page.waitForTimeout(action==='previous'?2000:12000);
   const sample=await page.evaluate(id=>window.readGalleryProbe(id),id);samples.push(sample);
   const result={width,baseline,bandwidthMbps:4,latencyMs:80,cpuSlowdown:2,action,...sample};
   report.checks.push({galleryPlayback:result});console.log('Gallery playback',JSON.stringify(result));save();
   if(!baseline){
    assert(sample.startupMs!==null&&sample.startupMs<2000,JSON.stringify(result));
    assert(sample.frames>=(action==='previous'?20:180),JSON.stringify(result));
    assert(sample.maxWallGapMs<300,JSON.stringify(result));
   }
  }
  await context.close();
 }
};
