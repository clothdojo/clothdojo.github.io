import * as T from './vendor/three-r180/three.module.min.js';
const clothStudy=document.body.hasAttribute('data-cloth-study');
const sourceMaterialStudy=clothStudy||new URLSearchParams(location.search).get('material')==='source';
let clothMode=new URLSearchParams(location.search).get('cloth')==='baseline'?'baseline':'full',fidelityReady=!clothStudy,geometryUpdates=0;

const root=document.getElementById('camera-viewer'),canvas=document.getElementById('live-canvas');
const slider=document.getElementById('camera-distance'),resume=document.getElementById('resume-auto');
const toggle=document.getElementById('motion-toggle'),label=document.getElementById('camera-label'),status=document.getElementById('camera-status');
let mode='auto',running=!matchMedia('(prefers-reduced-motion: reduce)').matches;
let zoom=0,renderedZoom=0,motion=0,autoTime=0,loaded=0,ready=false,error='';
let frames=0,lastInput=0,inputToFrame=0,lastStamp=null,manifest,renderer,camera,scene,drawnCells=0;
const base='assets/live46/';const bodies=[],cells=[];
const vec=new T.Vector3(),qa=new T.Quaternion(),qb=new T.Quaternion(),matrix=new T.Matrix4(),scale=new T.Vector3();
const forward=new T.Vector3(),right=new T.Vector3(),up=new T.Vector3(),worldUp=new T.Vector3(0,0,1);
const eye=new T.Vector3(),target=new T.Vector3();
function ui(){
  label.textContent=mode==='auto'?'自动运镜':'手动 · '+Math.round(zoom*100)+'%';
  toggle.textContent=running?'暂停动作':'继续动作';resume.disabled=mode==='auto';resume.textContent=mode==='auto'?'自动运镜中':'恢复自动';
  slider.setAttribute('aria-valuetext',mode==='auto'?'自动运镜':Math.round(zoom*100)+'% 镜头距离');
  root.dataset.mode=mode;root.setAttribute('aria-busy',String(!ready));
}
const smooth=u=>u*u*(3-2*u);
function route(t){
  t%=24;if(t<3)return 0;if(t<8)return .5*smooth((t-3)/5);
  if(t<10)return .5;if(t<15)return .5+.5*smooth((t-10)/5);
  if(t<18)return 1;if(t<23)return 1-smooth((t-18)/5);return 0;
}
function inverseSmooth(v){let a=0,b=1;for(let i=0;i<24;i++){const m=(a+b)/2;if(smooth(m)<v)a=m;else b=m;}return(a+b)/2;}
function manual(){mode='manual';zoom=Number(slider.value);lastInput=performance.now();ui();}
slider.addEventListener('input',manual);slider.addEventListener('pointerdown',manual);
resume.addEventListener('click',()=>{mode='auto';autoTime=zoom<.5?3+5*inverseSmooth(zoom*2):10+5*inverseSmooth((zoom-.5)*2);ui();});
toggle.addEventListener('click',()=>{running=!running;ui();});
document.addEventListener('visibilitychange',()=>{lastStamp=null;});
async function binary(file,folder=base){
  const response=await fetch(folder+file);if(!response.ok)throw new Error('场景数据加载失败：'+file);
  const bytes=await response.arrayBuffer();
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
function attribute(buffer,spec){
  const type=spec.type.endsWith('f4')?Float32Array:spec.type.endsWith('u4')?Uint32Array:Uint8Array;
  return new type(buffer,spec.offset,spec.length);
}
function geometry(buffer,spec){
  const g=new T.BufferGeometry();g.setAttribute('position',new T.BufferAttribute(attribute(buffer,spec.position).slice(),3));
  g.setIndex(new T.BufferAttribute(attribute(buffer,spec.index).slice(),1));
  if(spec.uv)g.setAttribute('uv',new T.BufferAttribute(attribute(buffer,spec.uv).slice(),2));
  if(spec.sourceVertex)g.setAttribute('sourceVertex',new T.BufferAttribute(attribute(buffer,spec.sourceVertex).slice(),1));
  if(spec.color?.type)g.setAttribute('color',new T.BufferAttribute(attribute(buffer,spec.color).slice(),3,true));
  g.computeVertexNormals();return g;
}
function fadeMaterial(material,instanced){
  material.userData.fade={value:1};
  material.customProgramCacheKey=()=>instanced?'workcell-instance-fade':'workcell-cloth-fade';
  material.onBeforeCompile=shader=>{
    if(instanced){
      shader.vertexShader='attribute float cellAlpha; varying float vCellAlpha;\n'+shader.vertexShader;
      shader.vertexShader=shader.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\nvCellAlpha=cellAlpha;');
      shader.fragmentShader='varying float vCellAlpha;\n'+shader.fragmentShader;
    }else{
      shader.uniforms.cellFade=material.userData.fade;shader.fragmentShader='uniform float cellFade;\n'+shader.fragmentShader;
    }
    shader.fragmentShader=shader.fragmentShader.replace('#include <colorspace_fragment>','#include <colorspace_fragment>\ngl_FragColor.rgb=mix(vec3(1.0),gl_FragColor.rgb,'+(instanced?'vCellAlpha':'cellFade')+');');
  };
  return material;
}
function fullNormals(cell){
  const g=cell.mesh.geometry,p=g.attributes.position.array,n=g.attributes.normal.array,ids=g.attributes.sourceVertex.array,index=g.index.array;
  const sums=cell.normalSums||(cell.normalSums=new Float32Array(cell.meta.source_vertices*3));sums.fill(0);
  for(let j=0;j<index.length;j+=3){
    const a=index[j]*3,b=index[j+1]*3,c=index[j+2]*3;
    const ax=p[b]-p[a],ay=p[b+1]-p[a+1],az=p[b+2]-p[a+2],bx=p[c]-p[a],by=p[c+1]-p[a+1],bz=p[c+2]-p[a+2];
    const x=ay*bz-az*by,y=az*bx-ax*bz,z=ax*by-ay*bx;
    for(let k=0;k<3;k++){const v=ids[index[j+k]]*3;sums[v]+=x;sums[v+1]+=y;sums[v+2]+=z;}
  }
  for(let k=0;k<ids.length;k++){
    const v=ids[k]*3,d=Math.hypot(sums[v],sums[v+1],sums[v+2])||1;
    n[k*3]=sums[v]/d;n[k*3+1]=sums[v+1]/d;n[k*3+2]=sums[v+2]/d;
  }
  g.attributes.normal.needsUpdate=true;
}
function applyClothMode(){
  if(!fidelityReady)return;
  for(const i of [0,24]){
    const cell=cells[i];if(!cell.full)continue;
    cell.mesh.visible=false;const variant=clothMode==='full'?cell.full:cell.baseline;
    cell.meta=variant.meta;cell.mesh=variant.mesh;cell.data=variant.data;cell.geometryPhase=null;
  }
  document.querySelectorAll('[data-cloth-mode]').forEach(button=>{button.disabled=false;button.setAttribute('aria-pressed',String(button.dataset.clothMode===clothMode));});
}
function markReady(){
  if(cells[0].ready&&cells[24].ready&&fidelityReady&&!ready){ready=true;slider.disabled=false;toggle.disabled=false;ui();}
}
async function loadFullCloth(){
  const folder='assets/cloth48/',study=await(await fetch(folder+'manifest.json')).json();
  const loader=new T.TextureLoader();
  const maps=await Promise.all(['base_color','normal_gl','roughness'].map(async name=>{
    const texture=await loader.loadAsync(folder+study.textures[name].file);texture.wrapS=texture.wrapT=T.RepeatWrapping;
    texture.anisotropy=Math.min(4,renderer.capabilities.getMaxAnisotropy());texture.colorSpace=name==='base_color'?T.SRGBColorSpace:T.NoColorSpace;return texture;
  }));
  await Promise.all(study.cells.map(async meta=>{
    const cell=cells[meta.i],[buffer,animation]=await Promise.all([binary(meta.geometry,folder),binary(meta.file,folder)]);
    const g=geometry(buffer,meta);g.attributes.position.setUsage(T.DynamicDrawUsage);
    const material=fadeMaterial(new T.MeshPhysicalMaterial({color:new T.Color().setRGB(...meta.texture_tint),map:maps[0],normalMap:maps[1],normalScale:new T.Vector2(.35,.35),roughnessMap:maps[2],roughness:1,metalness:0,ior:1.5,specularIntensity:.375,side:T.DoubleSide}),false);
    const mesh=new T.Mesh(g,material);mesh.position.copy(cell.mesh.position);mesh.rotation.copy(cell.mesh.rotation);mesh.visible=false;mesh.frustumCulled=false;mesh.castShadow=true;mesh.receiveShadow=true;scene.add(mesh);
    const data=new Uint16Array(animation),stride=meta.vertices*3;
    if(data.length!==250*stride)throw new Error('完整网格轨迹长度不匹配');
    for(let k=stride;k<data.length;k++)data[k]=(data[k]+data[k-stride])&65535;
    cell.full={meta:{...cell.meta,...meta,full:true},mesh,data};
  }));
  // Both variants are already resident; changing material never seeks or fetches.
  await Promise.all([cells[0].baselineLoaded,cells[24].baselineLoaded]);
  for(const i of [0,24])cells[i].baseline={meta:cells[i].meta,mesh:cells[i].mesh,data:cells[i].data};
  fidelityReady=true;applyClothMode();markReady();
}
document.querySelectorAll('[data-cloth-mode]').forEach(button=>button.addEventListener('click',()=>{clothMode=button.dataset.clothMode;applyClothMode();}));
function resize(){
  const w=root.clientWidth,h=root.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();
}
function cameraAt(value){
  const keys=value<.5?['2','8']:['8','48'];const u=value<.5?value*2:(value-.5)*2;
  const a=manifest.cameras[keys[0]],b=manifest.cameras[keys[1]];
  const at=a.camera_target,bt=b.camera_target;
  target.set(...at).lerp(vec.set(...bt),u);
  const da=Math.hypot(...a.camera_eye.map((x,i)=>x-at[i])),db=Math.hypot(...b.camera_eye.map((x,i)=>x-bt[i]));
  const distance=Math.exp(Math.log(da)*(1-u)+Math.log(db)*u)*Math.max(1,(20/7)/camera.aspect);
  eye.set(.82,0,.5724).normalize().multiplyScalar(distance).add(target);
  camera.position.copy(eye);camera.lookAt(target);camera.updateMatrixWorld();
  forward.copy(target).sub(eye).normalize();right.crossVectors(forward,worldUp).normalize();up.crossVectors(right,forward);
  renderedZoom=value;
}
function visibility(cell){
  const w=root.clientWidth,h=root.clientHeight,tan=Math.tan(T.MathUtils.degToRad(camera.fov/2));
  let edge=Infinity;
  for(const p of cell.meta.bounds){
    const x=p[0]-eye.x,y=p[1]-eye.y,z=p[2]-eye.z,d=x*forward.x+y*forward.y+z*forward.z;
    if(d<=0)return 0;
    const px=(.5+(x*right.x+y*right.y+z*right.z)/(2*d*tan*camera.aspect))*w;
    const py=(.5-(x*up.x+y*up.y+z*up.z)/(2*d*tan))*h;
    edge=Math.min(edge,px,w-px,py,h-py);
  }
  return smooth(Math.max(0,Math.min(1,(edge-Math.max(6,22*w/1600))/Math.max(3,8*w/1600))));
}
function animateCell(cell,i,a,b,mix){
  const alpha=cell.ready?(i===0||i===24?1:visibility(cell)):0;const visible=alpha>.001;
  cell.mesh.visible=visible;cell.mesh.material.userData.fade.value=alpha;
  if(!visible)return;
  const slot=drawnCells++;
  const s=cell.meta,cs=cell.cos,sn=cell.sin,poses=cell.poses;
  for(let j=0;j<22;j++){
    const body=bodies[j];if(!body)continue;
    body.geometry.getAttribute('cellAlpha').array[slot]=alpha;
    if(j===21){vec.set(...s.translation);qa.copy(cell.yaw);}
    else{
      const p=(a*21+j)*7,q=(b*21+j)*7;
      const x=poses[p]*(1-mix)+poses[q]*mix,y=poses[p+1]*(1-mix)+poses[q+1]*mix;
      vec.set(cs*x-sn*y+s.translation[0],sn*x+cs*y+s.translation[1],poses[p+2]*(1-mix)+poses[q+2]*mix+s.translation[2]);
      qa.set(poses[p+3],poses[p+4],poses[p+5],poses[p+6]).normalize();qb.set(poses[q+3],poses[q+4],poses[q+5],poses[q+6]).normalize();
      qa.slerp(qb,mix).premultiply(cell.yaw);
    }
    scale.setScalar(1);matrix.compose(vec,qa,scale);body.setMatrixAt(slot,matrix);
  }
  const phase=a+mix;
  if(cell.geometryPhase===phase)return;
  cell.geometryPhase=phase;geometryUpdates++;
  const values=cell.data,out=cell.mesh.geometry.getAttribute('position'),stride=s.vertices*3,ia=a*stride,ib=b*stride;
  for(let k=0;k<stride;k++)out.array[k]=s.lo[k%3]+s.span[k%3]*(values[ia+k]*(1-mix)+values[ib+k]*mix)/65535;
  out.needsUpdate=true;if(s.full)fullNormals(cell);else cell.mesh.geometry.computeVertexNormals();
}
function draw(stamp){
  requestAnimationFrame(draw);
  if(!manifest||!renderer||!cells.length||document.hidden||(clothStudy&&!ready)){lastStamp=null;return;}
  const dt=lastStamp===null?0:Math.min(.1,(stamp-lastStamp)/1000);lastStamp=stamp;
  if(ready&&running){motion=(motion+dt)%10;if(mode==='auto'&&loaded===48)autoTime+=dt;}
  if(mode==='auto'){zoom=route(autoTime);slider.value=String(zoom);}
  cameraAt(zoom);
  const f=motion*25,a=Math.floor(f),b=Math.min(a+1,249),t=f-a;
  drawnCells=0;
  for(let i=0;i<cells.length;i++)animateCell(cells[i],i,a,b,t);
  for(const body of bodies)if(body){body.count=drawnCells;body.instanceMatrix.needsUpdate=true;body.geometry.getAttribute('cellAlpha').needsUpdate=true;}
  canvas.style.opacity=String(running?Math.min(1,motion/.16,(10-motion)/.16):1);
  renderer.render(scene,camera);frames++;
  if(lastInput){inputToFrame=performance.now()-lastInput;lastInput=0;}
}
async function loadCell(i){
  const cell=cells[i],[buffer,poseBuffer]=await Promise.all([binary(cell.meta.file),binary(cell.meta.poses)]),data=new Uint16Array(buffer),stride=cell.meta.vertices*3;
  if(data.length!==250*stride)throw new Error('轨迹数据长度不匹配');
  for(let k=stride;k<data.length;k++)data[k]=(data[k]+data[k-stride])&65535;
  cell.data=data;cell.poses=new Float32Array(poseBuffer);cell.ready=true;loaded++;
  markReady();
  status.textContent=!fidelityReady?'正在载入两件完整网格与棉布贴图…':loaded===48?'':'已载入 '+loaded+' / 48 个工作区；镜头可直接拖动';
}
async function boot(){
  const response=await fetch(base+'manifest.json');if(!response.ok)throw new Error('场景清单加载失败');manifest=await response.json();
  status.textContent='正在载入共享几何与动作…';
  const buffer=await binary(sourceMaterialStudy?'geometry-material.bin':manifest.geometry);
  renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:false,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio,innerWidth<700?1.25:1.5));renderer.setClearColor(0xffffff);
  renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.25;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;
  scene=new T.Scene();scene.background=new T.Color(0xffffff);
  camera=new T.PerspectiveCamera(2*Math.atan(18/(20/7)/35)*180/Math.PI,20/7,.01,400);camera.up.set(0,0,1);
  const hemi=new T.HemisphereLight(0xffffff,0xdde0e1,1.8);hemi.position.set(0,0,10);scene.add(hemi);
  const light=new T.DirectionalLight(0xffffff,2.7);light.position.set(8,-12,22);light.target.position.set(-5,0,0);light.castShadow=true;
  light.shadow.mapSize.set(innerWidth<700?1024:2048,innerWidth<700?1024:2048);Object.assign(light.shadow.camera,{left:-19,right:19,top:19,bottom:-19,near:1,far:65});light.shadow.bias=-.00025;light.shadow.normalBias=.015;
  scene.add(light,light.target);
  const ground=new T.Mesh(new T.PlaneGeometry(400,400),new T.ShadowMaterial({color:0x26333b,opacity:.17}));ground.position.z=-.025;ground.receiveShadow=true;scene.add(ground);
  for(const spec of manifest.prototypes){
    if(!spec.position.length){bodies.push(null);continue;}
    const g=geometry(buffer,spec);g.setAttribute('cellAlpha',new T.InstancedBufferAttribute(new Float32Array(48),1));
    const m=fadeMaterial(new T.MeshStandardMaterial({vertexColors:true,roughness:sourceMaterialStudy?.5:.58,metalness:sourceMaterialStudy?0:.25}),true);
    const body=new T.InstancedMesh(g,m,48);body.instanceMatrix.setUsage(T.DynamicDrawUsage);body.castShadow=true;body.receiveShadow=true;body.frustumCulled=false;scene.add(body);bodies.push(body);
  }
  for(const meta of manifest.cells){
    const g=geometry(buffer,meta);g.getAttribute('position').setUsage(T.DynamicDrawUsage);
    const m=fadeMaterial(new T.MeshStandardMaterial({color:new T.Color().setRGB(...meta.color),roughness:.97,metalness:0,side:T.DoubleSide}),false);
    const mesh=new T.Mesh(g,m);mesh.position.set(...meta.translation);mesh.rotation.z=T.MathUtils.degToRad(meta.yaw);mesh.visible=false;mesh.frustumCulled=false;mesh.castShadow=true;mesh.receiveShadow=true;scene.add(mesh);
    const angle=T.MathUtils.degToRad(meta.yaw);cells.push({meta,mesh,ready:false,data:null,cos:Math.cos(angle),sin:Math.sin(angle),yaw:new T.Quaternion().setFromAxisAngle(worldUp,angle)});
  }
  resize();new ResizeObserver(resize).observe(root);
  cells[0].baselineLoaded=loadCell(0);cells[24].baselineLoaded=loadCell(24);
  await Promise.all([cells[0].baselineLoaded,cells[24].baselineLoaded,clothStudy?loadFullCloth():Promise.resolve()]);
  const order=[1,25,6,30,7,31,...Array.from({length:48},(_,i)=>i).filter(i=>![0,24,1,25,6,30,7,31].includes(i))];
  let cursor=0;
  await Promise.all(Array.from({length:3},async()=>{while(cursor<order.length)await loadCell(order[cursor++]);}));
}
Object.defineProperty(window,'liveCamera',{value:{get state(){return {ready,loaded,mode,zoom,renderedZoom,phase:motion,running,frames,inputToFrameMs:inputToFrame,error,camera:camera?.position.toArray(),triangles:renderer?.info.render.triangles,clothMode:clothStudy?clothMode:null,fidelityReady,geometryUpdates,focusVertices:cells.filter((_,i)=>i===0||i===24).map(c=>c.meta.vertices)};},seekFrame(frame){if(!clothStudy||running||!ready)return;motion=Math.max(0,Math.min(249,frame))/25;}}});
ui();requestAnimationFrame(draw);boot().catch(e=>{error=String(e);status.textContent='实时场景未能载入，可打开上方视频版对照。';console.error(e);});
