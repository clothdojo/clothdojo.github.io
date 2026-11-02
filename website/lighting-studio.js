import * as T from './vendor/three-r180/three.module.min.js';
import {RoomEnvironment} from './vendor/three-r180/addons/environments/RoomEnvironment.js';
import {SSAOPass} from './vendor/three-r180/addons/postprocessing/SSAOPass.js';

export function createStudio(renderer,scene,camera,light,hemi,ground,root){
  const generator=new T.PMREMGenerator(renderer),room=new RoomEnvironment();
  const environment=generator.fromScene(room,.04);room.dispose();generator.dispose();
  const ao=new SSAOPass(scene,camera,Math.max(128,root.clientWidth*.5),Math.max(96,root.clientHeight*.5),12);
  ao.renderToScreen=true;ao.normalMaterial.side=T.DoubleSide;ao.kernelRadius=.065;
  ao.minDistance=.0015/(camera.far-camera.near);ao.maxDistance=.13/(camera.far-camera.near);
  ao.ssaoMaterial.uniforms.studyStrength={value:.45};
  ao.ssaoMaterial.fragmentShader='uniform float studyStrength;\n'+ao.ssaoMaterial.fragmentShader.replace('vec3( 1.0 - occlusion )','vec3( 1.0 - occlusion * studyStrength )');
  let mode='enhanced',fitKey='';const p=new T.Vector3();
  function apply(next){
    mode=next;fitKey='';const improved=mode==='enhanced';
    scene.environment=improved?environment.texture:null;scene.environmentIntensity=improved?.65:1;
    scene.environmentRotation.set(improved?Math.PI/2:0,0,0);
    hemi.intensity=improved?.35:1.8;light.intensity=improved?3.2:2.7;
    light.position.set(...(improved?[-3,-4,9]:[8,-12,22]));light.target.position.set(improved?0:-5,0,0);
    renderer.toneMappingExposure=improved?1.05:1.25;
    renderer.shadowMap.type=improved?T.VSMShadowMap:T.PCFSoftShadowMap;
    light.shadow.radius=improved?4:1;light.shadow.blurSamples=8;
    light.shadow.bias=improved?-.00005:-.00025;light.shadow.normalBias=improved?.002:.015;
    ground.material.opacity=improved?.24:.17;
    if(!improved)Object.assign(light.shadow.camera,{left:-19,right:19,top:19,bottom:-19,near:1,far:65});
    light.shadow.camera.updateProjectionMatrix();renderer.shadowMap.needsUpdate=true;
  }
  function fit(cells){
    if(mode!=='enhanced')return;
    const visible=cells.filter(c=>c.mesh.visible),key=visible.map(c=>c.meta.trajectory).join('|');
    if(key===fitKey)return;fitKey=key;
    light.updateMatrixWorld();light.target.updateMatrixWorld();light.shadow.updateMatrices(light);
    const c=light.shadow.camera,inv=c.matrixWorldInverse;
    let left=Infinity,right=-Infinity,bottom=Infinity,top=-Infinity,near=Infinity,far=-Infinity;
    for(const cell of visible)for(const point of cell.meta.bounds){
      p.set(...point).applyMatrix4(inv);left=Math.min(left,p.x);right=Math.max(right,p.x);bottom=Math.min(bottom,p.y);top=Math.max(top,p.y);near=Math.min(near,-p.z);far=Math.max(far,-p.z);
    }
    if(!Number.isFinite(left))return;
    Object.assign(c,{left:left-.2,right:right+.2,bottom:bottom-.2,top:top+.2,near:Math.max(.1,near-.5),far:far+.5});c.updateProjectionMatrix();renderer.shadowMap.needsUpdate=true;
  }
  apply(mode);
  return {apply,fit,resize(){ao.setSize(Math.max(128,Math.round(root.clientWidth*.5)),Math.max(96,Math.round(root.clientHeight*.5)));},render(zoom){
    renderer.render(scene,camera);
    const strength=.45*(1-T.MathUtils.smoothstep(zoom,.25,.5));
    if(mode==='enhanced'&&strength>.001){ao.ssaoMaterial.uniforms.studyStrength.value=strength;ao.render(renderer,null,null);}
  },get mode(){return mode;}};
}
