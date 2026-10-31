import { createServer } from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';

// Browser-only fixtures. No requests are sent to the real backend or database.
const output = resolve('artifacts/f1-6');
await mkdir(output, { recursive: true });
const fabric = '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600"><defs><pattern id="p" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#b7b8af"/><path d="M0 0H8M0 0V8" stroke="#989c92"/></pattern></defs><rect width="800" height="600" fill="url(#p)"/><text x="30" y="560" fill="#142235" font-size="22">UI TEST FIXTURE - not a model prediction</text></svg>';
const item = { id:'ui-fixture', detected_at:'2026-10-15T12:00:00Z', thumbnail_url:'/fixture.svg', total_defects:2, defect_classes:['stain','cut'], model_version:'UI test fixture' };
const inspection = { ...item, image_width:800, image_height:600, image_url:'/fixture.svg', inference_time_ms:42, summary:{total_defects:2,classes_found:['stain','cut']}, model:{model_version:'UI test fixture',confidence_threshold:0.25,iou_threshold:0.5,imgsz:640,device:'cpu',experiment_id:null,model_weights_hash:null}, detections:[{id:'d1',detection_order:0,class_id:6,class_name:'stain',confidence:.91,bbox:{x_min:100,y_min:100,x_max:300,y_max:250},pre_nms_index:42},{id:'d2',detection_order:1,class_id:3,class_name:'cut',confidence:.76,bbox:{x_min:480,y_min:200,x_max:700,y_max:400},pre_nms_index:84}] };
const server = createServer(async (req,res) => {
  const path = new URL(req.url,'http://localhost').pathname;
  if(path.startsWith('/api/')) {
    res.setHeader('Content-Type','application/json');
    res.end(JSON.stringify(path.includes('/detections/') ? [{id:'explanation-ui-fixture', detection_id:'d1', xai_method:'grad-cam', status:'completed', outcome:'success', overlay_url:'/fixture.svg', raw_heatmap_url:null, model_version:'UI fixture only', target_layer:null, inference_time_ms:null}] : path.endsWith('/ui-fixture') ? inspection : {total:1,items:[item],skip:0,limit:10})); return;
  }
  if(path === '/fixture.svg') { res.setHeader('Content-Type','image/svg+xml'); res.end(fabric); return; }
  try {
    const file = resolve('dist', path.startsWith('/assets/') ? path.slice(1) : 'index.html');
    res.setHeader('Content-Type', {'.js':'text/javascript','.css':'text/css','.html':'text/html'}[extname(file)] || 'application/octet-stream');
    res.end(await readFile(file));
  } catch { res.statusCode=404; res.end(); }
});
await new Promise(r=>server.listen(4318,'127.0.0.1',r));
const browser = spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--no-first-run','--remote-debugging-port=9328',`--user-data-dir=${resolve(tmpdir(),`fabric-ui-${Date.now()}`)}`,'about:blank'],{windowsHide:true,stdio:'ignore'});
let ws;
try {
  let target;
  for(let i=0;i<60;i++) { try { target=(await (await fetch('http://127.0.0.1:9328/json')).json()).find(t=>t.type==='page'); if(target) break; } catch {} await new Promise(r=>setTimeout(r,250)); }
  if(!target) throw Error('Browser debugging connection unavailable');
  ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise(r=>ws.addEventListener('open',r,{once:true}));
  let sequence=0; const pending=new Map();
  ws.addEventListener('message',e=>{const m=JSON.parse(e.data); if(m.id && pending.has(m.id)){ const {ok,fail}=pending.get(m.id);pending.delete(m.id);if(m.error) fail(Error(m.error.message)); else ok(m.result); }});
  const send=(method,params={})=>new Promise((ok,fail)=>{const id=++sequence;pending.set(id,{ok,fail});ws.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.text);return r.result.value;};
  await send('Page.enable');
  const results=[];
  for(const width of [1440,768,390]) {
    await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    for(const [name,path] of [['upload','/component1'],['results','/component1/inspections/ui-fixture'],['history','/component1/history']]) {
      await send('Page.navigate',{url:`http://127.0.0.1:4318${path}`});
      for(let i=0;i<50;i++){if(await evaluate(`!!document.querySelector('.panel') && !document.body.innerText.includes('Loading inspection')`))break;await new Promise(r=>setTimeout(r,100));}
      await new Promise(r=>setTimeout(r,200));
      let measurements=await evaluate(`({viewport:innerWidth,client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth,heading:document.querySelector('h1')?.textContent})`);
      if(measurements.scroll>measurements.client)throw Error(`${name} overflows at ${width}: ${measurements.scroll}`);
      if(name==='results') {
        const aligned=await evaluate(`(()=>{const img=document.querySelector('img[alt="Inspected fabric"]');const box=document.querySelector('button[aria-label^="Select stain"]');const a=img.getBoundingClientRect(),b=box.getBoundingClientRect();return Math.abs(b.left-a.left-a.width*100/800)<1 && Math.abs(b.width-a.width*200/800)<1})()`);
        if(!aligned) throw Error(`Bounding-box alignment failed at ${width}`);
        await evaluate(`document.querySelector('.detection-button').click()`);
        await new Promise(r=>setTimeout(r,150));
        measurements=await evaluate(`({viewport:innerWidth,client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth,heading:document.querySelector('h1')?.textContent})`);
        if(measurements.scroll>measurements.client)throw Error(`Selected result overflows at ${width}: ${measurements.scroll}/${measurements.client}`);
        const ratio=await evaluate(`(()=>{const img=document.querySelector('img[alt="XAI heatmap overlay"]');if(!img)return false;const r=img.getBoundingClientRect();return Math.abs(r.width/r.height-800/600)<.01 && getComputedStyle(img).opacity==='1'})()`);
        if(!ratio)throw Error(`Explanation aspect ratio or default opacity failed at ${width}`);
      }
      const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
      await writeFile(resolve(output,`${name}-${width}.png`),Buffer.from(shot.data,'base64'));
      results.push({page:name,width,...measurements,bboxAligned:name==='results'?true:undefined});
    }
  }
  await writeFile(resolve(output,'browser-checks.json'),JSON.stringify(results,null,2));
  console.log(JSON.stringify(results,null,2));
  await send('Browser.close').catch(()=>{});
} finally { ws?.close();browser.kill();server.close(); }
