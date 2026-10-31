import { createServer } from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';

// Read-only live verification: reject every non-GET request, including inspect.
const output = resolve('artifacts/evaluation-cleanup');
await mkdir(output, { recursive: true });
const before = await (await fetch('http://127.0.0.1:8000/api/v1/component1/inspections?limit=1')).json();
const target = JSON.parse(await readFile('artifacts/xai-live-target.json','utf8'));
const savedId = target.inspectionId;
const inspection = await (await fetch(`http://127.0.0.1:8000/api/v1/component1/inspections/${savedId}`)).json();
const selectedIndex = inspection.detections.findIndex(d=>d.id===target.detectionId);
const bbox = inspection.detections[selectedIndex].bbox;
const server = createServer(async (req,res) => {
  const path = new URL(req.url,'http://localhost').pathname;
  if(req.method !== 'GET') { res.statusCode=405;res.end('Read-only verification');return; }
  if(path.startsWith('/api/')) {
    try {
      const upstream = await fetch(`http://127.0.0.1:8000${req.url}`);
      res.statusCode=upstream.status;
      res.setHeader('Content-Type', upstream.headers.get('content-type') || 'application/octet-stream');
      res.end(Buffer.from(await upstream.arrayBuffer()));
    } catch { res.statusCode=502;res.end(); }
    return;
  }
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
  const waitFor = async expression => {
    for(let i=0;i<100;i++) { if(await evaluate(expression))return; await new Promise(r=>setTimeout(r,100)); }
    throw Error(`Timed out: ${expression}`);
  };
  for(const width of [1440,768,390]) {
    await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
    await send('Page.navigate',{url:`http://127.0.0.1:4318/component1/inspections/${savedId}`});
    await waitFor(`!!document.querySelector('.detection-button')`);
    await evaluate(`document.querySelectorAll('.detection-button')[${selectedIndex}].click()`);
    await waitFor(`!!document.querySelector('[aria-label="Display mode"]')`);
    for(const mode of ['Overlay','Original','Heatmap Only','Defect Zoom']) {
      await evaluate(`[...document.querySelectorAll('[aria-label="Display mode"] button')].find(b=>b.textContent===${JSON.stringify(mode)}).click()`);
      if(mode==='Heatmap Only') await waitFor(`!!document.querySelector('img[alt="Raw heatmap visualization"]')`);
      await waitFor(`[...document.querySelectorAll('.xai-image-plane img')].every(i=>i.complete && i.naturalWidth>0)`);
      const measured=await evaluate(`(()=>{
        const boxes=[...document.querySelectorAll('.xai-defect-box')];
        const box=${JSON.stringify(bbox)};
        const aligned=boxes.length===2 && boxes.every(el=>{
          const p=el.parentElement.getBoundingClientRect(),r=el.getBoundingClientRect();
          return Math.abs(r.left-p.left-p.width*box.x_min/${inspection.image_width})<1 && Math.abs(r.top-p.top-p.height*box.y_min/${inspection.image_height})<1 && Math.abs(r.width-p.width*(box.x_max-box.x_min)/${inspection.image_width})<1 && Math.abs(r.height-p.height*(box.y_max-box.y_min)/${inspection.image_height})<1;
        });
        return {aligned,scroll:document.documentElement.scrollWidth,client:document.documentElement.clientWidth,crops:[...document.querySelectorAll('.xai-frame')].map(f=>f.dataset.crop)};
      })()`);
      if(!measured.aligned || measured.scroll>measured.client || measured.crops[0]!==measured.crops[1])throw Error(JSON.stringify({width,mode,measured}));
      const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
      await writeFile(resolve(output,`${mode.replaceAll(' ','-')}-${width}.png`),Buffer.from(shot.data,'base64'));
      results.push({width,mode,...measured});
    }
  }
  for (const path of ['/component1','/component1/history','/component1/evaluations']) {
    await send('Page.navigate',{url:`http://127.0.0.1:4318${path}`});
    await waitFor(`!!document.querySelector('h1')`);
    const info=await evaluate(`({heading:document.querySelector('h1').textContent,research:!!document.querySelector('.research-nav'),links:[...document.querySelectorAll('[aria-label="Inspection navigation"] a')].map(a=>a.textContent)})`);
    if(info.research || JSON.stringify(info.links)!==JSON.stringify(['New Inspection','Inspection History']))throw Error(JSON.stringify(info));
    if(path==='/component1/evaluations' && info.heading!=='New Inspection')throw Error('Evaluation route still active');
    results.push({path,...info});
  }
  const after=await (await fetch('http://127.0.0.1:8000/api/v1/component1/inspections?limit=1')).json();
  if(after.total!==before.total)throw Error('Inspection count changed');
  await writeFile(resolve(output,'browser-checks.json'),JSON.stringify({inspectionId:savedId,detectionId:target.detectionId,method:'grad-cam',results},null,2));
  console.log(JSON.stringify(results,null,2));
  await send('Browser.close').catch(()=>{});
} finally { ws?.close();browser.kill();server.close(); }
