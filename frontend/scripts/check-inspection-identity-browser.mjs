import { createServer } from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';

// Read-only live verification: reject every non-GET request, including inspect.
const output = resolve('artifacts/identity-live');
await mkdir(output, { recursive: true });
const before = await (await fetch('http://127.0.0.1:8000/api/v1/component1/inspections?limit=1')).json();
const savedId = before.items[0].id;
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
  await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  const waitFor = async expression => {
    for(let i=0;i<100;i++) { if(await evaluate(expression))return; await new Promise(r=>setTimeout(r,100)); }
    throw Error(`Timed out: ${expression}`);
  };
  const verify = async name => {
    await waitFor(`document.body.innerText.includes('E02 Optimized YOLOv8n')`);
    const result = await evaluate(`({label:document.querySelector('dt:nth-of-type(2)')?.textContent,identity:document.querySelector('dd:nth-of-type(2)')?.textContent,technicalDetails:!!document.querySelector('details')})`);
    if(result.label!=='Detection Model' || result.identity!=='E02 Optimized YOLOv8n' || result.technicalDetails)throw Error(JSON.stringify(result));
    results.push({scenario:name,...result});
    const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
    await writeFile(resolve(output,`${name}.png`),Buffer.from(shot.data,'base64'));
  };
  await send('Page.navigate',{url:`http://127.0.0.1:4318/component1/inspections/${savedId}`});
  await verify('saved-inspection');
  await send('Page.reload',{ignoreCache:true});
  await new Promise(r=>setTimeout(r,500));
  await verify('reloaded-inspection');
  await send('Page.navigate',{url:'http://127.0.0.1:4318/component1/history'});
  await waitFor(`!!document.querySelector('a[aria-label^="View inspection from"]')`);
  await evaluate(`document.querySelector('a[aria-label^="View inspection from"]').click()`);
  await verify('history-inspection');
  const after=await (await fetch('http://127.0.0.1:8000/api/v1/component1/inspections?limit=1')).json();
  if(after.total!==before.total)throw Error('Inspection count changed during read-only verification');
  await writeFile(resolve(output,'browser-checks.json'),JSON.stringify({countBefore:before.total,countAfter:after.total,results},null,2));
  console.log(JSON.stringify({countBefore:before.total,countAfter:after.total,results},null,2));
  await send('Browser.close').catch(()=>{});
} finally { ws?.close();browser.kill();server.close(); }
