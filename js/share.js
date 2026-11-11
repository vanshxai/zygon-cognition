// Test-version sharing: with the person's consent, each question and the title of Zygon's reply are sent to the developers,
// tagged with a random anonymous ID. Attached files are never sent. If the server is not there, nothing breaks: the app works as before.
const VERSION='2026-11-11';
// Chats go to a write-only webhook (Make.com) that files them in a private data store; the address can be posted to but cannot be read back.
const ENDPOINT=(typeof window!=='undefined'&&window.ZC_COLLECT)||'https://hook.eu2.make.com/yd00g7cfw6qd0i7wnm7mzw7inwfxeky8';
const lsGet=(k,d)=>{try{return JSON.parse(localStorage.getItem(k))??d}catch{return d}};
const lsSet=(k,v)=>{try{localStorage.setItem(k,JSON.stringify(v))}catch{}};
const rnd=()=>{try{return crypto.randomUUID()}catch{return 'u'+Date.now().toString(36)+Math.random().toString(36).slice(2,10)}};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

const cleanName=v=>String(v??'').replace(/[\u0000-\u001f<>]/g,'').replace(/\s+/g,' ').trim().slice(0,60);
let uid=lsGet('zc_uid',null);if(!uid){uid=rnd();lsSet('zc_uid',uid)}
const sid=rnd().slice(0,8);                       // one id per page visit
let outbox=lsGet('zc_outbox',[]);
let timer=null,failures=0,paused=false;

// people sometimes type an email address or a phone number into a message: these are blanked before anything leaves the page
const redact=s=>String(s??'').replace(/[\w.+-]+@[\w-]+\.[\w.-]+/g,'[email]').replace(/\+?\d[\d\s().-]{8,}\d/g,m=>m.replace(/\D/g,'').length>=9?'[number]':m);
const persist=()=>{while(outbox.length>300)outbox.shift();let s=JSON.stringify(outbox);while(s.length>400000&&outbox.length>10){outbox.shift();s=JSON.stringify(outbox)}try{localStorage.setItem('zc_outbox',s)}catch{}};

function build(x){
  const r=x.r||{},meta=r.meta||{},m=x.m;
  const ev={t:Date.now(),sid,type:x.type||'turn',chat:x.chat,n:x.n,mode:x.mode};
  if(x.q!=null)ev.q=redact(x.q).slice(0,4000);
  if(meta.heard)ev.heard=redact(meta.heard).slice(0,1500);
  if(meta.ignored)ev.ignored=meta.ignored;
  if(meta.intent)ev.intent=meta.intent;
  if(meta.topic)ev.topic=meta.topic;
  if(x.r){ev.kind=r.kind||(r.table?'result':'message');ev.title=String(r.title||'').slice(0,120);
    if(r.subtitle)ev.sub=redact(r.subtitle).slice(0,160);
    if(r.note)ev.note=String(r.note).slice(0,160);
    if(r.questions?.length)ev.asks=r.questions.slice(0,6).map(s=>String(s).slice(0,100));
    if(r.options?.length)ev.options=r.options.slice(0,5);
    if(r.table?.rows?.length)ev.rows=r.table.rows.slice(0,4).map(row=>row.map(String).join(' | ').slice(0,100));
    if(meta.also?.length)ev.also=meta.also.map(a=>redact(a.text).slice(0,100));
    if(r.switch)ev.switch=r.switch}
  if(m){ev.ms=Math.round((m.route||0)+(m.tool||0));ev.path=m.path}
  try{ev.lang=navigator.language;ev.w=innerWidth}catch{}
  return ev}

async function post(events,beacon,op){
  const p=new URLSearchParams({uid,name:cleanName(lsGet('zc_name','')),v:VERSION,op:op||'events',sent:String(Date.now()),events:JSON.stringify(events)});   // a plain form post: works from any page, no preflight
  if(beacon&&navigator.sendBeacon){try{return navigator.sendBeacon(ENDPOINT,p)}catch{}}
  const res=await fetch(ENDPOINT,{method:'POST',body:p,keepalive:p.toString().length<60000});
  return res.ok}

async function flush(beacon=false){
  if(!Share.enabled()||paused||!outbox.length)return;
  const batch=outbox.slice(0,40);
  try{const ok=await post(batch,beacon);
    if(ok){outbox.splice(0,batch.length);persist();failures=0;if(outbox.length)schedule(1500)}
    else bad()}
  catch{bad()}}
const bad=()=>{failures++;if(failures>=4)paused=true;else schedule(5000*failures)};   // stop trying for this visit after a few failures (e.g. no server)
const schedule=ms=>{clearTimeout(timer);timer=setTimeout(()=>flush(),ms)};
if(typeof window!=='undefined')addEventListener('pagehide',()=>{if(Share.enabled()&&outbox.length)flush(true)});

export const Share={
  uid:()=>uid,shortId:()=>uid.slice(0,8),
  name:()=>cleanName(lsGet('zc_name','')),
  setName(v){lsSet('zc_name',cleanName(v));if(outbox.length)schedule(500)},
  enabled:()=>lsGet('zc_share',null)==='yes',
  needsConsent:()=>lsGet('zc_share',null)===null,
  setConsent(v){lsSet('zc_share',v?'yes':'no');if(v){paused=false;failures=0;if(outbox.length)schedule(500)}else{outbox=[];persist()}},
  track(x){if(!Share.enabled())return;try{outbox.push(build(x));persist();schedule(outbox.length>=8?1000:25000)}catch{}},   // sent in batches: fewer calls, and the free storage service counts calls
  async forget(){try{const ok=await post([],false,'forget');if(ok){outbox=[];persist();return true}}catch{}return false},
  askConsent(){
    const box=document.getElementById('consent'),body=document.getElementById('consentbody');if(!box||!body)return;
    body.innerHTML=`<p>Zygon.cognition is being tested. To make it better, your name and the messages you type here, with the titles of Zygon's replies, are sent to the developers. They are tagged with a random ID. No email or other details are asked.</p>
      <p>Attached files are never sent. Please do not type personal or confidential information. You can switch this off, and delete what was shared, in Settings at any time.</p>
      <input id="consentname" class="nameinput" maxlength="60" placeholder="Your name" autocomplete="name" value="${esc(Share.name())}">
      <div class="row"><button class="btn pri" id="consentyes">OK, share and continue</button><button class="btn" id="consentno">Use without sharing</button></div>`;
    box.hidden=false;const ni=document.getElementById('consentname');setTimeout(()=>ni.focus(),50);
    document.getElementById('consentyes').onclick=()=>{if(!cleanName(ni.value)){ni.classList.add('need');ni.focus();return}Share.setName(ni.value);Share.setConsent(true);box.hidden=true};
    document.getElementById('consentno').onclick=()=>{Share.setConsent(false);box.hidden=true}}
};
