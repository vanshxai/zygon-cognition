import {intake,normalise} from './intake.js';
import {Hexonet} from './engine.js';
import {run,recalc,LABEL,SUGGEST_START,adv} from './tools.js';
import {f} from './calc.js';
import * as HOME from './home.js';
import {solve} from './formulas.js';
import {Share} from './share.js';
import {loadKB,findTopics,explainCue,knowledgeResult,fallbackResult,missResult,topicById,KMIN} from './knowledge.js';
const MODES={
 hvac:{name:'HVAC engineer',blurb:'Duct and pipe sizes, cooling load, motor breakers, running cost, invoices',tag:'for HVAC engineers',key:'',model:'data/model.json',suggest:'data/suggest.json',LABEL:{...LABEL,formula_solve:'Any physics, maths or engineering formula'},START:[...SUGGEST_START.slice(0,5),'Use the Reynolds number formula for water at 1 m/s in a 50 mm pipe'],run,later:Object.keys(adv),subtitle:'Engineering back-office work for HVAC. Ask in plain words.',
   cards:Object.keys(LABEL).filter(k=>!['greeting','help_me','out_of_scope','new_chat','undo','show_calc','followup_same','edit_value','export_doc'].includes(k)&&!k.startsWith('adv_')).concat('formula_solve')},
 home:{name:'Home electrician',blurb:'Electricity bill, lights and fans per room, MCB and wire sizes for Indian homes',tag:'for Indian homes',key:'_home',model:'data/model_home.json',suggest:'data/suggest_home.json',LABEL:HOME.HOME_LABEL,START:HOME.HOME_START,run:HOME.runHome,later:[],subtitle:'Electrical estimates for Indian homes. Ask in plain words.',cards:Object.keys(HOME.homeLabel)}};
let mode=(()=>{try{const m=JSON.parse(localStorage.getItem('zc_mode'));return MODES[m]?m:'hvac'}catch{return 'hvac'}})();
const M=()=>MODES[mode];
const recalcAny=r=>r.meta?.type==='homebill'?HOME.recalcHome(r):recalc(r);
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const lsGet=(k,d)=>{try{return JSON.parse(localStorage.getItem(k))??d}catch{return d}};const lsSet=(k,v)=>{try{localStorage.setItem(k,JSON.stringify(v))}catch{}};
const store={get:(k,d)=>lsGet('zc_store_'+k,d),set:(k,v)=>lsSet('zc_store_'+k,v)};
let modeReady=Promise.resolve();
let hx,suggestList=[],chats=lsGet('zc_chats',[]),cur=null,files={}, undo=[], busy=false, sugIdx=0, showCalc=false, previewFmt=null;
const lib={};const uid=()=>Date.now().toString(36)+Math.random().toString(36).slice(2,5);const tokens=s=>Math.max(1,Math.ceil(String(s||'').length/4));
const loadScript=src=>lib[src]??=new Promise((ok,no)=>{const s=document.createElement('script');s.src=src;s.onload=ok;s.onerror=()=>no(new Error('Could not load '+src));document.head.appendChild(s)});

// ---------- chats ----------
const MEM_MB=()=>{const v=+lsGet('zc_mem_mb',5);return v>0?v:5};   // chat memory limit in MB; 5 is the default, raise it later (browsers allow about 5 MB per site in this storage)
const save=()=>{if(chats.length>KEEP*2){const keep=new Set(chats.slice(0,KEEP).map(c=>c.id));chats=chats.filter(c=>keep.has(c.id)||c.id===cur?.id)}
  const pack=()=>JSON.stringify(chats.map(c=>({...c,fileNames:(files[c.id]?.all||[]).map(x=>x.name)})));let s=pack();const cap=MEM_MB()*1e6-3e5;   // 0.3 MB stays free for the other memories
  while(s.length>cap){   // over the limit: the oldest chats go first; if only one chat is left, its oldest replies go
    if(chats.length>1)chats.pop();
    else{const it=chats[0].items||[];if(it.length<2)break;const k=Math.min(it.length-1,Math.ceil(it.length*(1-cap/s.length))+1);it.splice(0,k);chats[0].active=Math.max(0,Math.min((chats[0].active||0)-k,it.length-1))}
    s=pack()}
  for(let k=0;k<50;k++){try{localStorage.setItem('zc_chats',s);return}catch(e){if(chats.length<=1)return;chats.pop();s=pack()}}}
function newChat(){const c={id:uid(),title:'New chat',mode:null,last:null,items:[],active:0,ts:Date.now()};chats.unshift(c);files[c.id]={all:[],sheets:[],pdfs:[]};open(c.id)}
function open(id){cur=chats.find(c=>c.id===id);files[cur.id]??={all:[],sheets:[],pdfs:[]};cur.items??=cur.last?.result?[{id:uid(),query:cur.last.query,result:cur.last.result,m:null}]:[];undo=[];previewFmt=null;
  if(cur.mode==null&&cur.items.length)cur.mode=mode;   // chats saved before modes existed belong to the mode in use
  if(cur.mode&&cur.mode!==mode){mode=cur.mode;try{localStorage.setItem('zc_mode',JSON.stringify(mode))}catch{}modeReady=loadMode()}   // opening a chat switches to its mode
  renderList();renderChips();$('#app').classList.remove('open');
  if(cur.items.length){cur.active=Math.min(cur.active??cur.items.length-1,cur.items.length-1);const it=cur.items[cur.active];cur.last={query:it.query,result:it.result};paint(it.query,it.result)}else{cur.last=null;empty()}renderSide()}
function activate(i){if(busy)return;cur.active=i;const it=cur.items[i];cur.last={query:it.query,result:it.result};previewFmt=null;undo=[];paint(it.query,it.result);renderSide()}
let showAllChats=false;const KEEP=500;
function renderList(){const ord=[];const add=(c,d)=>{ord.push([c,d]);chats.filter(k=>k.parent===c.id).forEach(k=>add(k,d+1))};const tops=chats.filter(c=>!c.parent||!chats.some(p=>p.id===c.parent));
  const shown=showAllChats?tops:tops.slice(0,15);if(!showAllChats&&cur&&!shown.includes(cur)){const r=tops.find(t=>t===cur||chats.some(k=>k.id===cur.id&&k.parent===t.id));if(r)shown.push(r)}
  shown.forEach(c=>add(c,0));
  $('#chatlist').innerHTML=ord.map(([c,d])=>`<div class="chatitem ${c.id===cur?.id?'on':''}" style="padding-left:${10+d*14}px" data-id="${c.id}" title="${esc(c.title)}">${d?'↳ ':''}${esc(c.title)}${c.mode?`<span class="cmode">${c.mode==='home'?'Home':'HVAC'}</span>`:''}<span class="x" data-del="${c.id}" title="Delete this chat">×</span></div>`).join('')+(tops.length>15?`<button class="morechats" id="morechats">${showAllChats?'Show fewer':'Show all '+tops.length+' chats'}</button>`:'');
  document.querySelectorAll('.chatitem').forEach(el=>el.onclick=e=>{if(e.target.dataset.del){e.stopPropagation();delChat(e.target.dataset.del)}else open(el.dataset.id)});
  $('#morechats')&&($('#morechats').onclick=()=>{showAllChats=!showAllChats;renderList()})}
function delChat(id){const ids=new Set([id]);chats.filter(k=>k.parent===id).forEach(k=>ids.add(k.id));chats=chats.filter(c=>!ids.has(c.id));ids.forEach(i=>delete files[i]);if(cur&&ids.has(cur.id)){if(chats.length)open(chats[0].id);else newChat()}else renderList();save()}
function renderChips(){$('#chips').innerHTML=(files[cur.id]?.all||[]).map(x=>`<span class="chip">${esc(x.name)}</span>`).join('')}

// ---------- stage ----------
const stage=$('#stage');
function empty(){
  if(!cur?.mode){   // a new chat asks which mode first
    stage.innerHTML=`<div class="wrap empty"><div class="logo big"><span class="z">Zygon</span><span class="dot">.</span><span class="c">cognition</span></div><p class="cap">What do you need help with?</p><div class="modes">${Object.entries(MODES).map(([k,v])=>`<button class="modecard ${k===mode?'last':''}" data-pickmode="${k}"><b>${esc(v.name)}</b><span>${esc(v.blurb)}</span>${k===mode?'<em>last used</em>':''}</button>`).join('')}</div></div>`;
    stage.querySelectorAll('[data-pickmode]').forEach(b=>b.onclick=()=>pickMode(b.dataset.pickmode));updateGhost();return}
  stage.innerHTML=`<div class="wrap empty"><button class="tag" id="chmode" title="Choose a different mode">${esc(M().tag)} · change</button><div class="logo big"><span class="z">Zygon</span><span class="dot">.</span><span class="c">cognition</span></div><p class="cap">Engineering back-office work, done. Ask in plain words or attach your files.</p><div class="sugs">${M().START.slice(0,6).map(s=>`<button class="sug">${esc(s)}</button>`).join('')}</div></div>`;
  stage.querySelectorAll('.sug').forEach(b=>b.onclick=()=>typeIn(b.textContent));$('#chmode').onclick=()=>{if(busy||cur.items.length)return;cur.mode=null;empty()};updateGhost()}
async function showTopic(id){const t=topicById(id);if(!t||busy)return;await modeReady;const r=knowledgeResult(t,mode);r.meta={topic:id,intent:'explain'};commit(t.title,r,null,{tin:0,tout:tokens(r.short),route:0,tool:0,path:'explainer'})}
async function pickMode(m){if(!MODES[m]||busy)return;cur.mode=m;if(m!==mode){mode=m;try{localStorage.setItem('zc_mode',JSON.stringify(m))}catch{}modeReady=loadMode()}await modeReady;save();renderList();empty();q.focus();Share.track({type:'mode',chat:cur.id,mode:m})}
function waving(label='Working…'){stage.innerHTML=`<div class="wrap"><div class="wave">${'<i></i>'.repeat(17)}</div><div class="wlabel">${label}</div></div>`}
const totals=r=>{const sub=r.table.rows.reduce((a,x)=>a+(+x[r.table.amountCol]||0),0);const tax=sub*(r.table.totals?.tax||0)/100;return {sub,tax,total:sub+tax}};
function tableHTML(r){const t=r.table;if(!t)return'';const num=v=>typeof v==='number';
  let body=t.rows.map((row,ri)=>`<tr class="${t.totalRow&&ri===t.rows.length-1?'tot':''}">${row.map((c,ci)=>{const ed=t.editableCols?.includes(ci)&&!(t.totalRow&&ri===t.rows.length-1);return `<td class="${num(c)?'num':''} ${ed?'ed':''}" ${ed?`contenteditable="true" data-r="${ri}" data-c="${ci}"`:''}>${num(c)?esc(f(c,2)):esc(c)}</td>`}).join('')}</tr>`).join('');
  if(r.meta?.type==='invoice'){const T=totals(r);const pad=t.columns.length-2;body+=[['Subtotal',T.sub],[`Tax ${t.totals.tax}%`,T.tax],['Total',T.total]].map(([a,b],i)=>`<tr class="${i==2?'tot':''}"><td colspan="${pad}" class="num">${a}</td><td class="num" colspan="2">${f(b)}</td></tr>`).join('')}
  return `<div class="tbl"><table><thead><tr>${t.columns.map((c,i)=>`<th class="${t.rows.length&&typeof t.rows[0][i]==='number'?'num':''}">${esc(c)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>`}
function paint(query,r,flash){
  const wc=query?query.trim().split(/\s+/).length:0;const said=query?`<div class="said"><b>You</b><span>${esc(wc>40?query.slice(0,220)+'… ('+wc+' words)':query)}</span></div>`+(r.meta?.heard?`<div class="heard"><b>Understood as</b><span>${esc(r.meta.heard)}</span>${r.meta.ignored?`<em>${r.meta.ignored} other sentence${r.meta.ignored>1?'s':''} set aside</em>`:''}</div>`:''):'';let inner='';
  if(r.kind==='help'){inner=`<h2>What I can do right now</h2><p class="sub">${esc(r.subtitle)}</p><div class="grid">${M().cards.map(k=>`<button class="mini go" data-open="${k}"><b>${esc(M().LABEL[k])}</b><span>tap to open</span></button>`).join('')}${M().later.map(k=>`<div class="mini soon"><b>${esc(M().LABEL[k])}</b><span>coming later</span></div>`).join('')}</div><p class="sub" style="margin-top:14px">${mode==='hvac'?'Attach PDFs or Excel sheets with the paperclip. ':''}Edit any highlighted cell, or say "the second number should be 12".</p>`}
  else if(r.kind==='advanced'){inner=`<span class="badge">ADVANCED · NOT IN THIS PROTOTYPE</span><h2>${esc(r.title)}</h2><p class="sub">${esc(r.subtitle)}</p>`}
  else if(r.kind==='ask'){inner=`<h2>${esc(r.title)}</h2><p class="sub">${esc(r.hint||'I need a little more to do this properly:')}</p><ol class="qlist">${r.questions.map(q=>`<li>${esc(q)}</li>`).join('')}</ol>`}
  else if(r.kind==='know'){inner=`<h2>${esc(r.title)}</h2><p class="kshort">${esc(r.short)}</p>${r.points?.length?`<ul class="kpts">${r.points.map(p=>`<li>${esc(p)}</li>`).join('')}</ul>`:''}${r.numbers?.length?`<div class="tbl"><table><tbody>${r.numbers.map(n=>`<tr><td>${esc(n[0])}</td><td class="num">${esc(n[1])}</td></tr>`).join('')}</tbody></table></div>`:''}${r.tryit?.length?`<div class="krow"><span class="pvl">Try it</span>${r.tryit.map(t=>`<button class="btn" data-try="${esc(t)}">${esc(t.length>60?t.slice(0,58)+'…':t)}</button>`).join('')}</div>`:''}${r.topics?.length?`<div class="krow"><span class="pvl">Learn about</span>${r.topics.map(t=>`<button class="btn" data-topic="${esc(t.id)}">${esc(t.title)}</button>`).join('')}</div>`:''}${r.related?.length?`<div class="krow"><span class="pvl">Related</span>${r.related.map(t=>`<button class="btn" data-topic="${esc(t.id)}">${esc(t.title)}</button>`).join('')}</div>`:''}${r.source?`<div class="srcline">More: <a href="${esc(r.source.url)}" target="_blank" rel="noopener">${esc(r.source.title)}</a></div>`:''}${r.points?.length?`<div class="note">General guidance to understand the idea. For real work, check the equipment nameplate and your local code.</div>`:''}`}
  else if(r.kind==='choose'){inner=`<h2>Which did you mean?</h2><p class="sub">I'm not sure. Pick one and I will remember it.</p><div class="row">${r.options.map(o=>`<button class="btn" data-pick="${o}">${esc(M().LABEL[o]||o)}</button>`).join('')}</div>`}
  else{inner=`<h2>${esc(r.title)}</h2><p class="sub">${esc(r.subtitle||'')}</p>${tableHTML(r)}${r.note?`<div class="note">${esc(r.note)}</div>`:''}${r.source?`<div class="srcline">Source: <a href="${esc(r.source.url)}" target="_blank" rel="noopener">${esc(r.source.title)}</a>${r.source.license?' · '+esc(r.source.license):''}</div>`:''}${r.steps?.length?`<div class="steps ${showCalc?'on':''}" id="steps"><b>Calculation</b><ol>${r.steps.map(s=>`<li>${esc(s)}</li>`).join('')}</ol></div>`:''}
   <div class="row">${r.table?`<span class="pvl">Preview</span>`+Object.entries(FM).map(([k,v])=>`<button class="btn ${previewFmt===k?'on':''}" data-pv="${k}">${v}</button>`).join(''):''}${r.steps?.length?`<button class="btn ${showCalc?'on':''}" id="calcbtn">${showCalc?'Hide':'Show'} calculation</button>`:''}${undo.length?`<button class="btn" id="undobtn">Undo</button>`:''}</div>${previewPanel(r)}`}
  if(r.switch&&MODES[r.switch])inner=`<div class="row switchrow"><span class="pvl">This sounds like ${esc(MODES[r.switch].name)} mode</span><button class="btn pri" data-switch="${r.switch}">Switch to it</button></div>`+inner;
  const alsoH=r.meta?.also?.length?`<div class="also"><b>I also heard</b>${r.meta.also.map((x,i)=>`<button class="btn" data-also="${i}">${esc(x.text.length>70?x.text.slice(0,68)+'…':x.text)}</button>`).join('')}</div>`:'';
  stage.innerHTML=`<div class="wrap">${said}<div class="card">${inner}</div>${alsoH}</div>`;
  stage.querySelectorAll('[data-try]').forEach(b=>b.onclick=()=>submit(b.dataset.try));
  stage.querySelectorAll('[data-topic]').forEach(b=>b.onclick=()=>showTopic(b.dataset.topic));
  stage.querySelectorAll('[data-switch]').forEach(b=>b.onclick=()=>setMode(b.dataset.switch));
  stage.querySelectorAll('[data-also]').forEach(b=>b.onclick=()=>submit(r.meta.also[+b.dataset.also].text));
  stage.querySelectorAll('[data-x]').forEach(b=>b.onclick=()=>exportAs(b.dataset.x,r));
  stage.querySelectorAll('[data-pick]').forEach(b=>b.onclick=()=>pick(r.meta?.prompt||query,b.dataset.pick));
  stage.querySelectorAll('[data-open]').forEach(b=>b.onclick=()=>openSub(b.dataset.open));
  stage.querySelectorAll('[data-pv]').forEach(b=>b.onclick=()=>{const k=b.dataset.pv;previewFmt=(k&&previewFmt!==k)?k:null;paint(query,r);if(previewFmt)stage.querySelector('.pv')?.scrollIntoView({behavior:'smooth',block:'start'})});
  $('#calcbtn')&&($('#calcbtn').onclick=()=>toggleCalc());$('#undobtn')&&($('#undobtn').onclick=()=>doUndo());
  stage.querySelectorAll('td.ed').forEach(td=>{td.onblur=()=>{const v=parseFloat(td.textContent.replace(/,/g,''));if(isNaN(v))return;const r0=+td.dataset.r,c0=+td.dataset.c;if(r.table.rows[r0][c0]===v)return;pushUndo(r);r.table.rows[r0][c0]=v;recalcAny(r);commit(query,r,[r0,c0],null,true)};td.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();td.blur()}}});
  if(flash){const td=stage.querySelector(`td[data-r="${flash[0]}"][data-c="${flash[1]}"]`);td&&td.classList.add('flash')}
  stage.scrollTop=0}
const clone=o=>JSON.parse(JSON.stringify(o));
const pushUndo=r=>{undo.push(clone(r));if(undo.length>20)undo.shift()};
function commit(query,r,flash,m,replace){const it=replace&&cur.items[cur.active];
  if(it){it.query=query;it.result=clone(r);if(m)it.m=m}else{cur.items.push({id:uid(),query,result:clone(r),m:m||null});if(cur.items.length>2000)cur.items.shift();cur.active=cur.items.length-1}
  const a=cur.items[cur.active];cur.last={query:a.query,result:a.result};cur.ts=Date.now();save();paint(query,r,flash);renderSide();
  if(!it||m)Share.track({type:it?'teach':'turn',chat:cur.id,n:cur.items.length,mode:cur.mode||mode,q:query,r,m})}
function toggleCalc(){showCalc=!showCalc;const s=$('#steps');s&&s.classList.toggle('on',showCalc);const b=$('#calcbtn');if(b){b.textContent=(showCalc?'Hide':'Show')+' calculation';b.classList.toggle('on',showCalc)}}
function doUndo(){if(!undo.length)return;const prev=undo.pop();commit(cur.last.query,prev,null,null,true)}

// ---------- metrics, timeline, sub-chats, previews ----------
const FM={pdf:'PDF',xlsx:'Excel',pptx:'PowerPoint',csv:'CSV'};
const titleOf=r=>({help:'What I can do',choose:'Which did you mean?'}[r.kind])||r.title||'Reply';
const learnedFor=(t,intent)=>hx.words(t).some(w=>(hx.V[w]?.[intent]||0)>0||(hx.Q[w]?.[intent]||0)>0);
function outText(r){const a=[r.title,r.subtitle,r.note,r.hint,...(r.questions||[])];if(r.table)a.push(...r.table.columns,...r.table.rows.flat());if(r.kind==='help')a.push(Object.values(M().LABEL).join(' '));return a.filter(x=>x!=null).join(' ')}
const ms=v=>{if(v==null)return'–';const t=v/1000;return (t<0.01?t.toFixed(4):t.toFixed(2))+' s'};
function renderSide(){
  const items=cur.items||[];
  if(!$('#ctxpop').hidden)buildCtx();
  const kids=chats.filter(c=>c.parent===cur.id),par=cur.parent&&chats.find(c=>c.id===cur.parent);
  $('#rail').innerHTML=`<h4>This chat</h4><div class="tl">${items.map((it,i)=>`<button class="rit ${i===cur.active?'on':''}" data-i="${i}" title="${esc(it.query)}"><i></i><span class="t">${esc(titleOf(it.result))}</span><span class="q">${esc(it.query)}</span></button>`).join('')||'<div class="mnote">Replies line up here. Click one to bring it back.</div>'}</div>`
   +(par?`<h4>Opened from</h4><button class="rit br" data-chat="${par.id}">↰ ${esc(par.title)}</button>`:'')+(kids.length?`<h4>Branches</h4>${kids.map(c=>`<button class="rit br" data-chat="${c.id}">↳ ${esc(c.title)}</button>`).join('')}`:'');
  $('#rail').querySelectorAll('[data-i]').forEach(b=>b.onclick=()=>activate(+b.dataset.i));
  $('#rail').querySelectorAll('[data-chat]').forEach(b=>b.onclick=()=>open(b.dataset.chat));
  $('#rail .rit.on')?.scrollIntoView({block:'nearest'})}

async function openSub(intent){
  if(busy||!M().LABEL[intent])return;const parent=cur,label=M().LABEL[intent];
  const old=chats.find(k=>k.parent===parent.id&&k.topic===intent);if(old){open(old.id);return}
  const c={id:uid(),title:label,mode:parent.mode||mode,parent:parent.id,topic:intent,last:null,items:[],active:0,ts:Date.now()};chats.unshift(c);files[c.id]={all:[],sheets:[],pdfs:[]};
  const links=lsGet('zc_links',[]);links.push({from:parent.id,to:c.id,topic:intent,ts:Date.now()});lsSet('zc_links',links);
  open(c.id);busy=true;waving('Opening '+label+'…');await new Promise(r=>setTimeout(r,450));
  let res;const t1=performance.now();
  try{res=await dispatch(intent,label,{files:[],sheets:[],pdfs:[],store,current:null})}catch(e){res={title:'Something went wrong',subtitle:String(e.message||e),table:null,steps:[]}}
  const toolMs=performance.now()-t1;busy=false;
  if(res)commit(label,res,null,{tin:tokens(label),tout:tokens(outText(res)),route:0,tool:toolMs,path:'opened from a card'});else renderSide();renderList()}

function previewPanel(r){if(!previewFmt||!r.table)return'';
  return `<div class="pv"><div class="pvh"><span>Preview · ${FM[previewFmt]} · nothing is downloaded yet</span><span><button class="btn pri" data-x="${previewFmt}">Download ${FM[previewFmt]}</button><button class="btn" data-pv="">Close</button></span></div><div class="pvb">${previewHTML(previewFmt,r)}</div></div>`}
function previewHTML(k,r){
  const cols=r.table.columns,rows=exportRows(r),cell=v=>typeof v==='number'?esc(f(v)):esc(v),steps=r.steps||[];
  const pt=`<table class="pt"><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(x=>`<tr>${x.map(v=>`<td class="${typeof v==='number'?'num':''}">${cell(v)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  if(k==='csv')return `<pre class="csvp">${esc([cols,...rows].map(x=>x.map(v=>`"${String(v).replace(/"/g,'""')}"`).join(',')).join('\n'))}</pre>`;
  if(k==='xlsx'){const L=i=>String.fromCharCode(65+i);
    const grid=(head,body)=>`<table class="xl"><thead><tr><th></th>${head.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`;
    const b1=[cols,...rows].map((x,ri)=>`<tr><th>${ri+1}</th>${x.map(v=>`<td class="${ri===0?'xh ':''}${typeof v==='number'?'num':''}">${cell(v)}</td>`).join('')}</tr>`).join('');
    let out=`<div class="sheetname">Sheet “Result”</div>${grid(cols.map((_,i)=>L(i)),b1)}`;
    if(steps.length)out+=`<div class="sheetname">Sheet “Calculation”</div>${grid(['A'],steps.map((s,i)=>`<tr><th>${i+1}</th><td>${esc(s)}</td></tr>`).join(''))}`;
    return out}
  if(k==='pptx')return `<div class="slide"><div class="st">${esc(r.title)}</div><div class="ss">${esc(r.subtitle||'')}</div>${pt}</div>`+(steps.length?`<div class="slide"><div class="st">Calculation</div><ul>${steps.map(s=>`<li>${esc(s)}</li>`).join('')}</ul></div>`:'');
  return `<div class="paper"><h3>${esc(r.title)}</h3><div class="ss">${esc(r.subtitle||'')}</div>${pt}${steps.length?`<h4>Calculation</h4><ul>${steps.map(s=>`<li>${esc(s)}</li>`).join('')}</ul>`:''}</div>`}

// ---------- understanding ----------
const ORD={first:1,'1st':1,second:2,'2nd':2,third:3,'3rd':3,fourth:4,'4th':4,fifth:5,'5th':5,sixth:6,last:-1};
const LOWSCORE=1.8;
const TALK=[[/^(thanks|thank you|thx|ty|thanks a lot|thank you so much)\b/i,'You are welcome. Ask me anything else, or pick a topic to learn about.'],[/\b(bye|goodbye|see you|good night|talk later)\b/i,'Goodbye. Your chats stay saved in this browser.'],[/\b(how are you|how r u|how do you do|how is it going|whats up|what\'s up)\b/i,"I'm fine, thanks. I'm a small program, so no feelings to report, but I'm ready. What shall we work on?"],[/\b(who are you|what is your name|what's your name|whats your name|your name|are you (chat ?gpt|gpt|an? ai|a bot|a robot|human|real|a person)|who (made|built|created|developed) you)\b/i,"I'm Zygon.cognition, a small assistant that runs in your browser. I'm not ChatGPT and I don't write free text. I work out what you are asking, then run an exact calculation or show a short note, and I always show the steps."]];
const modeHint=t=>{const s=t.toLowerCase();if(mode==='hvac'&&/\b(home|house|household|flat|apartment|bhk|ceiling fans?|bulbs?|tube ?lights?|geyser|mcb|wiring|electricity bills?|light bills?)\b/.test(s))return 'home';if(mode==='home'&&/\b(hvac|duct|ducts|cfm|chiller|chilled|tonnage|ahu|refrigeration|psychrometric|dew point|invoice|labour)\b/.test(s))return 'hvac';return null};
window.__zyg={peek:t=>intake(t,hx),score:t=>hx.score(t)};   // read-only helper for testing the intake filter
async function submit(text){
  text=(text||'').trim();if(!text){text=currentSuggestion();if(!text)return}
  if(busy)return;busy=true;$('#q').value='';updateGhost();
  await modeReady;if(!cur.mode){cur.mode=mode;renderList()}
  // intake: whatever was typed or said is cleaned and reduced to the clauses that matter
  const t00=performance.now();const ik=lsGet('zc_intake',1)?intake(text,hx):{prompt:text,changed:false,ignored:0,also:[]};const intakeMs=performance.now()-t00;const q=ik.prompt||text;
  const heard=ik.changed&&q!==text?{heard:q,ignored:ik.ignored,words:ik.words,also:ik.also,intakeMs:+intakeMs.toFixed(2),prompt:q}:{prompt:q,also:[]};
  if(cur.title==='New chat'){cur.title=q.slice(0,38);renderList()}
  const t0=performance.now();waving();
  const t0b=performance.now();const tr=hx.trace(q,3);const traceMs=performance.now()-t0b;let intent=tr.top[0].intent,res,mm=null;if((ik.mode==='filtered'||ik.mode==='meta')&&ik.intent){intent=ik.intent;tr.confident=true}else if(tr.confident&&tr.top[0].score<LOWSCORE&&!/\d/.test(text))tr.confident=false;   // words it barely knows are not a request
  const ctx={files:files[cur.id].all,sheets:files[cur.id].sheets,pdfs:files[cur.id].pdfs,store,current:cur.last?.result};
  try{
    await new Promise(r=>setTimeout(r,650));const t1=performance.now();
    // a few polite phrases get a short stored reply
    {const t=text.trim(),st=t.split(/\s+/).length<=9&&TALK.find(([re])=>re.test(t));
     if(st){const r={kind:'know',title:'Zygon.cognition',short:st[1],points:[],numbers:[],tryit:M().START.slice(0,3),related:[],meta:{...heard,intent:'smalltalk'}};commit(text,r,null,{tin:tokens(text),tout:tokens(st[1]),route:traceMs,tool:0,path:'stored reply'});busy=false;return}}
    // questions like "explain me about lights and fans" get a plain-language note, not a calculation
    await loadKB();
    {const cue=explainCue(text),hasNum=/\d/.test(text)||/\d\s*(?:kw|w|v|a|hp|tons?|tr|cfm|hours?|hrs?|%|percent|m\/s|c|ft|sq|m|mm|kg|rpm|rupees?|rs|people|by|bhk|floors?)\b/i.test(q),metaI=['help_me','out_of_scope','greeting'].includes(intent);
     const tops=findTopics(cue?(q+' '+text):q,mode,3),top=tops[0];
     const done=r=>{r.meta={...heard,...(r.meta||{}),intent:'explain'};const sw=modeHint(text);if(sw&&!r.switch)r.switch=sw;commit(text,r,null,{tin:tokens(text),tout:tokens(outText(r)),route:traceMs,tool:0,path:'explainer'});busy=false};
     if(cue&&/\b(this|that|the result|the answer|the above|how did you|how you got)\b/i.test(text)&&cur.last?.result?.steps?.length){res=await dispatch('show_calc',q,ctx);if(res){res.meta={...(res.meta||{}),...heard,intent:'show_calc'};commit(text,res,null,{tin:tokens(text),tout:tokens(outText(res)),route:traceMs,tool:0,path:'explained the last answer'})}busy=false;return}
     if(top&&top.score>=KMIN&&!hasNum&&(cue||!tr.confident||metaI)){const r=knowledgeResult(top.topic,mode);const more=tops.slice(1).filter(x=>x.score>=0.6*top.score&&x.score>=KMIN).map(x=>({id:x.topic.id,title:x.topic.title}));r.related=[...more,...r.related.filter(x=>!more.some(m=>m.id===x.id))].slice(0,4);done(r);return}
     if(cue&&!hasNum&&(metaI||!tr.confident||intent==='formula_solve')){done(missResult(mode,M().START));return}}
    if(!tr.confident){const opts=tr.top.map(x=>x.intent).filter(i=>!['greeting','help_me','out_of_scope','show_calc','undo','new_chat','edit_value','export_doc','followup_same','store_docs'].includes(i));
      if(opts.length>=2&&(tr.top[0].score>=3||/\d/.test(text))){res={kind:'choose',options:opts,meta:{...heard}}}else{res=fallbackResult(mode,M().START);res.meta={...heard,topic:'(none)',intent:'fallback'}}
      {const sw=modeHint(text);if(sw)res.switch=sw}commit(text,res,null,{tin:tokens(text),tout:tokens(outText(res)),route:traceMs,tool:0,path:'asked you'});busy=false;return}
    res=await dispatch(intent,q,ctx);
    if(res&&!res.switch&&(['help_me','out_of_scope','formula_solve'].includes(intent)||res.kind==='ask'||res.kind==='choose')){const sw=modeHint(text);if(sw)res.switch=sw}
    if(res){res.meta={...(res.meta||{}),intent,traceMs:+traceMs.toFixed(2),...heard};mm={tin:tokens(text),tout:tokens(outText(res)),route:traceMs+intakeMs,tool:performance.now()-t1,path:learnedFor(q,intent)?'taught (your words)':'pre-learned'}}
  }catch(e){res={title:'Something went wrong',subtitle:String(e.message||e),table:null,steps:[]}}
  busy=false;if(res)commit(text,res,null,mm)
}
async function dispatch(intent,text,ctx){
  switch(intent){
  case 'new_chat':newChat();return null;
  case 'undo':if(!undo.length)return {title:'Nothing to undo',subtitle:'',table:null,steps:[]};{const p=undo.pop();return p}
  case 'show_calc':showCalc=true;{const r=cur.last?.result;if(r&&r.steps?.length)return r;return {title:'No calculation to show yet',subtitle:'Ask for a calculation first.',table:null,steps:[]}}
  case 'export_doc':{const r=cur.last?.result;const fmt=/pdf/i.test(text)?'pdf':/ppt|power|slide/i.test(text)?'pptx':/csv/i.test(text)?'csv':'xlsx';if(r?.table){previewFmt=fmt;commit(cur.last.query,r,null,null,true);return null}return {title:'Nothing to export yet',subtitle:'Create a result first, then export it.',table:null,steps:[]}}
  case 'edit_value':{const r=cur.last?.result;if(!r?.table)return {title:'Nothing to edit yet',subtitle:'Create a table first.',table:null,steps:[]};
    // a calculated result is never overwritten: if the table has no editable inputs, the changed number is put into the original question and recalculated
    if(!r.table.editableCols){const q0=cur.last.result?.meta?.prompt||cur.last.query,it=r.meta?.intent,um=text.match(/(-?\d+(?:\.\d+)?)\s*([a-z°%][a-z0-9°%\/²³]*(?:\/[a-z0-9]+)?)/i);
      if(um&&it){const re=new RegExp('(-?\\d+(?:\\.\\d+)?)\\s*'+um[2].replace(/[.*+?^${}()|[\]\\\/]/g,'\\$&'),'i');if(re.test(q0))return dispatch(it,q0.replace(re,um[1]+' '+um[2]),ctx)}
      return {title:'I only change inputs, not results',subtitle:'A calculated value is never overwritten, because the other numbers would then be wrong.',table:null,steps:[],note:'Say the input again with the new value, for example: "duct size for 1500 cfm at 6 m/s".'}}
    const o=text.toLowerCase().match(/\b(first|second|third|fourth|fifth|sixth|last|1st|2nd|3rd|4th|5th)\b/);const nums=[...text.matchAll(/\d+(?:\.\d+)?/g)].map(x=>+x[0]).filter(n=>!(o&&/^\d+(st|nd|rd|th)$/.test(o[0])&&String(n)===o[0].replace(/\D/g,'')));
    const newV=/not\s+\d/.test(text.toLowerCase())?nums[0]:nums[nums.length-1];let ri=o?ORD[o[1]]:1;if(ri===-1)ri=r.table.rows.length;ri-=1;
    if(newV==null||ri<0||ri>=r.table.rows.length)return {title:'Which value?',subtitle:'Say for example "the second quantity should be 12".',table:null,steps:[]};
    const cols=r.table.editableCols||[];const ci=cols.length?cols[0]:r.table.rows[0].length-1;pushUndo(r);r.table.rows[ri][ci]=newV;recalcAny(r);commit(cur.last.query,r,[ri,ci],null,true);return null}
  case 'followup_same':{const r=cur.last?.result;if(!r)return {title:'Nothing to repeat yet',subtitle:'',table:null,steps:[]};const site=(text.match(/(?:for|to|at|:)\s+([A-Za-z0-9' ]+)$/)||[])[1];const c=clone(r);c.subtitle=`${c.subtitle||''}${site?' · '+site:''}`.replace(/^ · /,'');return c}
  case 'formula_solve':{const r=await solve(text);if(r.browse)return {title:'Which formula do you mean?',subtitle:'I can work out about 200 physics, maths and engineering formulas. Name one and give the numbers.',table:{columns:['Subject','You could type'],rows:[['Electricity','Ohm law voltage with I 2 A and R 115 ohm'],['Electricity','power with V 230 V and I 5 A'],['Motion and energy','kinetic energy with m 5 kg and v 10 m/s'],['Geometry','area of a circle with radius 5 m'],['Money','EMI for 5 lakh at 9% for 5 years'],['Maths','percentage change from 80 to 100'],['Heat','conduction with k 0.5, A 2, dT 40, L 0.1']]},note:'Every formula shows its source link under the answer.',steps:[]};if(r.ask)return {kind:'ask',title:r.title,hint:r.hint,questions:r.questions};return r}
  case 'greeting':case 'help_me':if(mode==='home')return {kind:'help',title:'Zygon.cognition',subtitle:M().subtitle};return run(intent,text,ctx);
  case 'out_of_scope':if(mode==='home')return {title:'Outside my job',subtitle:'I only do home electrical estimates and formulas.',table:null,steps:[],note:'Try a bill estimate, lights for a room, MCB and wire sizes, or a physics formula.'};return run(intent,text,ctx);
  default:{const r=await M().run(intent,text,ctx);return r||{title:'Not wired yet',subtitle:M().LABEL[intent]||intent,table:null,steps:[]}}}
}
async function pick(text,intent){const tl=performance.now();const info=hx.teach(text,intent);const learnMs=performance.now()-tl;const ctx={files:files[cur.id].all,sheets:files[cur.id].sheets,pdfs:files[cur.id].pdfs,store};busy=true;waving();await new Promise(r=>setTimeout(r,450));
  const t1=performance.now();let res=await dispatch(intent,text,ctx);const toolMs=performance.now()-t1;busy=false;if(res){res.meta={...(res.meta||{}),intent};res.note=((res.note?res.note+'  ':'')+`Learned: stored in ${info.space}.`);commit(text,res,null,{tin:tokens(text),tout:tokens(outText(res)),route:0,tool:toolMs,learn:learnMs,path:'taught just now'},true)}renderList();renderSide()}


// ---------- speak instead of typing ----------
// Speech recognition is the browser's. Zygon improves on it in three ways: (1) it asks the recogniser for several alternatives for every phrase and keeps the one the
// Zygon brain understands best (the jargon words win over look-alike everyday words); (2) it tells the recogniser the engineering words to expect, where the browser supports that;
// (3) it prefers on-device recognition (works offline, audio stays here) and only falls back to the browser's online service after asking you.
(()=>{const SR=window.SpeechRecognition||window.webkitSpeechRecognition;const mic=$('#mic');if(!SR||!mic)return;mic.hidden=false;let rec=null,on=false,text='';
  const JARGON=['kilowatt','cfm','duct','tons of refrigeration','chilled water','breaker','MCB','EER','COP','BTU','psychrometric','dew point','humidity','three phase','voltage','amps','horsepower','lakh','rupees per unit','BHK','tube light','ceiling fan','wire size','Reynolds number','EMI','invoice','cooling load','airflow','delta T','pipe size','enthalpy'];
  const hear=t=>{const v=Object.values(hx.score(normalise(t)));return v.length?Math.max(...v):0};
  const stop=()=>{on=false;mic.classList.remove('on');$('#q').placeholder='';try{rec&&rec.stop()}catch{}};
  const begin=(local)=>{
    rec=new SR();rec.continuous=true;rec.interimResults=true;rec.maxAlternatives=5;rec.lang=lsGet('zc_lang','en-IN');text='';
    try{if(local&&'processLocally' in rec)rec.processLocally=true}catch{}
    try{if(window.SpeechRecognitionPhrase&&'phrases' in rec)rec.phrases=JARGON.map(w=>new window.SpeechRecognitionPhrase(w,6))}catch{}
    rec.onresult=e=>{let s='';for(let i=0;i<e.results.length;i++){const r=e.results[i];let pick=r[0].transcript,sc=-1;
        if(r.isFinal)for(let k=0;k<r.length;k++){const v=hear(r[k].transcript)+3*(r[k].confidence||0);if(v>sc){sc=v;pick=r[k].transcript}}   // keep the alternative the Zygon brain understands best
        s+=pick+' '}text=s.trim();$('#q').value=text;updateGhost()};
    rec.onend=()=>{const had=on;stop();if(had&&text)submit(text)};   // stopping sends what was heard
    rec.onerror=ev=>{const was=on;stop();
      if(local&&(ev.error==='language-not-supported'||ev.error==='service-not-allowed')&&was){startCloud();return}
      $('#q').placeholder=ev.error==='not-allowed'?'Microphone is blocked for this page':ev.error==='no-speech'?'I did not hear anything':''};
    try{rec.start();on=true;mic.classList.add('on');$('#q').placeholder=local?'Listening (on this device)…':'Listening…'}catch{stop()}};
  const startCloud=()=>{if(!lsGet('zc_mic_ok',false)){if(!confirm("On-device speech is not available in this browser, so speaking would use your browser's online speech service. In Chrome that can send your audio to the internet, unlike the rest of Zygon.\n\nContinue?"))return;lsSet('zc_mic_ok',true)}begin(false)};
  mic.onclick=async()=>{
    if(on){stop();return}
    let mode='cloud';
    try{if('processLocally' in SR.prototype&&SR.available){const av=await SR.available({langs:[lsGet('zc_lang','en-IN')],processLocally:true});
      if(av==='available'||av==='readily')mode='local';
      else if((av==='downloadable'||av==='after-download')&&SR.install&&confirm('Install the offline speech pack for this language? Your browser downloads it once; after that speaking works offline and audio stays on this device.')){try{await SR.install({langs:[lsGet('zc_lang','en-IN')],processLocally:true});mode='local'}catch{}}}}catch{}
    mode==='local'?begin(true):startCloud()};
  window.__voice={feed:t=>{$('#q').value=t;submit(t)}}})();

// ---------- files ----------
$('#attach').onclick=()=>$('#file').click();
$('#file').onchange=async e=>{const fs=[...e.target.files];e.target.value='';const bag=files[cur.id];let pasted=null;
  for(const file of fs){if(/\.(txt|md|text)$/i.test(file.name)){pasted=await file.text();continue}bag.all.push(file);try{if(/\.(xlsx|xlsm)$/i.test(file.name)){await loadScript('vendor/exceljs.min.js');const wb=new ExcelJS.Workbook();await wb.xlsx.load(await file.arrayBuffer());wb.eachSheet(ws=>{const rows=[];ws.eachRow({includeEmpty:false},row=>rows.push(row.values.slice(1).map(v=>v&&typeof v==='object'?(v.result??v.text??''):v)));const hi=rows.findIndex(r=>r.filter(x=>x!==''&&x!=null).length>=2);if(hi>=0)bag.sheets.push({name:`${file.name} · ${ws.name}`,header:rows[hi],rows:rows.slice(hi+1)})})}
    else if(/\.csv$/i.test(file.name)){const rows=(await file.text()).split(/\r?\n/).filter(Boolean).map(l=>l.split(',').map(c=>{c=c.replace(/^"|"$/g,'').trim();return c!==''&&!isNaN(c)?+c:c}));bag.sheets.push({name:file.name,header:rows[0],rows:rows.slice(1)})}
    else if(/\.pdf$/i.test(file.name)){const pdfjs=await import('../vendor/pdf.min.mjs');pdfjs.GlobalWorkerOptions.workerSrc=new URL('../vendor/pdf.worker.min.mjs',import.meta.url).href;const doc=await pdfjs.getDocument({data:await file.arrayBuffer()}).promise;const pages=[];for(let i=1;i<=doc.numPages;i++){const c=await (await doc.getPage(i)).getTextContent();let s='';for(const it of c.items){s+=it.str+(it.hasEOL?'\n':' ')}pages.push(s)}bag.pdfs.push({name:file.name,pages})}}catch(err){console.warn('read fail',file.name,err)}}
  renderChips();save();if(pasted)submit(pasted)};

// ---------- export ----------
function exportRows(r){const t=r.table;const rows=t.rows.map(x=>x.slice());if(r.meta?.type==='invoice'){const T=totals(r);const e=new Array(t.columns.length).fill('');rows.push(Object.assign(e.slice(),{[t.columns.length-2]:'Subtotal',[t.columns.length-1]:T.sub}),Object.assign(e.slice(),{[t.columns.length-2]:`Tax ${t.totals.tax}%`,[t.columns.length-1]:T.tax}),Object.assign(e.slice(),{[t.columns.length-2]:'Total',[t.columns.length-1]:T.total}))}return rows}
const dl=(blob,name)=>{const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),4000)};
async function exportAs(kind,r){const base=(r.title||'result').replace(/[^\w]+/g,'_').slice(0,40);const cols=r.table.columns,rows=exportRows(r);
  if(kind==='csv'){dl(new Blob([[cols,...rows].map(x=>x.map(v=>`"${String(v).replace(/"/g,'""')}"`).join(',')).join('\n')],{type:'text/csv'}),base+'.csv')}
  else if(kind==='xlsx'){await loadScript('vendor/exceljs.min.js');const wb=new ExcelJS.Workbook(),ws=wb.addWorksheet('Result');ws.addRow(cols).font={bold:true};rows.forEach(x=>ws.addRow(x));ws.columns.forEach(c=>c.width=22);if(r.steps?.length){const s=wb.addWorksheet('Calculation');r.steps.forEach(x=>s.addRow([x]));s.getColumn(1).width=100}dl(new Blob([await wb.xlsx.writeBuffer()],{type:'application/octet-stream'}),base+'.xlsx')}
  else if(kind==='pptx'){await loadScript('vendor/pptxgen.bundle.js');const p=new PptxGenJS(),s=p.addSlide();s.addText(r.title,{x:.5,y:.3,w:9,h:.6,fontSize:26,bold:true,color:'3B3B55'});s.addText(r.subtitle||'',{x:.5,y:.9,w:9,h:.4,fontSize:14,color:'777777'});s.addTable([cols.map(c=>({text:c,options:{bold:true,fill:{color:'7C4DFF'},color:'FFFFFF'}})),...rows.map(x=>x.map(v=>({text:typeof v==='number'?f(v):String(v)})))],{x:.5,y:1.5,w:9,fontSize:13,border:{type:'solid',color:'CCCCCC',pt:.5}});if(r.steps?.length){const s2=p.addSlide();s2.addText('Calculation',{x:.5,y:.3,fontSize:24,bold:true});s2.addText(r.steps.map(t=>({text:t,options:{bullet:true,breakLine:true}})),{x:.5,y:1.1,w:9,h:4,fontSize:14})}await p.writeFile({fileName:base+'.pptx'})}
  else{await loadScript('vendor/pdf-lib.min.js');const {PDFDocument,StandardFonts,rgb}=PDFLib;const doc=await PDFDocument.create();const font=await doc.embedFont(StandardFonts.Helvetica),bold=await doc.embedFont(StandardFonts.HelveticaBold);
    const W=595,H=842,M=44;let page=doc.addPage([W,H]),y=H-M;const clip=(s,w,fnt,sz)=>{s=String(s);while(s.length>1&&fnt.widthOfTextAtSize(s,sz)>w)s=s.slice(0,-1);return s};
    page.drawText(clip(r.title,W-2*M,bold,20),{x:M,y,size:20,font:bold,color:rgb(.2,.2,.3)});y-=22;if(r.subtitle){page.drawText(clip(r.subtitle,W-2*M,font,10.5),{x:M,y,size:10.5,font,color:rgb(.45,.45,.5)});y-=24}
    const cw=(W-2*M)/cols.length;const rowDraw=(cells,fnt,fill)=>{if(y<M+30){page=doc.addPage([W,H]);y=H-M}if(fill)page.drawRectangle({x:M,y:y-5,width:W-2*M,height:18,color:rgb(.93,.92,.98)});cells.forEach((c,i)=>page.drawText(clip(typeof c==='number'?f(c):c,cw-8,fnt,10),{x:M+i*cw+4,y,size:10,font:fnt,color:rgb(.1,.1,.15)}));y-=20};
    rowDraw(cols,bold,true);rows.forEach(x=>rowDraw(x,font,false));
    if(r.steps?.length){y-=14;if(y<M+60){page=doc.addPage([W,H]);y=H-M}page.drawText('Calculation',{x:M,y,size:12,font:bold});y-=18;r.steps.forEach(s=>{if(y<M){page=doc.addPage([W,H]);y=H-M}page.drawText(clip('- '+s,W-2*M,font,9.5),{x:M,y,size:9.5,font,color:rgb(.3,.3,.35)});y-=14})}
    dl(new Blob([await doc.save()],{type:'application/pdf'}),base+'.pdf')}}

// ---------- composer: ghost suggestions ----------
const q=$('#q'),ghost=$('#ghost');
function currentSuggestion(){if(!cur?.mode)return '';const v=q.value;if(!v)return M().START[sugIdx%M().START.length];const lv=v.toLowerCase();const hit=suggestList.find(s=>s.toLowerCase().startsWith(lv)&&s.length>v.length);return hit||''}
function updateGhost(){const v=q.value,s=currentSuggestion();ghost.innerHTML=s?(v?`<span class="t">${esc(v)}</span>${esc(s.slice(v.length))}`:esc(s)):''}
q.addEventListener('input',updateGhost);
let typing=false;
function typeIn(text){typing=true;q.value='';let i=0;const step=Math.max(1,Math.ceil(text.length/26));const id=setInterval(()=>{i=Math.min(text.length,i+step);q.value=text.slice(0,i);updateGhost();if(i>=text.length){clearInterval(id);typing=false;q.focus()}},16)}
function enter(){if(typing)return;if(!q.value.trim()){const s=currentSuggestion();if(s)typeIn(s);return}submit(q.value)}   // empty box: first Enter types the suggestion in, second Enter sends
q.addEventListener('keydown',e=>{if(e.key==='Tab'){const s=currentSuggestion();if(s&&s!==q.value){e.preventDefault();typeIn(s)}}else if(e.key==='Enter'){e.preventDefault();enter()}});
$('#send').onclick=enter;
setInterval(()=>{if(!q.value){sugIdx++;updateGhost()}},5000);


// ---------- context popover (right of the send button) and settings ----------
const docTokens=()=>{const b=files[cur.id]||{sheets:[],pdfs:[]};let n=0;for(const s of b.sheets)n+=JSON.stringify([s.header,...s.rows]).length;for(const p of b.pdfs)n+=p.pages.join(' ').length;return Math.ceil(n/4)};
const fmtN=n=>n>=1e6?(n/1e6).toFixed(n>=1e7?0:1)+'M':n>=1e3?(n/1e3).toFixed(n>=1e4?0:1)+'K':String(n);
function buildCtx(){
  const items=cur.items||[],a=items[cur.active],m=a?.m,dt=docTokens(),nf=(files[cur.id]?.all||[]).length;
  const lo=Math.log10(1e3),hi=Math.log10(1e8),pos=v=>Math.max(0,Math.min(100,(Math.log10(Math.max(v,1))-lo)/(hi-lo)*100));
  const kv=(k,v)=>`<div class="kv"><span>${k}</span><b>${v}</b></div>`;
  const ticks=[[3e4,'30K'],[2e6,'2M'],[1.8e7,'18M']].map(([v,t],i)=>`<div class="tk ${i%2?'big':''}" style="left:${pos(v)}%"><span>${t}</span></div>`).join('');
  $('#ctxpop').innerHTML=`<h4>Context</h4><div class="scale"><div class="rail2"><i style="width:${pos(dt)}%"></i></div>${ticks}</div>
   ${kv('Held now',fmtN(dt)+' tokens · '+nf+' file'+(nf===1?'':'s'))}${kv('Replies in this chat',items.length)}
   <h4>Latest reply</h4>${m?kv('Tokens in',m.tin)+kv('Tokens out',m.tout)+kv('Words voted',hx.words(a.query).length)+kv('Voting',ms(m.route))+kv('Building reply',ms(m.tool))+(m.learn!=null?kv('Learning',ms(m.learn)):'')+kv('Total',ms(m.route+m.tool+(m.learn||0))):kv('–','–')}
   <h4>Reasoning</h4>${kv('Level','Standard')}${kv('Method','Lookup + exact calculators')}`}
$('#ctxbtn').onclick=e=>{e.stopPropagation();const p=$('#ctxpop');p.hidden=!p.hidden;$('#ctxbtn').classList.toggle('on',!p.hidden);if(!p.hidden)buildCtx()};
document.addEventListener('click',e=>{const p=$('#ctxpop');if(!p.hidden&&!p.contains(e.target)){p.hidden=true;$('#ctxbtn').classList.remove('on')}});

async function openSettings(){
  $('#settings').hidden=false;
  const used=(localStorage.getItem('zc_chats')||'').length/1e6;
  $('#setbody').innerHTML=`${modeSection()}
   <h4>Benchmarks</h4><table><thead><tr><th>Target</th><th>Standard</th></tr></thead><tbody>
    <tr><td>Transparency</td><td>100%, no black box</td></tr>
    <tr><td>Context window</td><td>Unlimited</td></tr>
    <tr><td>Mathematical accuracy</td><td>100%</td></tr>
    <tr><td>Hardware</td><td>No GPU</td></tr></tbody></table>
   <h4>Memory</h4><div class="mem">Chat memory: ${used<0.1?Math.round(used*1000)+' KB':used.toFixed(2)+' MB'} of ${MEM_MB()} MB used</div>
   <h4>Sharing</h4><div class="mem"><label class="sw"><input type="checkbox" id="shareon" ${Share.enabled()?'checked':''}> Share my chats with the developers (test version)</label><div class="sub2">Your name: <input id="sharename" class="nameinput small" maxlength="60" value="${esc(Share.name())}" placeholder="Your name"> · ID <code>${esc(Share.shortId())}</code>. Attached files are never shared.</div><button class="btn" id="shareforget">Ask to delete what I shared</button> <span id="sharemsg" class="sub2"></span></div>`;
  $('#setbody').querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
  $('#shareon').onchange=e=>Share.setConsent(e.target.checked);
  $('#sharename').onchange=e=>Share.setName(e.target.value);
  $('#shareforget').onclick=async()=>{$('#sharemsg').textContent='Deleting…';const ok=await Share.forget();$('#sharemsg').textContent=ok?'Request sent. The developers will delete it.':'Could not reach the server; nothing was changed.'}}
$('#settingsbtn').onclick=openSettings;$('#setclose').onclick=()=>$('#settings').hidden=true;
$('#settings').onclick=e=>{if(e.target.id==='settings')$('#settings').hidden=true};

// ---------- chrome ----------
$('#menu').onclick=()=>{const a=$('#app');if(innerWidth<=1000)a.classList.toggle('open');else{a.classList.toggle('collapsed');lsSet('zc_collapsed',a.classList.contains('collapsed'))}};
$('#scrim').onclick=()=>$('#app').classList.remove('open');
$('#theme').onclick=()=>{const d=document.documentElement;d.dataset.theme=d.dataset.theme==='dark'?'light':'dark';try{localStorage.setItem('zc_theme',d.dataset.theme)}catch{}};
$('#newchat').onclick=newChat;
if(lsGet('zc_collapsed',false)&&innerWidth>1000)$('#app').classList.add('collapsed');

// ---------- boot ----------
async function loadMode(){const m=M(),[model,sug]=await Promise.all([fetch(m.model).then(r=>r.json()),fetch(m.suggest).then(r=>r.json())]);hx=new Hexonet(model,m.key);suggestList=sug}
async function setMode(m){if(!MODES[m]||m===mode)return;mode=m;try{localStorage.setItem('zc_mode',JSON.stringify(m))}catch{}await loadMode();$('#settings').hidden=true;newChat();cur.mode=m;empty();save()}
function modeSection(){return `<h4>Mode</h4><div class="seg">${Object.entries(MODES).map(([k,v])=>`<button class="btn ${k===mode?'on':''}" data-mode="${k}">${esc(v.name)}</button>`).join('')}</div>`}
(async()=>{await loadMode();
  if(chats.length){chats.forEach(c=>files[c.id]={all:[],sheets:[],pdfs:[]});open(chats[0].id)}else newChat();q.focus();if(Share.needsConsent())Share.askConsent();window.__zc={get hx(){return hx},get cur(){return cur},submit,setMode,get mode(){return mode}}})();