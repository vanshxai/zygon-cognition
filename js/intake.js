// Intake filter: whatever the user says or types (5 words or 5,000, typed or dictated) is cleaned, split into clauses,
// each clause is scored by the same voting brain, and only the clauses that matter become the prompt. Nothing is generated.
import {RULES} from './physics.js';
const W1={zero:0,one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9,ten:10,eleven:11,twelve:12,thirteen:13,fourteen:14,fifteen:15,sixteen:16,seventeen:17,eighteen:18,nineteen:19};
const TENS={twenty:20,thirty:30,forty:40,fifty:50,sixty:60,seventy:70,eighty:80,ninety:90};
const SCALE={thousand:1e3,lakh:1e5,lakhs:1e5,lac:1e5,lacs:1e5,crore:1e7,crores:1e7,million:1e6};
const isNW=w=>w in W1||w in TENS||w==='hundred'||w in SCALE;
// "fifteen hundred" -> 1500, "thirty two" -> 32, "two point five" -> 2.5, "five lakh" -> 500000
export function numberWords(t){
  const tk=t.split(/(\s+)/);const out=[];let i=0;
  while(i<tk.length){
    const w=(tk[i]||'').toLowerCase().replace(/[,.;!?]+$/,'');const tail=(tk[i]||'').match(/[,.;!?]+$/)?.[0]||'';
    const startsRun=isNW(w)||((w==='a'||w==='an')&&['hundred','thousand','lakh','crore','million'].includes((tk[i+2]||'').toLowerCase()));
    if(!startsRun){out.push(tk[i]);i++;continue}
    let total=0,cur=0,j=i,last=i,seen=false,dec='',prevHundred=false;
    while(j<tk.length){
      const x=(tk[j]||'').toLowerCase().replace(/[,.;!?]+$/,'');
      if(/^\s*$/.test(tk[j]||'')){j++;continue}
      if(x==='a'||x==='an'){if(!seen&&['hundred','thousand','lakh','crore','million'].includes((tk[j+2]||'').toLowerCase())){cur=1;j++;continue}break}
      if(x in W1){if(cur>=1&&cur<=9&&W1[x]>=10){cur=cur*100+W1[x]}else cur+=W1[x];seen=true;last=j}
      else if(x in TENS){if(cur>=1&&cur<=9){cur=cur*100+TENS[x]}else cur+=TENS[x];seen=true;last=j}
      else if(x==='hundred'){cur=(cur||1)*100;seen=true;last=j}
      else if(x in SCALE){total+=(cur||1)*SCALE[x];cur=0;seen=true;last=j}
      else if(x==='and'&&seen&&isNW((tk[j+2]||'').toLowerCase().replace(/[,.;!?]+$/,''))){}
      else if(x==='point'&&seen){let k=j+2,ds='';while(k<tk.length){const d=(tk[k]||'').toLowerCase().replace(/[,.;!?]+$/,'');if(d in W1&&W1[d]<10){ds+=W1[d];last=k;k+=2}else break}if(ds){dec='.'+ds;j=k;break}else break}
      else break;
      if(/[,.;!?]$/.test(tk[j]||''))break;
      j++}
    if(!seen){out.push(tk[i]);i++;continue}
    {const firstW=(tk[i]||'').toLowerCase().replace(/[,.;!?]+$/,'');const single=(last===i);const nextW=(tk[last+2]||'').toLowerCase().replace(/[,.;!?]+$/,'');
     if(single&&(firstW==='one'||!/^(?:kw|kilowatts?|kilo|tons?|tonnes?|hours?|hrs?|people|persons?|percent|per|volts?|v|amps?|amperes?|cfm|metres?|meters?|ft|feet|foot|floors?|storeys?|bhk|bedrooms?|fans?|lights?|bulbs?|tubes?|rupees?|rs|units?|phase|hp|horsepower|degrees?|celsius|mm|cm|kg|litres?|workers?|men|days?|months?|years?|lakh|crore|k)$/.test(nextW))){out.push(tk[i]);i++;continue}}
    out.push(String(total+cur)+dec+((tk[last]||'').match(/[,.;!?]+$/)?.[0]||''));i=last+1}
  return out.join('')}
const UNITS=[[/\bc\.?\s*f\.?\s*m\b/gi,'cfm'],[/\be\.?\s*e\.?\s*r\b/gi,'EER'],[/\bb\.?\s*t\.?\s*u\b/gi,'BTU'],[/\bm\.?\s*c\.?\s*b\b/gi,'MCB'],[/\bk\.?\s*v\.?\s*a\b/gi,'kVA'],[/\bk\.?\s*w\b/gi,'kW'],[/\bkilo ?what[s]?\b/gi,'kW'],[/\btonnes?\b/gi,'tons'],[/\bkilo ?watts?\b/gi,'kW'],[/\bmega ?watts?\b/gi,'MW'],[/\bhorse ?power\b/gi,'hp'],[/\bcubic feet per minute\b/gi,'cfm'],[/\b(?:metres?|meters?) per second\b/gi,'m/s'],[/\bfeet per minute\b/gi,'fpm'],
 [/\bsquare (?:feet|foot)\b/gi,'sq ft'],[/\bsquare (?:metres?|meters?)\b/gi,'sq m'],[/\bper ?cent\b/gi,'percent'],[/(\d)\s*(?:volts?)\b/gi,'$1 V'],[/(\d)\s*(?:amps?|amperes?)\b/gi,'$1 A'],
 [/\btons? of refrigeration\b/gi,'tons'],[/(\d(?:\.\d+)?)\s*(?:degrees?|deg)\s*(?:celsius|centigrade|c)\b/gi,'$1 C'],[/\bcelsius\b/gi,'C'],[/\brupees? per unit\b/gi,'rupees a unit'],[/\b(\d+)\s*hrs?\b/gi,'$1 hours']];
const HINGLISH=[[/\bkitn[aei]\b/gi,'how many'],[/\bghant[ea]s?\b/gi,'hours'],[/\brupa(?:ye|y|iya)\b/gi,'rupees'],[/\bbijli\b/gi,'electricity'],[/\bkharch[a]?\b/gi,'cost'],[/\bkamra\b/gi,'room'],[/\bpankh[ae]\b/gi,'fan'],[/\bmer[ai]\b/gi,'my'],
 [/\b(?:batao|bataiye|batana|bolo|bol do|batado|dikhao|nikalo|nikaliye)\b/gi,' '],[/\b(?:ka|ke|ki|ko|mein|pe|hai|hain|hota|hote|hoti|lagega|lagegi|lagenge|chahiye|chahie|aur|yaar|bhai|kya|wala|wali|toh)\b/gi,' '],[/\bke liye\b/gi,' ']];
const HINDI_MARK=/\b(?:kitn[aei]|batao|bataiye|ghant[ea]s?|rupa(?:ye|y|iya)|bijli|kharch[a]?|mer[ai]|hai|hain|lagega|lagegi|chahiye|bhai|yaar|kya|ka|ke|ki|mein|pe)\b/gi;
const FILL=/\b(?:um+|uh+|er+|hmm+|hey there|hello there|hi there|hello|hi|hey|please|kindly|thanks? a lot|thank you|thanks|sorry|anyway|basically|actually|literally|you know|i mean|sort of|kind of|by the way|honestly|so yeah|yeah|ok(?:ay)?|good morning|good evening|good afternoon|just wanted to|i was just wondering|i was wondering if you could|i was wondering|i would like to know|i want to know|i need to know|i would like|i am trying to|i am trying to figure out|i need you to|can you please|could you please|if possible|as soon as possible)\b/gi;
const CUE=/\b(?:how (?:many|much|big|large|long|do|should)|what(?:'s| is| are| size| will)?|which|calculate|compute|find|size|sizing|estimate|convert|work out|determine|figure out|should i|give me|need|require|want|plan|compare)\b|\?/i;
const META=new Set(['greeting','help_me','out_of_scope','new_chat','undo','show_calc','followup_same','edit_value','export_doc','store_docs']);
const UNITNUM=/\d\s*(?:kw|w|v|a|hp|tons?|tr|cfm|hours?|%|percent|m\/s|c|ft|sq ft|sq m|m|mm|kg|n|pa|rpm|rupees?|rs|people|persons?|by\s*\d+)\b/i;
const GENERIC_FACT=/\d\s*(?:by|x)\s*\d|\bpeople|persons?|occupants|sq|square|m2|\bbhk\b|floors?|storeys?|\d\s*(?:kw|w|v|a|hp|tons?|tr|cfm|hours?|hrs?|%|percent|m\/s|c|ft|m|mm|kg|n|pa|rpm|rupees?|rs)\b|\b(?:at|rate|tariff)\s*(?:rs\.?|₹)?\s*\d/i;
export function normalise(t){
  let s=t.replace(/ /g,' ').replace(/[‘’]/g,"'").replace(/[“”]/g,'"');
  s=numberWords(s);for(const [re,to] of UNITS)s=s.replace(re,to);if((s.match(HINDI_MARK)||[]).length>=2)for(const [re,to] of HINGLISH)s=s.replace(re,to);
  return s.replace(/\s+/g,' ').trim()}
const clean=c=>c.replace(/^\s*(?:also|and|then|plus)[,\s]+/i,'').replace(FILL,' ').replace(/\s+([,.;?!])/g,'$1').replace(/^[\s,.;:!-]+|[\s,;:-]+$/g,'').replace(/\s+/g,' ').trim();
function clauses(s){
  const parts=s.split(/(?<=[.!?;])\s+|\n+|\s+(?:and then|but|so|because|also|plus|and also)\s+/i).map(x=>x.trim()).filter(Boolean);
  const out=[];for(const p of parts){const n=p.split(/\s+/).length;if(n>36){p.split(/,\s+|\s+(?:and)\s+/).forEach(q=>q.trim()&&out.push(q.trim()))}else out.push(p)}
  return out}
const nums=c=>(c.match(/\d+(?:\.\d+)?/g)||[]).length;
// raw: what the user typed or said. hx: the router. Returns the focused prompt and what was set aside.
export function intake(raw,hx){
  const original=String(raw||'').trim();const words=original.split(/\s+/).filter(Boolean).length;
  const norm=normalise(original);const sentences=norm.split(/[.!?;\n]+/).filter(x=>x.trim()).length;
  const sure=t=>{const e=Object.entries(hx.score(t)).sort((a,b)=>b[1]-a[1]);return e.length&&e[0][1]>=8&&e[0][1]-(e[1]?.[1]??0)>=4};
  if(words>14&&words<=80&&!/\d/.test(norm)){const e=Object.entries(hx.score(clean(norm))).sort((a,b)=>b[1]-a[1]);if(e[0]&&e[0][0]==='help_me'&&e[0][1]>=10&&e[0][1]-(e[1]?.[1]??0)>=4)return {prompt:'What can you do?',changed:true,ignored:0,words,also:[],mode:'meta',intent:'help_me'}}
  if(words<=8||(sentences<=1&&words<=30&&sure(clean(norm)))){const c=clean(norm)||norm;return {prompt:c||original,changed:c!==original,ignored:0,words,also:[],mode:'short'}}
  const cl=clauses(norm);let rows=[];
  const scoreRow=(text,extra)=>{const sc=hx.score(text);const e=Object.entries(sc).sort((a,b)=>b[1]-a[1]);const top=e[0]||['',0];return {text,intent:top[0],score:top[1],margin:top[1]-(e[1]?.[1]??0),cue:CUE.test(text)?1:0,...extra}};
  cl.forEach((c,i)=>{const cc=clean(c);if(!cc)return;const r=scoreRow(cc,{i,raw:c});rows.push(r);
    // a long run of words with no full stops (dictated speech) is also read in overlapping windows, so the request inside it can be found
    const w=cc.split(/\s+/);if(w.length>8&&!(r.score>=8&&r.margin>=4)){for(const size of [5,7,9,12])for(let st=0;st+size<=w.length+2;st+=3){const ws=st,we=Math.min(w.length,st+size);if(we-ws<4)continue;const win=scoreRow(w.slice(ws,we).join(' '),{i:i+(ws*20+size)/100000,raw:c,parent:i,ws,we,words:w});rows.push(win)}}});
  // a question like "which breaker should I use?" only makes sense with its facts. The clauses that carry numbers and units are grouped by the job they point to,
  // and the question is judged together with each group; the best match decides what the question is about
  const base=rows.filter(r=>r.parent==null);
  const grp={};{const seenF=new Set();for(const f of base){if(META.has(f.intent)||f.score<2||!UNITNUM.test(f.text))continue;const k=f.text.toLowerCase();if(seenF.has(k))continue;seenF.add(k);(grp[f.intent]??=[]).push(f)}}
  const groups=Object.entries(grp).map(([i,l])=>[i,l.sort((a,b)=>b.score-a.score).slice(0,6)]).sort((a,b)=>b[1].reduce((s,x)=>s+x.score,0)-a[1].reduce((s,x)=>s+x.score,0)).slice(0,5);
  for(const r of base){if(!r.cue||(r.score>=6&&r.margin>=2)||(UNITNUM.test(r.text)&&r.score>=2&&r.margin>=1))continue;let bestI=null,bestS=r.score;
    const near=base.filter(f=>f.i!==r.i&&UNITNUM.test(f.text)).sort((x,y)=>Math.abs(x.i-r.i)-Math.abs(y.i-r.i)).slice(0,4);
    // first the groups whose own job agrees with the combined reading; only if none helps, the nearest number-and-unit clauses whatever job they point to
    for(const [gi,l] of groups){const sc=hx.score(clean(r.text+' '+l.map(x=>x.text).join(' ')));const e=Object.entries(sc).sort((x,y)=>y[1]-x[1]);const [ti,tv]=e[0]||['',0];if(ti===gi&&tv>bestS&&!META.has(ti)){bestS=tv;bestI=ti}}
    if(!bestI&&near.length){const sc=hx.score(clean(r.text+' '+near.map(x=>x.text).join(' ')));const e=Object.entries(sc).sort((x,y)=>y[1]-x[1]);const [ti,tv]=e[0]||['',0];if(tv>bestS&&!META.has(ti)){bestS=tv;bestI=ti}}
    if(bestI){r.intent=bestI;r.score=bestS;r.margin=Math.max(r.margin,1)}}
  // a clause votes only if it looks like part of a request (a question or an instruction, or a number with a unit) and the brain is sure about it; a repeated sentence counts once
  // overlapping windows of one run of words would vote many times for the same stretch, so only the best window of each run may vote
  const qual=r=>r.score*(1+0.5*r.cue)*(1+(UNITNUM.test(r.text)?0.5:0));const bestWin={};for(const r of rows){if(r.parent!=null&&(!bestWin[r.parent]||qual(r)>qual(bestWin[r.parent])))bestWin[r.parent]=r}
  const seen=new Set();const votes=rows.filter(r=>{if(r.parent!=null&&bestWin[r.parent]!==r)return false;if(META.has(r.intent)||r.score<2||r.margin<0.5)return false;if(!(r.cue||UNITNUM.test(r.text)))return false;const k=r.text.toLowerCase();if(seen.has(k))return false;seen.add(k);return true});
  const span=r=>r.parent==null?r.text:r.words.slice(r.ws,r.we).join(' ');
  if(!votes.length||(!votes.some(r=>r.cue)&&votes.filter(r=>UNITNUM.test(r.text)).length<2&&!votes.some(r=>r.score>=6))){
    // no task found: maybe the person simply asked what Zygon can do, or greeted it
    const wc=r=>span(r).split(/\s+/).length;const hm=rows.filter(r=>['help_me'].includes(r.intent)&&r.score>=3&&r.margin>=0.5);const top=Math.max(0,...hm.map(r=>r.score));const meta=hm.filter(r=>r.score>=Math.max(3,0.35*top)).sort((a,b)=>wc(a)-wc(b)||b.score-a.score)[0];   // the shortest stretch that still says it
    if(meta)return {prompt:'What can you do?',changed:true,ignored:rows.length-1,words,also:[],mode:'meta',intent:meta.intent};
    // nothing was sure enough: still hand the router the stretch that looks most like a request instead of the whole ramble
    const weak=rows.filter(r=>(r.cue||UNITNUM.test(r.text))&&!META.has(r.intent)).sort((a,b)=>qual(b)-qual(a))[0];
    if(weak)return {prompt:clean(span(weak)),changed:true,ignored:rows.length-1,words,also:[],mode:'weak'};
    return {prompt:clean(norm).slice(0,600),changed:true,ignored:0,words,also:[],mode:'nothing-found'}}
  const tot={};for(const r of votes)tot[r.intent]=(tot[r.intent]||0)+r.score*(r.cue?1.5:1);
  const ranked=Object.entries(tot).sort((a,b)=>b[1]-a[1]);const W=ranked[0][0];
  const mine=votes.filter(r=>r.intent===W);
  const best=mine.reduce((a,b)=>(b.cue-a.cue)||(b.score-a.score)>0?b:a,mine[0]);   // the clause that asks, preferring one with a question cue
  const rules=RULES[W]||[];const chosen=new Set([best.i]);
  if(rules.length){
    for(const r of rules){const re=new RegExp(r.re.source,'i');if(re.test(best.text.toLowerCase()))continue;let hit=null;for(const c of base){if(c.i===best.i)continue;if(re.test(c.text.toLowerCase())&&(!hit||Math.abs(c.i-best.i)<Math.abs(hit.i-best.i)))hit=c}if(hit)chosen.add(hit.i)}
    for(const c of mine)chosen.add(c.i)}
  else for(const c of mine)chosen.add(c.i);
  if(/^(cooling_load|home_)/.test(W)){for(const c of base){if(/\d\s*(?:by|x)\s*\d|\bbhk\b|floors?|storeys?|\bpeople\b|persons?|occupants|sq|square/i.test(c.text)&&Math.abs(c.i-best.i)<=6)chosen.add(c.i)}}
  const keep=[...chosen].sort((a,b)=>a-b).map(i=>rows.find(r=>r.i===i));
  // windows from the same run of words are merged into one stretch, so nothing is repeated
  const parts=[];const byParent={};for(const r of keep){if(r.parent==null)parts.push({at:r.i,text:r.text});else(byParent[r.parent]??=[]).push(r)}
  for(const [p,l] of Object.entries(byParent)){l.sort((a,b)=>a.ws-b.ws);let cur=null;const spans=[];for(const r of l){if(cur&&r.ws<=cur.we+1){cur.we=Math.max(cur.we,r.we)}else{cur={ws:r.ws,we:r.we};spans.push(cur)}}spans.forEach(sp=>parts.push({at:+p+sp.ws/1000,text:l[0].words.slice(sp.ws,sp.we).join(' ')}))}
  parts.sort((a,b)=>a.at-b.at);
  const prompt=clean(parts.map(x=>x.text.replace(/[.!]+$/,'')).join('. '));
  const also=[];for(const r of [...base].sort((a,b)=>b.score-a.score)){if(also.length>=2)break;if(r.cue&&r.score>=3&&!META.has(r.intent)&&r.intent!==W&&!chosen.has(r.i)&&!also.some(x=>x.text.toLowerCase()===r.text.toLowerCase()))also.push({text:r.text,intent:r.intent})}
  return {prompt,changed:true,ignored:base.length-keep.filter(r=>r.parent==null).length,words,also,mode:'filtered',intent:W,score:+ranked[0][1].toFixed(1),kept:keep.length,dbg:ranked.slice(0,3).map(([i,v])=>i+' '+v.toFixed(1))}}
