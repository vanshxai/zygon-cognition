// Formula library: loads data/formulas.json, evaluates formulas, finds the right one for a sentence, and self-tests.
let DB=null;
export async function load(){DB??=await fetch('data/formulas.json').then(r=>r.json());return DB}
const NS={sqrt:Math.sqrt,sin:Math.sin,cos:Math.cos,tan:Math.tan,asin:Math.asin,acos:Math.acos,atan:Math.atan,exp:Math.exp,log10:Math.log10,ln:Math.log,pi:Math.PI};
const cache={};
export function evaluate(f,inputs){
  const consts=DB.consts,names=[...Object.keys(NS),...Object.keys(consts),...f.vars.map(v=>v.sym)];
  const fn=cache[f.id]??=new Function(...names,'return ('+f.expr+');');
  const vals=[...Object.values(NS),...Object.values(consts).map(c=>c.value),...f.vars.map(v=>v.unit==='deg'?inputs[v.sym]*Math.PI/180:inputs[v.sym])];
  return fn(...vals)}
export async function selfTest(){
  await load();const fails=[];let pass=0;
  for(const f of DB.formulas){const got=evaluate(f,f.test.inputs),want=f.test.expected;const ok=(want===0?Math.abs(got)<1e-9:Math.abs(got-want)/Math.abs(want)<1e-6)&&checkInputs(f,f.test.inputs).length===0&&checkResult(f,got).length===0;ok?pass++:fails.push(f.id)}
  return {pass,total:DB.formulas.length,fails}}
const STOP=new Set('the a an of for to is at in on and or with what how calculate calculator find use using apply solve formula please me my i do does value values if where when from by it this that'.split(' '));
const words=s=>(s.toLowerCase().match(/[a-z0-9]+/g)||[]).filter(w=>!STOP.has(w)&&!/^\d/.test(w));
let IDX=null;
function index(){if(IDX)return IDX;const df={};const docs=DB.formulas.map(f=>{const w=new Set(words([f.name,f.kw,f.domain,f.out.name,f.vars.map(v=>v.name).join(' ')].join(' ')));w.forEach(x=>df[x]=(df[x]||0)+1);return w});
  return IDX={docs,idf:Object.fromEntries(Object.entries(df).map(([w,c])=>[w,Math.log(1+DB.formulas.length/c)]))}}
export function find(text,n=8){
  const {docs,idf}=index(),q=words(text);
  return DB.formulas.map((f,i)=>{let s=0;for(const w of q)if(docs[i].has(w))s+=idf[w];return {f,s}}).filter(x=>x.s>0).sort((a,b)=>b.s-a.s).slice(0,n)}
const UNITRE={ohm:'ohms?|Ω',kg:'kg|kilograms?',m:'m|metres?|meters?',s:'s|sec|seconds?',W:'W|watts?',kW:'kW|kilowatts?',A:'A|amps?',V:'V|volts?',Hz:'Hz',K:'K|kelvin',C:'C|°C',F:'F|°F|farads?',J:'J|joules?',N:'N|newtons?',Pa:'Pa|pascals?',rpm:'rpm',h:'h|hrs?|hours?',deg:'deg|degrees?|°',mol:'mol|moles?',L:'L|litres?',m2:'m2|m²',m3:'m3|m³','m/s':'m/s',bar:'bar',psi:'psi',kWh:'kWh|units',TR:'TR|tons?',hp:'hp',cfm:'cfm',kPa:'kPa'};
const NUM='(-?\\d+(?:\\.\\d+)?(?:[eE]-?\\d+)?)';
// ---- the number, with the unit written right after it, turned into the formula's own unit ----
const CONV={m:{mm:1e-3,cm:1e-2,km:1e3,ft:0.3048,feet:0.3048,inch:0.0254,inches:0.0254,in:0.0254},W:{kw:1e3,mw:1e6,hp:745.7},kW:{w:1e-3,mw:1e3,hp:0.7457},
 s:{min:60,mins:60,minutes:60,ms:1e-3,h:3600,hr:3600,hours:3600},h:{min:1/60,mins:1/60,minutes:1/60,s:1/3600,days:24,day:24},
 Pa:{kpa:1e3,mpa:1e6,bar:1e5,psi:6894.757},m2:{cm2:1e-4,mm2:1e-6,sqft:0.09290304,'sq ft':0.09290304,'sq.ft':0.09290304},m3:{l:1e-3,litres:1e-3,liters:1e-3,cm3:1e-6},
 kg:{g:1e-3,tonne:1e3,tonnes:1e3,ton:1e3},'m3/s':{'l/s':1e-3,'m3/h':1/3600,cfm:0.000471947},deg:{rad:180/Math.PI,radians:180/Math.PI},V:{kv:1e3,mv:1e-3},A:{ma:1e-3},ohm:{kohm:1e3,mohm:1e6}};
function unitize(x,after,v){
  const a=after.match(/^\s*(%|per\s*cent|percent|°\s*c\b|°\s*f\b|degrees?\s*c(?:elsius)?\b|deg\s*c\b|celsius|[a-zA-Z²³/0-9.]+(?:\s*ft)?)/i);const u=a?a[1].replace(/\s+/g,' ').toLowerCase().trim():'';
  const nm=v.name.toLowerCase();
  if(/^(%|percent|per cent)$/.test(u)){return /rate|fraction|efficiency|usable|emissivity|factor|ratio|humidity|slip|coefficient/.test(nm)?x/100:x}
  if(/years?|time in years/.test(nm)&&/^(months?|mos?)$/.test(u))return x/12;
  if(/months/.test(nm)&&/^(years?|yrs?)$/.test(u))return x*12;
  if(v.unit==='K'&&/^(°c|degrees? ?c(elsius)?|deg c|celsius)$/.test(u.replace(/\s/g,'')) )return x+273.15;
  if(v.unit==='K'&&/^(°f)$/.test(u.replace(/\s/g,'')))return (x-32)*5/9+273.15;
  const t=CONV[v.unit];if(t&&t[u]!=null)return x*t[u];
  return x}
const num=s=>parseFloat(s);
// "5 lakh" -> 500000, "2 crore" -> 20000000, "3 million" -> 3000000, "5 thousand" -> 5000
const bigNums=t=>t.replace(/(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|million|thousand)\b/gi,(m,n,u)=>String(+n*({lakh:1e5,lakhs:1e5,lac:1e5,lacs:1e5,crore:1e7,crores:1e7,million:1e6,thousand:1e3}[u.toLowerCase()])));
export function fill(f,text){
  const got={},assumed=[],t=bigNums(text.replace(/,(?=\d{3}\b)/g,''));
  const unitCount={};f.vars.forEach(v=>unitCount[v.unit]=(unitCount[v.unit]||0)+1);
  const take=(v,m)=>{const after=t.slice(m.index+m[0].length);got[v.sym]=unitize(parseFloat(m[1]),after,v)};
  for(const v of f.vars){
    // the symbol must be set apart: "m 5 kg" or "m = 5", never the "m" inside "m3"
    let m=t.match(new RegExp('(?:^|[^A-Za-z0-9])'+v.sym.replace(/[^A-Za-z0-9]/g,'')+'(?:\\s*(?:=|:|is|of)\\s*|\\s+)'+NUM));
    if(!m)for(const w of words(v.name)){m=t.match(new RegExp('\\b'+w+'\\b\\s*(?:=|:|is|of|at)?\\s*'+NUM,'i'));if(m)break}
    if(!m&&unitCount[v.unit]===1&&UNITRE[v.unit])m=t.match(new RegExp(NUM+'\\s*(?:'+UNITRE[v.unit]+')(?![A-Za-z0-9])'));
    if(m)take(v,m)}
  let order=false;
  if(!Object.keys(got).length){const toks=[...t.matchAll(/(?<![A-Za-z0-9.])(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)(?![A-Za-z]?\d)/g)];if(toks.length===f.vars.length||(toks.length&&toks.length<f.vars.length&&f.vars.slice(toks.length).every(v=>v.default!=null))){f.vars.slice(0,toks.length).forEach((v,i)=>take(v,toks[i]));order=true}}
  for(const v of f.vars)if(/rate \(fraction\)/.test(v.name)&&got[v.sym]>1&&got[v.sym]<=100){assumed.push(`${v.sym}: you wrote ${got[v.sym]} with no % sign, so it is read as ${got[v.sym]}% = ${got[v.sym]/100}`);got[v.sym]=got[v.sym]/100}
  for(const v of f.vars)if(got[v.sym]==null&&v.default!=null){got[v.sym]=v.default;assumed.push(`${v.sym} = ${v.default}: ${v.default_note}`)}
  return {got,order,assumed,missing:f.vars.filter(v=>got[v.sym]==null)}}
const fmt=v=>{if(!isFinite(v))return String(v);const a=Math.abs(v);return a!==0&&(a>=1e7||a<1e-3)?v.toExponential(4):String(+v.toPrecision(7))};
// ---- physics rules: every input, every formula condition, and the result are checked against what physics allows ----
const cchk={};
const outside=(x,r)=>r&&((r.lo!=null&&(x<r.lo||(r.lo_open&&x===r.lo)))||(r.hi!=null&&(x>r.hi||(r.hi_open&&x===r.hi))));
const range=r=>{const a=r.lo!=null?(r.lo_open?'above ':'at least ')+fmt(r.lo):'',b=r.hi!=null?(r.hi_open?'below ':'at most ')+fmt(r.hi):'';return [a,b].filter(Boolean).join(' and ')};
export function checkInputs(f,got){
  const bad=[];
  for(const v of f.vars){const x=got[v.sym];if(typeof x!=='number'||!isFinite(x)){bad.push(`${v.name} is not a valid number`);continue}
    if(outside(x,v.rule))bad.push(`${v.name} = ${fmt(x)}${v.unit&&v.unit!=='deg'?' '+v.unit:''}: ${v.rule.why||'outside the allowed range'} (allowed: ${range(v.rule)})`)}
  if(bad.length)return bad;
  const names=f.vars.map(v=>v.sym);
  for(const c of (f.rules?.checks||[])){const fn=cchk[f.id+c.expr]??=new Function(...names,'return ('+c.expr+');');
    let ok=false;try{ok=!!fn(...names.map(n=>got[n]))}catch(e){ok=false}
    if(!ok)bad.push(`${c.why} (condition: ${c.expr.replace(/\*\*/g,'^')})`)}
  return bad}
export function checkResult(f,val){
  if(typeof val!=='number'||!isFinite(val))return ['the result is not a finite number (a division by zero or a root of a negative number), so these values cannot be right'];
  const r=f.rules?.out;if(r&&outside(val,r))return [`the result ${fmt(val)} breaks a physics limit: ${r.why} (allowed: ${range(r)})`];return[]}
const FORMULA_NAME_MIN=6;
export async function solve(text){
  await load();const cands=find(text);
  if(!cands.length)return {ask:true,title:'Which formula?',questions:['Name it, e.g. "Ohm law", "kinetic energy" or "pump power"','Then give the values, e.g. "I 2 A and R 115 ohm"']};
  // a sentence with no number and no clear formula name is a question about what I can do, not a formula request
  if(!/\d/.test(text)&&(cands[0].s<FORMULA_NAME_MIN||text.trim().split(/\s+/).length>=5))return {browse:true};
  let best=null;
  for(const {f,s} of cands){if(s<0.6*cands[0].s)continue;const r=fill(f,text);const filled=f.vars.length-r.missing.length;const sc=s+(r.missing.length===0?(r.order?0.5:4):0)+0.5*filled;if(!best||sc>best.sc)best={f,r,sc}}
  const {f,r}=best;const src=DB.sources[f.src];
  if(r.missing.length)return {ask:true,title:f.name,hint:'I found the formula. I still need:',questions:r.missing.map(v=>`${v.sym} = ${v.name}${v.unit?' ('+v.unit+')':''}`),formula:f,source:src};
  const badIn=checkInputs(f,r.got);
  if(badIn.length)return {ask:true,title:'That breaks a physics rule',hint:`${f.name}. I did not calculate because:`,questions:[...badIn,'Please give values that physics allows.'],formula:f,source:src};
  const val=evaluate(f,r.got);const badOut=checkResult(f,val);
  if(badOut.length)return {ask:true,title:'That breaks a physics rule',hint:`${f.name}. I calculated it but I will not show it because:`,questions:[...badOut,'Please check the values.'],formula:f,source:src};
  const sub=f.expr.replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/g,w=>(w in r.got)?`${fmt(r.got[w])}`:w).replace(/\*\*/g,'^').replace(/\*/g,' × ');
  const nchk=(f.rules?.checks||[]).length;
  const steps=[`Formula (${f.name}): ${f.eq}`,r.order?'Values were taken in the order the formula lists them: '+f.vars.map(v=>v.sym).join(', '):'Values read from your sentence',...r.assumed.map(a=>'Assumed: '+a),`Substitute: ${f.out.sym} = ${sub}`,`Result: ${f.out.sym} = ${fmt(val)}${f.out.unit?' '+f.out.unit:''}`,
    `Physics rules checked: ${f.vars.length} inputs inside their allowed ranges${nchk?`, ${nchk} condition${nchk>1?'s':''} met (${f.rules.checks.map(c=>c.why.split(' (')[0]).join('; ')})`:''}, result is finite${f.rules?.out?' and within its limit':''}`];
  const usedConsts=Object.keys(DB.consts).filter(k=>new RegExp('\\b'+k+'\\b').test(f.expr));
  if(usedConsts.length)steps.push('Constants: '+usedConsts.map(k=>`${k} = ${DB.consts[k].value} ${DB.consts[k].unit} (${DB.consts[k].name})`).join('; '));
  return {title:f.name,subtitle:`${f.domain} · ${f.eq}`,table:{columns:['Quantity','Value','Unit'],rows:[...f.vars.map(v=>[`${v.sym} · ${v.name}`,r.got[v.sym],v.unit]),[`${f.out.sym} · ${f.out.name}`,+(+val).toPrecision(7),f.out.unit]],totalRow:true},steps,source:src,meta:{type:'formula',rules:'checked'}}}
export const all=()=>DB.formulas;export const sources=()=>DB.sources;export const consts=()=>DB.consts;
