import {gate,refuse,clean,plainSci} from './physics.js';
// Home electrician mode for Indian households. Every typical value carries its basis:
// "sourced" = taken from a page listed in SRC (retail or manufacturer pages, so treat as typical, not exact),
// "typical" = a common engineering value I could not source this session. Edit any number to your own nameplate.
import {f} from './calc.js';
export const SRC={
  fan:{title:'Atomberg: ceiling fan power consumption (regular fans about 70-80 W)',url:'https://atomberg.com/blog/post/ceiling-fan-power-consumption1'},
  bldc:{title:'Atomberg: BLDC fan vs normal fan (BLDC about 25-35 W)',url:'https://atomberg.com/blog/post/bldc-fan-vs-normal-fan'},
  star5:{title:'Atomberg: 5 star ceiling fan (about 40-50 W at full speed)',url:'https://atomberg.com/blog/post/5-star-ceiling-fan'},
  led9:{title:'EESL Ujala 9 W LED bulb listing (900 lumen)',url:'https://www.flipkart.com/eesl-ujala-9-w-basic-round-b22-led-bulb/p/itm8b48144522be0'},
  fridge:{title:'Bijli Bachao: fridge guide with BEE annual kWh (about 107 kWh/yr direct-cool 220 L, 160 kWh/yr frost-free 300 L)',url:'https://www.bijlibachao.com/refrigerators/fridge-buying-guide.html'},
  lux:{title:'Design forum summary of NBC 2016 lux levels (living 100-200, bedroom 75-150, kitchen 300-500, bathroom 150-300)',url:'https://frontdesk.co.in/forum/showthread.php?tid=2792'},
  nbc:{title:'BIS National Building Code of India 2016 (official; check Part 8 for exact lux)',url:'https://bis.gov.in/others/national-building-code/'},
  mcb:{title:'Choosing MCB rating in India (lighting 6 A, sockets 16 A, AC/geyser 20-25 A)',url:'https://www.studiomatrx.org/guides/choosing-mcb-rating-india'},
  wire:{title:'House wiring guide India (1.5 mm2 lighting, 2.5 mm2 power, 4-6 mm2 heavy loads)',url:'https://www.bricknbolt.com/blogs-and-articles/construction-guide/house-wiring-india-guide'},
  is732:{title:'IS 732 (code of practice for electrical wiring installations)',url:'https://infralens.in/code/IS-732'}};
// id, name, watts, hours/day, duty, basis, source key, lumens, pattern
export const APPS=[
 {id:'bldc_fan',name:'BLDC ceiling fan',w:28,h:10,d:1,b:'sourced',s:'bldc',re:/bldc|atomberg|inverter fan/},
 {id:'fan5',name:'5-star ceiling fan (non-BLDC)',w:50,h:10,d:1,b:'sourced',s:'star5',re:/5[ -]?star (ceiling )?fans?/},
 {id:'table_fan',name:'Table / pedestal fan',w:55,h:6,d:1,b:'typical',re:/table fans?|pedestal fans?|stand fans?/},
 {id:'exhaust',name:'Exhaust fan',w:35,h:2,d:1,b:'typical',re:/exhaust|chimney/},
 {id:'fan',name:'Ceiling fan (regular)',w:75,h:10,d:1,b:'sourced',s:'fan',re:/ceiling fans?|\bfans?\b|pankha/},
 {id:'led_tube',name:'LED tube light 20 W',w:20,h:5,d:1,b:'typical',lm:2000,re:/led (tube|batten)s?|tube ?lights? led|batten/},
 {id:'tube_t8',name:'Tube light T8 36 W + choke',w:44,h:5,d:1,b:'typical',re:/tube ?lights?|tubes?\b|fluorescent|tubelight/},
 {id:'led_bulb',name:'LED bulb 9 W',w:9,h:5,d:1,b:'sourced',s:'led9',lm:900,re:/led (bulb|light|lamp)s?|bulbs?|led\b|lights?|lamps?/},
 {id:'cfl',name:'CFL 15 W',w:15,h:5,d:1,b:'typical',re:/\bcfl/},
 {id:'incand',name:'Incandescent bulb 60 W',w:60,h:3,d:1,b:'typical',re:/incandescent|filament/},
 {id:'tv',name:'LED TV (32-43 inch)',w:60,h:4,d:1,b:'typical',re:/\btvs?\b|television/},
 {id:'fridge_ff',name:'Refrigerator, frost-free ~300 L (BEE kWh/yr as average watts)',w:18.3,h:24,d:1,b:'sourced',s:'fridge',re:/frost[- ]?free|double door/},
 {id:'fridge_dc',name:'Refrigerator, direct-cool ~220 L (BEE kWh/yr as average watts)',w:12.2,h:24,d:1,b:'sourced',s:'fridge',re:/fridge|refrigerator|direct[- ]cool/},
 {id:'ac',name:'Split AC (input about 1000 W per ton)',w:1500,h:8,d:0.6,b:'typical',re:/\bacs?\b|air ?conditioner|split/},
 {id:'geyser',name:'Water heater (geyser) 2000 W',w:2000,h:1,d:0.5,b:'typical',re:/geysers?|water heater|immersion/},
 {id:'wm',name:'Washing machine',w:500,h:1,d:1,b:'typical',re:/washing machine|washer/},
 {id:'iron',name:'Electric iron',w:1000,h:0.5,d:0.7,b:'typical',re:/\biron\b/},
 {id:'mixer',name:'Mixer grinder',w:500,h:0.3,d:1,b:'typical',re:/mixer|grinder/},
 {id:'micro',name:'Microwave oven',w:1200,h:0.3,d:1,b:'typical',re:/microwave|oven/},
 {id:'induction',name:'Induction cooktop',w:1800,h:1,d:0.8,b:'typical',re:/induction/},
 {id:'pump',name:'Water pump 0.5 hp (373 W by 1 hp = 746 W)',w:373,h:1,d:1,b:'typical',re:/pump|motor/},
 {id:'router',name:'Wi-Fi router',w:10,h:24,d:1,b:'typical',re:/router|wifi|wi-fi|modem/},
 {id:'laptop',name:'Laptop with charger',w:65,h:6,d:1,b:'typical',re:/laptop/},
 {id:'desktop',name:'Desktop computer',w:150,h:6,d:1,b:'typical',re:/desktop|computer|\bpc\b/},
 {id:'charger',name:'Phone charger',w:10,h:3,d:1,b:'typical',re:/charger|phone/}];
export const ROOMS=[ // lux = lower end of the range in the NBC-summary source; dining and passage are my assumptions
 {k:'living',name:'Living room',share:.20,lux:100,src:'lux'},{k:'dining',name:'Dining',share:.08,lux:100,src:null},{k:'kitchen',name:'Kitchen',share:.09,lux:300,src:'lux'},
 {k:'bedroom',name:'Bedroom',share:null,lux:75,src:'lux'},{k:'bath',name:'Bathroom',share:.04,lux:150,src:'lux'},{k:'passage',name:'Passage / balcony',share:.08,lux:50,src:null}];
const NW={one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9,ten:10,a:1,an:1};
const NUMW=t=>t.replace(/\b(one|two|three|four|five|six|seven|eight|nine|ten)\b/g,w=>NW[w]);
const UF=0.5,MF=0.8,DAYS=30,FT2=0.09290304;
export const homeLabel={home_plan:'Plan a home: lights and fans',home_bill:'Monthly electricity bill',home_lighting:'Lights for a room',home_fans:'Fan size and count',home_wiring:'MCB and wire sizes',home_compare_bill:'Compare with my real bill',home_watts:'Typical watts of appliances',formula_solve:'Any physics, maths or engineering formula'};
export const HOME_LABEL={...homeLabel,greeting:'Hello',help_me:'What I can do',out_of_scope:'Outside home electricals',new_chat:'New chat',undo:'Undo',show_calc:'Show calculation',edit_value:'Edit a value',export_doc:'Export'};
export const HOME_START=['Estimate my monthly bill: 4 ceiling fans, 12 LED bulbs, 2 tube lights, 1 fridge, 1 AC 1.5 ton for 6 hours, at 7 rupees a unit','3 BHK house 30 by 40 ft, 2 floors, how many lights and fans do I need?','How many LED bulbs for a 12 by 14 ft bedroom?','My bill was 1850 for 260 units, compare it with the estimate','MCB and wire size for a 2000 W geyser','How many watts does a BLDC fan use?'];
const ask=(title,questions,hint)=>({kind:'ask',title,questions,hint});
const tariffOf=t=>{const m=t.match(/(?:at|rate|tariff|@)\s*(?:rs\.?|₹|inr)?\s*(\d+(?:\.\d+)?)/i)||t.match(/(?:rs\.?|₹)\s*(\d+(?:\.\d+)?)\s*(?:per|\/)\s*unit/i)||t.match(/(\d+(?:\.\d+)?)\s*(?:rupees?|rs)\s*(?:per|a|\/)\s*unit/i);return m?parseFloat(m[1]):null};
export function parseItems(text){
  const out=[];
  for(const raw of NUMW(text.toLowerCase()).split(/,|;|\band\b|\bplus\b|\+|\n|\bwith\b/)){
    let c=raw.trim();if(!c)continue;const app=APPS.find(a=>a.re.test(c));if(!app)continue;
    let hours=null,w=null;
    if(/24\s*x\s*7|24\/7|all day|full day|whole day|round the clock/.test(c))hours=24;
    c=c.replace(/(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b(?:\s*(?:a|per|each|every)?\s*day)?/,(m,x)=>{hours=parseFloat(x);return' '});
    c=c.replace(/(\d+(?:\.\d+)?)\s*(?:watts?|w)\b/,(m,x)=>{w=parseFloat(x);return' '});
    c=c.replace(/(\d+(?:\.\d+)?)\s*(?:tons?|tr)\b/,(m,x)=>{w=parseFloat(x)*1000;return' '});
    const q=c.match(/\b(\d+)\b/);out.push({app,qty:q?+q[1]:1,hours,w})}
  return out}
const kwh=(r)=>r[1]*r[2]*r[3]*r[4]*DAYS/1000;
const rowOf=(app,qty,hours,w)=>{const r=[app.name,qty,w??app.w,hours??app.h,app.d,0,0,app.b==='sourced'?'sourced':'typical'];return r};
export function recalcHome(res){
  const tb=res.table,t=res.meta?.tariff??7;if(res.meta?.type!=='homebill')return;
  const body=tb.rows.slice(0,-1);let k=0;
  body.forEach(r=>{r[5]=+kwh(r).toFixed(2);r[6]=+(r[5]*t).toFixed(2);k+=r[5]});
  tb.rows[tb.rows.length-1]=['TOTAL','','','','',+k.toFixed(2),+(k*t).toFixed(2),''];
  res.steps[0]=`Units per item = quantity × watts × hours per day × duty × ${DAYS} days ÷ 1000`;
  res.steps[1]=`Total = ${f(k)} units × ₹${f(t)} per unit = ₹${f(k*t)}`;
  res.subtitle=`${f(k)} units a month · ₹${f(k*t)} at ₹${f(t)} per unit`}
function billFrom(rows,tariff,assumed,extra={}){
  const res={title:'Monthly electricity estimate',subtitle:'',table:{columns:['Item','Qty','Watts','Hours/day','Duty','Units/month','Cost (₹)','Basis'],rows:[...rows,[]],editableCols:[1,2,3,4],totalRow:true},steps:['',''],
    note:(assumed?`Tariff assumed ₹${f(tariff)} per unit. Tell me your rate ("at 6.5") or edit nothing else. `:'')+'Watts and hours are typical values: edit any highlighted cell to your nameplate and the total recalculates.',
    source:SRC.fan,meta:{type:'homebill',tariff,days:DAYS,...extra}};
  recalcHome(res);return res}
function areaOf(t){
  const m=t.match(/(\d+(?:\.\d+)?)\s*(m\b|metres?|meters?|ft|feet|foot)?\s*(?:by|x|\*)\s*(\d+(?:\.\d+)?)\s*(m\b|metres?|meters?|ft|feet|foot)?(?:\s*(?:by|x|\*)\s*(\d+(?:\.\d+)?))?\s*(m\b|metres?|meters?|ft|feet|foot)?/);
  if(m){let l=+m[1],b=+m[3],h=m[5]?+m[5]:null,u=m[6]||m[4]||m[2]||'';const metric=/^m/.test(u);const a=metric?l*b:l*b*FT2;return {areaM2:a,h:h==null?null:(metric?h:h*0.3048),dims:`${l} × ${b}${h?' × '+h:''} ${metric?'m':'ft'}`}}
  const s=t.match(/(\d[\d,]*(?:\.\d+)?)\s*(sq\.? ?ft|square feet|sqft|sq m|sqm|square metres?|square meters?|m2|m²)/);
  if(s){const v=parseFloat(s[1].replace(/,/g,''));return {areaM2:/ft|feet/.test(s[2])?v*FT2:v,h:null,dims:`${f(v)} ${/ft|feet/.test(s[2])?'sq ft':'sq m'}`}}
  return null}
function lightsFor(areaM2,lux){const need=lux*areaM2/(UF*MF);const tube=need>=1800,lm=tube?2000:900,w=tube?20:9,n=Math.max(1,Math.ceil(need/lm));return {need,lm,w,n,type:tube?'LED tube 20 W':'LED bulb 9 W',watts:n*w}}
const fansFor=(sqft)=>sqft<=100?[1,'900 mm (36 in)']:sqft<=150?[1,'1050 mm (42 in)']:sqft<=225?[1,'1200 mm (48 in)']:sqft<=300?[1,'1400 mm (56 in)']:[2,'1200 mm (48 in) each'];
const MCBS=[6,10,16,20,25,32,40,63];const wireFor=i=>i<=10?1.5:i<=16?2.5:i<=25?4:i<=32?6:10;
export async function runHome(intent,text,ctx){
  text=plainSci(text);const bad=gate(intent,text);if(bad.length)return refuse(bad);
  const r=await runHomeCore(intent,text,ctx);const nf=clean(r);if(nf)return refuse(nf);return r}
async function runHomeCore(intent,text,ctx){
  const t=text.toLowerCase();
  switch(intent){
  case 'home_watts':{
    const hit=APPS.filter(a=>a.re.test(NUMW(t)));const list=(hit.length&&!/list|all|common|every/.test(t))?hit.slice(0,1):APPS;
    return {title:'Typical watts',subtitle:list.length===APPS.length?'Common Indian household appliances':list.map(a=>a.name).join(', '),
      table:{columns:['Appliance','Typical watts','Hours/day','Basis'],rows:list.map(a=>[a.name,a.w,a.h,a.b==='sourced'?'sourced':'typical'])},
      steps:['Typical values for planning. Real nameplate watts vary by brand, age and speed.'],note:'"sourced" rows come from retail or manufacturer pages listed under Source; "typical" rows are common values I could not source this session. Check your own nameplate.',
      source:(list.find(a=>a.s)&&SRC[list.find(a=>a.s).s])||SRC.fan,meta:{type:'homewatts'}}}
  case 'home_bill':{
    const items=parseItems(text),tar=tariffOf(text),plan=ctx.current?.meta?.type==='homeplan'?ctx.current:null,cb=ctx.current?.meta?.type==='homebill'?ctx.current:null;
    if(cb&&items.length&&/\b(add|also|include|plus)\b/.test(t)){const r=JSON.parse(JSON.stringify(cb));r.table.rows.splice(r.table.rows.length-1,0,...items.map(x=>rowOf(x.app,x.qty,x.hours,x.w)));if(tar)r.meta.tariff=tar;recalcHome(r);return r}
    let rows=items.map(x=>rowOf(x.app,x.qty,x.hours,x.w));
    if(plan){const C=plan.meta.counts;const add=(id,n)=>{if(n>0&&!rows.some(r=>r[0]===APPS.find(a=>a.id===id).name))rows.unshift(rowOf(APPS.find(a=>a.id===id),n))};add('led_tube',C.tubes);add('led_bulb',C.bulbs);add('fan',C.fans);add('exhaust',C.exhaust)}
    if(!rows.length)return ask('Tell me what is in the house',['How many ceiling fans, LED bulbs, tube lights (and bulbs or tubes of other kinds)','Big appliances: fridge, AC (tons and hours a day), geyser, washing machine, TV','Hours each runs per day (optional, I will use typical hours)','Your rate per unit, e.g. "at 7 rupees" (optional)'],'Example: 4 ceiling fans, 12 LED bulbs, 1 fridge, 1 AC 1.5 ton for 6 hours. Or plan a house first.');
    return billFrom(rows,tar??7,tar==null,{})}
  case 'home_plan':{
    const a=areaOf(t),fl=+(t.match(/(\d+)\s*(?:floors?|storeys?|stories|levels?)/)||[])[1]||1,bhk=+(t.match(/(\d+)\s*bhk/)||[])[1]||(/(\d+)\s*(?:bed ?rooms?)/.test(t)?+t.match(/(\d+)\s*(?:bed ?rooms?)/)[1]:null);
    if(!a)return ask('I need the house size',['Length by breadth, e.g. "30 by 40 ft" (add a height with a third number, or leave it)','Number of floors','Number of bedrooms (e.g. 3 BHK)'],'Example: 3 BHK house 30 by 40 ft, 2 floors.');
    const bed=bhk||3,total=a.areaM2*fl,fixed=ROOMS.filter(r=>r.share).reduce((s,r)=>s+(r.k==='bath'?r.share*bed:r.k==='dining'&&bed<2?0:r.share),0),bedEach=Math.min(.22,Math.max(.08,(1-fixed)/bed));
    const rooms=[];for(const r of ROOMS){if(r.k==='bedroom'){for(let i=1;i<=bed;i++)rooms.push({...r,name:`Bedroom ${i}`,share:bedEach})}else if(r.k==='bath'){for(let i=1;i<=bed;i++)rooms.push({...r,name:`Bathroom ${i}`})}else if(r.k==='dining'&&bed<2)continue;else rooms.push(r)}
    let tubes=0,bulbs=0,fans=0,exhaust=0,watts=0;
    const rows=rooms.map(r=>{const m2=total*r.share,sq=m2/FT2,L=lightsFor(m2,r.lux);let fn=0;if(r.k==='living')fn=sq>200?2:1;else if(r.k==='dining')fn=1;else if(r.k==='bedroom')fn=sq>180?2:1;
      if(L.type.includes('tube'))tubes+=L.n;else bulbs+=L.n;fans+=fn;if(r.k==='kitchen')exhaust=1;watts+=L.watts;return [r.name,+sq.toFixed(0),r.lux,`${L.n} × ${L.type}`,L.watts,fn||(r.k==='kitchen'?'exhaust':'–')]});
    rows.push(['TOTAL',+(total/FT2).toFixed(0),'',`${tubes} tubes, ${bulbs} bulbs`,watts,`${fans}${exhaust?' + 1 exhaust':''}`]);
    return {title:'Home electrical plan',subtitle:`${a.dims}${fl>1?`, ${fl} floors`:''}, ${bed} bedroom${bed>1?'s':''} · built-up about ${f(total/FT2,0)} sq ft`,
      table:{columns:['Room','Area (sq ft)','Light target (lux)','Lights','Light watts','Fans'],rows,totalRow:true},
      steps:[`Built-up area = ${a.dims.replace(' ×',' ×')} × ${fl} floor${fl>1?'s':''} = ${f(total/FT2,0)} sq ft (${f(total)} m2)`,`Each room's share of the area is an assumption (living 20%, kitchen 9%, bathroom 4%, passage 8%, rest to bedrooms).`,`Lumens needed = lux × area ÷ (utilisation ${UF} × maintenance ${MF}). Lux values are the lower end of the range in the NBC summary.`,`Lamp choice: LED bulb 9 W gives 900 lm (EESL listing); LED tube 20 W is taken as 2000 lm (100 lm/W).`,`Fans: 1 per room, 2 in a big living room or bedroom, plus one kitchen exhaust.`],
      note:'This is an estimate of rooms and counts, not exact placement. Say "bill for this house" to get a monthly bill from these lights and fans.',source:SRC.lux,meta:{type:'homeplan',counts:{tubes,bulbs,fans,exhaust}}}}
  case 'home_lighting':{
    const a=areaOf(t);if(!a)return ask('I need the room size',['Length by breadth, e.g. "12 by 14 ft"','Room type: living, bedroom, kitchen, bathroom (optional)']);
    const room=ROOMS.find(r=>t.includes(r.k)||(r.k==='bath'&&/toilet|washroom/.test(t))||(r.k==='living'&&/\b(hall|drawing|lounge)\b/.test(t)))||ROOMS[3];const L=lightsFor(a.areaM2,room.lux);
    return {title:`Lights for a ${room.name.toLowerCase()}`,subtitle:`${a.dims} · target ${room.lux} lux`,table:{columns:['Item','Value'],rows:[['Room area',`${f(a.areaM2/FT2,0)} sq ft (${f(a.areaM2)} m2)`],['Light needed',`${f(L.need,0)} lumens`],['Suggested',`${L.n} × ${L.type}`],['Connected load',`${L.watts} W`]]},
      steps:[`Lumens needed = ${room.lux} lux × ${f(a.areaM2)} m2 ÷ (${UF} × ${MF}) = ${f(L.need,0)} lm`,`Count = ${f(L.need,0)} ÷ ${L.lm} lm per lamp, rounded up = ${L.n}`],source:SRC.lux,meta:{type:'homelight'}}}
  case 'home_fans':{
    const a=areaOf(t);if(!a)return ask('I need the room size',['Length by breadth, e.g. "12 by 14 ft"']);const sq=a.areaM2/FT2,[n,sweep]=fansFor(sq);
    const reg=n*75*10*DAYS/1000,bl=n*28*10*DAYS/1000;
    return {title:'Fan size and count',subtitle:/sq ft/.test(a.dims)?a.dims:`${a.dims} · about ${f(sq,0)} sq ft`,table:{columns:['Item','Value'],rows:[['Fans',n],['Sweep',sweep],['Regular fan, 10 h a day',`${f(reg)} units/month`],['BLDC fan, 10 h a day',`${f(bl)} units/month`],['Saving with BLDC',`${f(reg-bl)} units/month`]]},
      steps:['Sweep by room area is a manufacturer rule of thumb (up to 100 sq ft: 900 mm; 100-150: 1050 mm; 150-225: 1200 mm; 225-300: 1400 mm; more: two fans).','Regular fan 75 W and BLDC fan 28 W are typical values from the Atomberg pages.'],note:'The sweep table is a rule of thumb I have not sourced to a standard.',source:SRC.bldc,meta:{type:'homefan'}}}
  case 'home_wiring':{
    const items=parseItems(text).filter(x=>x.app.w>=200||x.w);
    if(!items.length)return {title:'Typical home circuits',subtitle:'Indian residential practice',table:{columns:['Circuit','MCB','Copper wire','Basis'],rows:[['Lights and fans',`6 A`,'1.5 mm2','sourced'],['Power sockets','16 A','2.5 mm2','sourced'],['Geyser','20-25 A','4 mm2','sourced'],['1-1.5 ton AC','20-25 A','4 mm2','sourced'],['2 ton AC','32 A','6 mm2','sourced']]},
      steps:['Rule: load current ≤ MCB rating ≤ cable safe current.','A 30 mA RCCB is commonly added for shock protection.'],note:'Sources are wiring guides, not the full IS 732 tables. A licensed electrician must confirm final sizes.',source:SRC.mcb,meta:{type:'homewire'}};
    const rows=items.map(x=>{const w=(x.w??x.app.w)*x.qty,res=/geyser|iron|induction|heater/.test(x.app.id)||x.app.id==='geyser',pf=res?1:0.9,I=w/(230*pf),mcb=MCBS.find(m=>m>=I*1.25)||63;return [x.app.name,x.qty,w,+I.toFixed(1),`${mcb} A`,`${wireFor(mcb)} mm2`]});
    return {title:'MCB and wire for these loads',subtitle:'230 V single phase',table:{columns:['Load','Qty','Watts','Current (A)','MCB','Copper wire'],rows},steps:['Current = watts ÷ (230 V × power factor). Power factor 1 for heaters, 0.9 for motors and AC.','MCB = next standard size at or above 1.25 × current.','Wire: up to 10 A 1.5 mm2, 16 A 2.5 mm2, 25 A 4 mm2, 32 A 6 mm2, 40 A 10 mm2 (typical wiring-guide values).'],note:'A licensed electrician must confirm final sizes against IS 732 and your wiring length.',source:SRC.wire,meta:{type:'homewire'}}}
  case 'home_compare_bill':{
    const cb=ctx.current?.meta?.type==='homebill'?ctx.current:null;
    const u=(t.match(/(\d+(?:\.\d+)?)\s*units?/)||[])[1],b=(t.match(/(?:bill|paid|amount|rs\.?|₹)\s*(?:was|of|is|:)?\s*(?:rs\.?|₹)?\s*(\d[\d,]*(?:\.\d+)?)/)||[])[1];
    if(!cb)return ask('Make an estimate first',['Ask for a monthly bill estimate with your appliances, then say "my bill was 1850 for 260 units".']);
    if(u==null&&b==null)return ask('What was the real bill?',['Units used from the bill (e.g. 260 units)','Amount paid (e.g. 1850)']);
    const rows=cb.table.rows,est=rows[rows.length-1][5],estCost=rows[rows.length-1][6],uA=u!=null?+u:null,bA=b!=null?+b.replace(/,/g,''):null,out=[];
    if(uA!=null)out.push(['Units',est,uA,`${f(est-uA)} (${f((est-uA)/uA*100,1)}%)`]);if(bA!=null)out.push(['Cost (₹)',estCost,bA,`${f(estCost-bA)} (${f((estCost-bA)/bA*100,1)}%)`]);
    if(uA!=null&&bA!=null)out.push(['Effective rate (₹/unit)',cb.meta.tariff,+(bA/uA).toFixed(2),'bill ÷ units']);
    const top=[...rows.slice(0,-1)].sort((x,y)=>y[5]-x[5]).slice(0,3);
    return {title:'Estimate vs real bill',subtitle:'Use the effective rate from your bill to remove tariff error',table:{columns:['Measure','Estimate','Real bill','Difference'],rows:out},
      steps:[uA!=null?`Unit error = (estimate − real) ÷ real = (${f(est)} − ${f(uA)}) ÷ ${f(uA)} = ${f((est-uA)/uA*100,1)}%`:'Units not given.',`Biggest items in the estimate: ${top.map(r=>`${r[0]} ${f(r[5])} units`).join('; ')}`,'Fix the biggest items first: check their real hours and watts.'],note:'Your real bill also has fixed charges, taxes and slabs, so compare units first, then cost.',meta:{type:'homecompare'}}}
  default:return null}
}
