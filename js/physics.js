// Physics gate: the same idea as the formula rules, for the calculators. Nothing is calculated from a value that physics forbids.
// Each rule is data: where to find the number in the sentence, and what the quantity is allowed to be, with the reason.
const g=(t,re)=>{const m=t.match(re);return m?parseFloat(m[1].replace(/,/g,'')):null};
const N='(-?\\d[\\d,]*(?:\\.\\d+)?)';
const R=(name,re,lo,hi,why,o={})=>({name,re,lo,hi,why,loOpen:o.loOpen??true,hiOpen:o.hiOpen??false,int:o.int});
export const RULES={
 duct_size:[R('Airflow',new RegExp(N+'\\s*cfm'),0,1e7,'airflow must be above zero'),R('Air velocity',new RegExp(N+'\\s*m\\/s'),0,100,'above about 100 m/s (Mach 0.3) air compresses and the duct formulas stop being valid')],
 airflow_required:[R('Unit size',new RegExp(N+'\\s*(?:tons?|tr)\\b'),0,1e5,'cooling capacity must be above zero')],
 chw_pipe:[R('Cooling duty',new RegExp(N+'\\s*kw'),0,1e7,'cooling duty must be above zero'),R('Delta T',new RegExp(N+'\\s*(?:k\\b|degree|deg|°)'),0,50,'a chilled-water delta T must be above 0 and below about 50 K (water stays liquid within about 100 K)')],
 motor_breaker:[R('Motor power',new RegExp(N+'\\s*(?:kw|hp)'),0,1e5,'motor power must be above zero'),R('Voltage',new RegExp(N+'\\s*v\\b'),0,1e6,'voltage must be above zero')],
 psychro:[R('Dry-bulb temperature',new RegExp(N+'\\s*(?:°|degrees?|deg|c\\b)'),-273.15,100,'below -273.15 C is below absolute zero; above 100 C at sea level water boils and the humidity formulas stop being valid'),R('Relative humidity',new RegExp(N+'\\s*(?:%|percent|per cent)'),0,100,'relative humidity lies between 0 and 100% (0% has no dew point)')],
 energy_cost:[R('Power',new RegExp(N+'\\s*kw'),0,1e7,'power must be above zero'),R('Hours per day',new RegExp(N+'\\s*(?:hours?|hrs?)\\s*(?:per|a|each)\\s*day'),0,24,'a day has only 24 hours'),R('Running hours',new RegExp(N+'\\s*(?:hours?|hrs?|h)\\b'),0,1e6,'hours must be above zero'),R('Tariff',new RegExp('(?:at|tariff|rate)\\s*(?:is|of|=|:)?\\s*(?:rs\\.?|inr|₹|\\$)?\\s*'+N),0,1e5,'a tariff cannot be negative',{loOpen:false})],
 eer_cop:[R('EER',new RegExp('eer\\s*(?:of|is|=|:)?\\s*'+N),0,100,'EER must be above zero'),R('COP',new RegExp('cop\\s*(?:of|is|=|:)?\\s*'+N),0,30,'COP must be above zero (and a real machine is below about 30)')],
 units_convert:[R('Value',new RegExp(N),0,1e9,'a value this large (above a billion) does not exist for any of these quantities',{loOpen:false})],
 tonnage_convert:[R('Capacity',new RegExp(N+'\\s*(?:tons?|tr|kw|btu)'),0,1e9,'capacity must be above zero')],
 labour_cost:[R('Hours',new RegExp(N+'\\s*(?:hours?|hrs?)'),0,1e6,'hours must be above zero'),R('Workers',new RegExp(N+'\\s*(?:men|man|workers?|techs?|technicians?|people|persons?)'),0,1e5,'the number of workers must be above zero',{int:true})],
 invoice_create:[R('Tax',new RegExp(N+'\\s*(?:%|percent)'),0,100,'tax lies between 0 and 100%',{loOpen:false}),R('Hours',new RegExp(N+'\\s*(?:hours?|hrs?)'),0,1e6,'hours must be above zero')],
 home_bill:[R('Hours per day',new RegExp(N+'\\s*(?:hours?|hrs?|h)\\b(?:\\s*(?:a|per|each|every)?\\s*day)?'),0,24,'a day has only 24 hours'),R('Tariff',new RegExp('(?:at|rate|tariff|@)\\s*(?:rs\\.?|₹|inr)?\\s*'+N),0,1e4,'a tariff cannot be negative',{loOpen:false})],
 home_compare_bill:[R('Units',new RegExp(N+'\\s*units'),0,1e7,'units used must be above zero'),R('Bill',new RegExp('(?:bill|rs\\.?|₹)\\s*(?:was|is|of)?\\s*(?:rs\\.?|₹)?\\s*'+N),0,1e9,'a bill cannot be negative',{loOpen:false})],
 home_wiring:[R('Load',new RegExp(N+'\\s*(?:w|watts?)\\b'),0,1e6,'a load must be above zero')],
};
const lbl=(r,x)=>`${r.name} = ${x}: ${r.why} (allowed: ${r.loOpen?'above':'at least'} ${r.lo} and ${r.hiOpen?'below':'at most'} ${r.hi})`;
const out=(x,r)=>x<r.lo||(r.loOpen&&x===r.lo)||x>r.hi||(r.hiOpen&&x===r.hi);
// numbers written as 1e12 become plain digits, so they are judged like any other number
export const plainSci=t=>t.replace(/(\d+(?:\.\d+)?)[eE](\d{1,3})\b/g,(m,a,b)=>String(+a*Math.pow(10,+b)));
export function gate(intent,text){
  const t=text.toLowerCase().replace(/,(?=\d{3}\b)/g,'');const bad=[];
  const mal=t.match(/\d+\.\d+\.\d+[\d.]*/);if(mal)bad.push(`I could not read the number "${mal[0]}" (two decimal points). Write it like 1500.5`);
  const rules=RULES[intent]||[];
  const neg=t.match(/(?:^|[\s(=:@])(-\s?\d[\d.]*)|\bminus\s+\d/);
  if(neg&&intent!=='psychro'&&!(intent in{}) &&(rules.length||/^(cooling_load|home_)/.test(intent)))bad.push(`I read a negative number ("${(neg[1]||neg[0]).trim()}"). The quantities here cannot be negative (a flow, power, size or time below zero does not exist).`);
  for(const r of rules){const x=g(t,r.re);if(x==null)continue;if(out(x,r))bad.push(lbl(r,x));else if(r.int&&!Number.isInteger(x))bad.push(`${r.name} = ${x}: must be a whole number`)}
  if(/^(cooling_load|home_plan|home_lighting|home_fans)$/.test(intent)){
    const d=t.match(/(\d+(?:\.\d+)?)\s*(?:m|metres?|meters?|ft|feet)?\s*(?:by|x|\*)\s*(\d+(?:\.\d+)?)/);
    if(d){const a=+d[1],b=+d[2];if(!(a>0&&b>0))bad.push('A room or house side must be above zero.');if(a>1e4||b>1e4)bad.push(`A side of ${Math.max(a,b)} is beyond any real building (the largest buildings are a few kilometres across).`)}
    const p=g(t,/(\d+)\s*(?:people|persons?|occupants|pax)/);const area=d?+d[1]*+d[2]:g(t,/(\d[\d,]*(?:\.\d+)?)\s*(?:sq|square|m2|m²|sqm)/);
    if(p!=null&&area&&/\bm\b|metres?|meters?|sqm|m2|m²|square metres?/.test(t)&&p>10*area)bad.push(`${p} people in ${area} m²: more than 10 people per m² is a crush and not a normal occupancy.`)}
  return bad}
export const refuse=bad=>({kind:'ask',title:'That breaks a physics rule',hint:'I did not calculate, because:',questions:[...bad,'Please give values that physics allows.']});
// a result containing Infinity / NaN is never shown
export const clean=r=>{if(!r||!r.table)return null;const s=JSON.stringify(r.table.rows);return /Infinity|NaN|∞|undefined/.test(s)?['the result is not a finite number, so the values cannot be right (for example a division by zero)']:null};
export const multi=t=>/\b(and also|and then|then also|as well as|also calculate|also find|additionally)\b/i.test(t);
