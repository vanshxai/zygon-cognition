// Pure HVAC calculators. Every function returns {values, steps[]} so the user can always see the working.
export const f=(n,d=2)=>Number(n).toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:0});
export const TR_KW=3.51685, BTU_KW=3412.14;
const nextOf=(arr,x)=>arr.find(v=>v>=x)??arr[arr.length-1];
export const TR_SIZES=[1,1.5,2,2.5,3,4,5,7.5,10,15,20,25,30,40,50,75,100,150,200,300,500];
export const DUCT_MM=[100,125,150,160,200,250,300,315,350,400,450,500,550,600,700,800,900,1000,1200,1500];
export const DN=[15,20,25,32,40,50,65,80,100,125,150,200,250,300,350,400,450,500,600];
export const BREAKER=[6,10,16,20,25,32,40,50,63,80,100,125,160,200,250,315,400,500,630];
export const W_PER_M2={office:110,'server room':500,'restaurant kitchen':250,'conference room':150,'retail shop':130,bedroom:90,'living room':95,warehouse:60,'hospital ward':120,classroom:130,gym:160,lobby:100,clinic:120,'bank branch':120,lab:160,hall:130,shop:130,restaurant:180,default:110};
export function coolingLoad({areaM2,people=0,room='default'}){
  const wpm=W_PER_M2[room]??W_PER_M2.default, base=areaM2*wpm, ppl=people*130, total=base+ppl, tr=total/1000/TR_KW;
  return {values:{totalKW:total/1000,tr,btu:total/1000*BTU_KW,pick:nextOf(TR_SIZES,tr),wpm},steps:[
   `Area = ${f(areaM2)} m2 (${f(areaM2*10.7639,0)} sq ft)`,`Space type "${room}": ${wpm} W per m2 covers envelope, lights and normal equipment (screening value)`,
   `Base load = ${f(areaM2)} x ${wpm} = ${f(base,0)} W`,`People = ${people} x 130 W (75 sensible + 55 latent) = ${f(ppl,0)} W`,
   `Total = ${f(total,0)} W = ${f(total/1000)} kW`,`Tons of refrigeration = ${f(total/1000)} / ${TR_KW} = ${f(tr)} TR`,`Next standard unit size: ${nextOf(TR_SIZES,tr)} TR`]};
}
export function ductSize({cfm,velMs=4}){
  const q=cfm*0.000471947, a=q/velMs, d=Math.sqrt(4*a/Math.PI)*1000, std=nextOf(DUCT_MM,d), vAct=q/(Math.PI*(std/1000)**2/4);
  return {values:{q,areaM2:a,dMm:d,std,vAct},steps:[`Flow = ${f(cfm,0)} cfm x 0.000471947 = ${f(q,4)} m3/s (${f(q*3600,0)} m3/h)`,`Target velocity = ${f(velMs)} m/s (${f(velMs*196.85,0)} fpm)`,`Area = flow / velocity = ${f(q,4)} / ${f(velMs)} = ${f(a,4)} m2`,`Diameter = sqrt(4 x area / pi) = ${f(d,0)} mm`,`Next standard round size: ${std} mm, giving ${f(vAct)} m/s`]};
}
export function airflowFromTons({tons,cfmPerTon=400}){const cfm=tons*cfmPerTon;return {values:{cfm,m3h:cfm*1.699,ls:cfm*0.4719},steps:[`Comfort cooling rule of thumb: ${cfmPerTon} cfm per ton`,`Airflow = ${f(tons)} x ${cfmPerTon} = ${f(cfm,0)} cfm`,`= ${f(cfm*1.699,0)} m3/h = ${f(cfm*0.4719,0)} L/s`]};}
export function chwPipe({kw,dT=5,velMs=1.5}){
  const lps=kw/(4.186*dT), q=lps/1000, d=Math.sqrt(4*q/(Math.PI*velMs))*1000, dn=nextOf(DN,d), vAct=q/(Math.PI*(dn/1000)**2/4);
  return {values:{lps,gpm:lps*15.8503,d,dn,vAct},steps:[`Water flow = kW / (4.186 x dT) = ${f(kw)} / (4.186 x ${dT}) = ${f(lps)} L/s (${f(lps*15.8503,0)} gpm)`,`Target velocity ${f(velMs)} m/s`,`Diameter = sqrt(4 x flow / (pi x v)) = ${f(d,0)} mm`,`Next nominal size: DN${dn}, giving ${f(vAct)} m/s (inner diameter taken as nominal; check your pipe schedule)`]};
}
export function motorBreaker({kw,v=415,ph=3,pf=0.85,eff=0.9}){
  const I=ph===3?kw*1000/(Math.sqrt(3)*v*pf*eff):kw*1000/(v*pf*eff), min=I*1.25, br=nextOf(BREAKER,min);
  return {values:{I,min,br},steps:[`Shaft power ${f(kw)} kW, ${ph}-phase, ${v} V, assumed pf ${pf}, efficiency ${eff} (use nameplate values when you have them)`,`Current = ${ph===3?`${f(kw*1000,0)} / (1.732 x ${v} x ${pf} x ${eff})`:`${f(kw*1000,0)} / (${v} x ${pf} x ${eff})`} = ${f(I)} A`,`Breaker minimum = 125% x ${f(I)} = ${f(min)} A`,`Next standard rating: ${br} A (final choice depends on your local code and starter type)`]};
}
export function convertTons(x,unit){
  let kw; if(unit==='tr')kw=x*TR_KW; else if(unit==='kw')kw=x; else kw=x/BTU_KW;
  return {values:{tr:kw/TR_KW,kw,btu:kw*BTU_KW},steps:[`1 TR = ${TR_KW} kW = 12,000 BTU/h`,`${f(x)} ${unit==='btu'?'BTU/h':unit.toUpperCase()} -> ${f(kw)} kW = ${f(kw/TR_KW)} TR = ${f(kw*BTU_KW,0)} BTU/h`]};
}
export function energyCost({kw,hours,tariff}){const kwh=kw*hours;return {values:{kwh,cost:kwh*tariff},steps:[`Energy = ${f(kw)} kW x ${f(hours,0)} h = ${f(kwh)} kWh`,`Cost = ${f(kwh)} x ${f(tariff)} = ${f(kwh*tariff)}`,`Assumes the unit draws ${f(kw)} kW the whole time (real compressors cycle, so actual use is usually lower)`]};}
export function eerCop({eer,cop}){if(eer!=null){const c=eer/3.412;return {values:{eer,cop:c,kwPerTr:12/eer},steps:[`COP = EER / 3.412 = ${f(eer)} / 3.412 = ${f(c)}`,`kW per TR = 12 / EER = ${f(12/eer)}`]};}const e=cop*3.412;return {values:{eer:e,cop,kwPerTr:TR_KW/cop},steps:[`EER = COP x 3.412 = ${f(cop)} x 3.412 = ${f(e)}`,`kW per TR = 3.517 / COP = ${f(TR_KW/cop)}`]};}
export function psychro({db,rh,P=101.325}){
  const ps=0.6112*Math.exp(17.62*db/(243.12+db)), pv=rh/100*ps, W=0.622*pv/(P-pv), g=Math.log(rh/100)+17.62*db/(243.12+db), td=243.12*g/(17.62-g), h=1.006*db+W*(2501+1.86*db);
  return {values:{td,W:W*1000,h,pv,ps},steps:[`Saturation pressure at ${db} C = 0.6112 x exp(17.62 x ${db} / (243.12 + ${db})) = ${f(ps,3)} kPa`,`Vapour pressure = ${rh}% x ${f(ps,3)} = ${f(pv,3)} kPa`,`Dew point (Magnus) = ${f(td)} C`,`Humidity ratio = 0.622 x ${f(pv,3)} / (${P} - ${f(pv,3)}) = ${f(W*1000)} g/kg dry air`,`Enthalpy = 1.006 x ${db} + W x (2501 + 1.86 x ${db}) = ${f(h)} kJ/kg (sea-level pressure ${P} kPa)`]};
}
export const UNITS={'cfm to m3/h':x=>x*1.69901,'m3/h to cfm':x=>x/1.69901,'feet to metres':x=>x*0.3048,'metres to feet':x=>x/0.3048,'psi to bar':x=>x*0.0689476,'bar to psi':x=>x/0.0689476,'tons to kw':x=>x*TR_KW,'kw to tons':x=>x/TR_KW,'btu/h to kw':x=>x/BTU_KW,'kw to btu/h':x=>x*BTU_KW,'celsius to fahrenheit':x=>x*9/5+32,'fahrenheit to celsius':x=>(x-32)*5/9,'litres per second to gpm':x=>x*15.8503,'gpm to litres per second':x=>x/15.8503};
