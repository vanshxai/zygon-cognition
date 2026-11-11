// Hexonet v0 engine. Three memory spaces: QUERY (temporary), KEY (bridge, counts), VALUE (permanent).
// v0 uses Frost word->task strengths (counting + add/subtract learning). The angle-tracing core plugs in behind trace()/teach().
const get=(k,d)=>{try{return JSON.parse(localStorage.getItem(k))??d}catch{return d}};
const put=(k,v)=>{try{localStorage.setItem(k,JSON.stringify(v))}catch{}};
export class Hexonet{
  constructor(model,key=''){this.key=key;this.intents=model.intents;this.S=model.S;this.Q=get('hx_query'+key,{});this.K=get('hx_key'+key,{});this.V=get('hx_value'+key,{});}
  tok(s){return (s.toLowerCase().match(/[a-z0-9&]+(?:\.\d+)?/g)||[]).map(w=>/^\d+(\.\d+)?$/.test(w)?'<num>':w)}
  words(s){return [...new Set(this.tok(s))]}
  score(s){const sc={};for(const i of this.intents)sc[i]=0;
    for(const w of this.words(s))for(const L of [this.S[w],this.V[w],this.Q[w]])if(L)for(const i in L)sc[i]=(sc[i]||0)+L[i];return sc}
  trace(s,n=3){const sc=this.score(s);const top=Object.entries(sc).sort((a,b)=>b[1]-a[1]).slice(0,n).map(([intent,score])=>({intent,score}));
    return {top,margin:top[0].score-(top[1]?.score??0),confident:top[0].score>1.0&&top[0].score-(top[1]?.score??0)>=1.0}}
  teach(s,intent){ // add/subtract only; step size = how far off we are
    const ws=this.words(s);let moved=0;
    for(let r=0;r<8;r++){const sc=this.score(s);const rival=Object.entries(sc).filter(([i])=>i!==intent).sort((a,b)=>b[1]-a[1])[0][0];
      const gap=sc[rival]-sc[intent]+0.5;if(gap<=0)break;const step=gap/(2*ws.length);
      for(const w of ws){(this.Q[w]??={});this.Q[w][intent]=(this.Q[w][intent]||0)+step;this.Q[w][rival]=(this.Q[w][rival]||0)-step;}moved++}
    const sig=ws.join(' ');const k=(this.K[sig]??={intent,count:0});if(k.intent!==intent){k.intent=intent;k.count=0}k.count++;
    let promoted=false;
    if(k.count>=3){for(const w of ws){if(!this.Q[w])continue;(this.V[w]??={});for(const i in this.Q[w])this.V[w][i]=(this.V[w][i]||0)+this.Q[w][i];delete this.Q[w]}promoted=true;k.count=0}
    put('hx_query'+this.key,this.Q);put('hx_key'+this.key,this.K);put('hx_value'+this.key,this.V);return {moved,promoted,space:promoted?'value (permanent)':'query (temporary)'}}
  stats(){return {queryWords:Object.keys(this.Q).length,keyBridges:Object.keys(this.K).length,valueWords:Object.keys(this.V).length,baseWords:Object.keys(this.S).length}}
}
