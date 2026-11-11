// Explainer notes: short, plain-language answers to "what is / explain / how does ... work" questions.
// They are stored text, found by the same idea as the router: words vote for the topic that owns them (weighted by how rare the word is).
let KB=null,IDX=null,VOCAB=[];
export const KMIN=3;   // a topic must score at least this to be offered
const STOP=new Set('a an the of for to is are was be me my i you we it its this that these those and or in on at by with about into from what whats how do does did can could would should please tell explain explanation understand know want need learn teach give show list there here some any all just very also more most much many like which who when where why if so then than but not no yes ok okay thing things stuff something one ones use used using work works working mean means meaning your our their his her he she they them get got make made way ways own give'.split(' '));
const stem=w=>{w=w.toLowerCase();if(w.length>4)w=w.replace(/ies$/,'y').replace(/(ing|ed)$/,'').replace(/es$/,'e');if(w.length>3)w=w.replace(/s$/,'');return w};
const toks=s=>(String(s).toLowerCase().match(/[a-z0-9][a-z0-9.+/]*/g)||[]).map(w=>w.replace(/[.]+$/,''));
const words=s=>toks(s).filter(w=>!STOP.has(w)).map(stem).filter(w=>w.length>1);
const gram=w=>{const t='^'+w+'$',g=new Set();for(let i=0;i<t.length-2;i++)g.add(t.slice(i,i+3));return g};
const dice=(a,b)=>{const A=gram(a),B=gram(b);let n=0;for(const x of A)if(B.has(x))n++;return 2*n/(A.size+B.size)};
export async function loadKB(){if(KB)return KB;KB=await fetch('data/knowledge.json').then(r=>r.json());build();return KB}
function build(){
  const df={};const docs=KB.map(t=>{const w={};for(const x of words(t.title))w[x]=(w[x]||0)+2;for(const k of t.kw)for(const x of words(k))w[x]=(w[x]||0)+1;for(const x of Object.keys(w))df[x]=(df[x]||0)+1;return w});
  const idf={};for(const x in df)idf[x]=Math.log(1+KB.length/df[x]);IDX={docs,idf};VOCAB=Object.keys(df).filter(x=>x.length>=5)}
export function topicById(id){return KB?.find(t=>t.id===id)}
export const topicList=mode=>(KB||[]).filter(t=>t.mode===mode||t.mode==='both');
// what the person is asking for: an explanation, not a calculation
export const explainCue=t=>/\b(explain|explanation|what is|what's|what are|what does|what do .* mean|tell me about|tell me more|teach me|how does|how do .* work|how it works|meaning of|difference between|basics of|basics|learn about|know about|understand|guide to|why\b|i don't know|i dont know|no idea|overview|introduction)\b/i.test(t);
// returns the best topics for a sentence: [{topic,score}]
export function findTopics(text,mode,n=3){
  if(!IDX)return [];const q=[...new Set(words(text))];if(!q.length)return [];
  const low=' '+toks(text).join(' ')+' ';
  // a misspelt word is read as the closest known word
  const fixed=q.map(w=>{if(IDX.idf[w]!==undefined||w.length<5)return w;let best=null,bs=0.62;for(const v of VOCAB){if(v[0]!==w[0])continue;const d=dice(w,v);if(d>bs){bs=d;best=v}}return best||w});
  return KB.map((t,i)=>{let s=0;const d=IDX.docs[i];for(const w of fixed)if(d[w])s+=IDX.idf[w]*Math.min(d[w],2.5);
    for(const k of t.kw)if(k.includes(' ')&&low.includes(' '+k+' '))s+=(IDX.idf[stem(k.split(' ')[0])]||1.5)*1.2;   // a whole phrase such as "dew point" is stronger than its parts
    if(t.mode===mode||t.mode==='both')s*=1.25;return {topic:t,score:s}}).sort((a,b)=>b.score-a.score).slice(0,n)}
export function knowledgeResult(t,mode){
  const here=t.mode===mode||t.mode==='both';
  return {kind:'know',title:t.title,short:t.short,points:t.points,numbers:t.numbers,
    tryit:here?t.tryit:[],related:(t.related||[]).map(id=>topicById(id)).filter(Boolean).map(x=>({id:x.id,title:x.title})),
    source:t.source,switch:here?undefined:t.mode,meta:{topic:t.id}}}
export function fallbackResult(mode,starts){
  const ts=topicList(mode);
  return {kind:'know',title:"I'm not sure I understood that",short:mode==='home'?'I can explain home electrical basics and work out bills, lights, fans, MCB and wire sizes. Here are things you can ask, or pick a topic to learn about.':'I can explain HVAC basics and work out duct sizes, cooling load, breakers, pipes and costs. Here are things you can ask, or pick a topic to learn about.',
    points:[],numbers:[],tryit:starts.slice(0,4),topics:ts.map(x=>({id:x.id,title:x.title})),related:[],meta:{topic:'(none)'}}}
export function missResult(mode,starts){const r=fallbackResult(mode,starts);r.title="I don't have a note on that yet";r.short="I don't know that one. I only explain the topics below, and I can calculate things. What you asked is saved, so a note can be added if it fits.";r.meta={topic:'(miss)'};return r}
