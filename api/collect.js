// Receives the chats that people agreed to share. POST { uid, v, events:[...] }  or  { uid, op:"forget" }.
const R=require('../lib/redis.js');
const MAX_BODY=150000, MAX_EVENTS=50, MAX_EVENT=12000, MAX_PER_HOUR=1200, KEEP_PER_USER=5000;
const send=(res,code,obj)=>{res.statusCode=code;res.setHeader('content-type','application/json');res.setHeader('cache-control','no-store');res.end(JSON.stringify(obj))};
const readBody=req=>new Promise((ok,no)=>{
  if(req.body!==undefined&&req.body!==null){try{ok(typeof req.body==='string'?JSON.parse(req.body||'{}'):(Buffer.isBuffer(req.body)?JSON.parse(req.body.toString()||'{}'):req.body))}catch(e){no(e)}return}
  let s='';req.on('data',c=>{s+=c;if(s.length>MAX_BODY){no(new Error('too big'));req.destroy()}});req.on('end',()=>{try{ok(JSON.parse(s||'{}'))}catch(e){no(e)}});req.on('error',no)});
module.exports=async(req,res)=>{
  if(req.method!=='POST')return send(res,405,{error:'POST only'});
  if(!R.configured())return send(res,503,{error:'storage not configured'});
  let b;try{b=await readBody(req)}catch{return send(res,400,{error:'bad body'})}
  const uid=String(b.uid||'');if(!/^[A-Za-z0-9-]{8,64}$/.test(uid))return send(res,400,{error:'bad id'});
  try{
    if(b.op==='forget'){await R.pipeline([['DEL','log:'+uid],['DEL','user:'+uid],['SREM','users',uid]]);return send(res,200,{ok:true})}
    const evs=Array.isArray(b.events)?b.events.slice(0,MAX_EVENTS):[];if(!evs.length)return send(res,200,{ok:true,stored:0});
    const now=Date.now(),cc=String(req.headers['x-vercel-ip-country']||'').slice(0,2),ua=String(req.headers['user-agent']||'').slice(0,140);
    const rows=[];for(const e of evs){if(!e||typeof e!=='object')continue;const s=JSON.stringify({...e,ts:now,cc});if(s.length<=MAX_EVENT)rows.push(s)}
    if(!rows.length)return send(res,200,{ok:true,stored:0});
    const hour=Math.floor(now/3600000);
    const [n]=await R.pipeline([['INCRBY','rl:'+uid+':'+hour,rows.length],['EXPIRE','rl:'+uid+':'+hour,7200]]);
    if(n>MAX_PER_HOUR)return send(res,429,{error:'slow down'});
    await R.pipeline([['RPUSH','log:'+uid,...rows],['LTRIM','log:'+uid,-KEEP_PER_USER,-1],['SADD','users',uid],['HSETNX','user:'+uid,'first',String(now)],['HSET','user:'+uid,'last',String(now),'ua',ua,'v',String(b.v||'').slice(0,20),'cc',cc]]);
    return send(res,200,{ok:true,stored:rows.length})
  }catch(e){return send(res,500,{error:'storage error'})}};
