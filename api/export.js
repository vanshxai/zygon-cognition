// Lets the developer read the shared chats. Send the header  x-admin-key: <ADMIN_KEY>  (ADMIN_KEY is a setting you add in Vercel).
//   GET /api/export?users=1          -> list of anonymous users with counts
//   GET /api/export?uid=<id>         -> every event of one user
//   GET /api/export?all=1            -> every event of every user (up to 20,000)
const crypto=require('crypto');const R=require('../lib/redis.js');
const send=(res,code,obj)=>{res.statusCode=code;res.setHeader('content-type','application/json');res.setHeader('cache-control','no-store');res.end(JSON.stringify(obj))};
const same=(a,b)=>{const x=Buffer.from(String(a)),y=Buffer.from(String(b));return x.length===y.length&&crypto.timingSafeEqual(x,y)};
const parse=a=>(a||[]).map(s=>{try{return JSON.parse(s)}catch{return null}}).filter(Boolean);
module.exports=async(req,res)=>{
  const key=process.env.ADMIN_KEY;
  if(!key)return send(res,503,{error:'ADMIN_KEY is not set in Vercel'});
  if(!same(req.headers['x-admin-key']||'',key))return send(res,401,{error:'wrong key'});
  if(!R.configured())return send(res,503,{error:'storage not configured'});
  const q=new URL(req.url,'http://x').searchParams;
  try{
    if(q.get('uid')){const uid=q.get('uid');if(!/^[A-Za-z0-9-]{8,64}$/.test(uid))return send(res,400,{error:'bad id'});const [l,u]=await R.pipeline([['LRANGE','log:'+uid,0,-1],['HGETALL','user:'+uid]]);return send(res,200,{uid,events:parse(l)})}
    const ids=(await R.pipeline([['SMEMBERS','users']]))[0]||[];
    if(q.get('all')){const out=[];for(let i=0;i<ids.length&&out.length<20000;i+=20){const part=ids.slice(i,i+20);const lists=await R.pipeline(part.map(id=>['LRANGE','log:'+id,0,-1]));lists.forEach((l,k)=>parse(l).forEach(e=>out.push({uid:part[k],...e})))}return send(res,200,{events:out.slice(0,20000)})}
    const info=await R.pipeline(ids.flatMap(id=>[['LLEN','log:'+id],['HGETALL','user:'+id]]));
    const users=ids.map((id,i)=>{const h=info[i*2+1]||[];const o={};for(let k=0;k<h.length;k+=2)o[h[k]]=h[k+1];return {uid:id,events:info[i*2],first:+o.first||0,last:+o.last||0,country:o.cc||'',version:o.v||''}}).sort((a,b)=>b.last-a.last);
    return send(res,200,{users})
  }catch(e){return send(res,500,{error:'storage error'})}};
