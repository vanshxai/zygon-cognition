// Tiny Redis client over the Upstash REST API (no packages needed). Vercel's Upstash/KV integration sets these variables for you.
const URL_=process.env.KV_REST_API_URL||process.env.UPSTASH_REDIS_REST_URL;
const TOKEN=process.env.KV_REST_API_TOKEN||process.env.UPSTASH_REDIS_REST_TOKEN;
exports.configured=()=>!!(URL_&&TOKEN);
// cmds: [["RPUSH","key","value"],["LLEN","key"],...]  ->  array of results
exports.pipeline=async cmds=>{
  const r=await fetch(URL_.replace(/\/$/,'')+'/pipeline',{method:'POST',headers:{Authorization:'Bearer '+TOKEN,'Content-Type':'application/json'},body:JSON.stringify(cmds)});
  if(!r.ok)throw new Error('storage '+r.status);
  const j=await r.json();return j.map(x=>{if(x.error)throw new Error(x.error);return x.result})};
