"""1) what does Frost answer to a one-word prompt?   2) every number of LAYER 1 (index 0) for that word's own token.
    python the_layer1.py [word]   ->  model/frost_work/layer1_<word>.json   (then: python plain_layer.py <word>)
"""
import json, os, sys, time
import numpy as np
from engine import Model, chatml, rmsnorm

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
word = sys.argv[1] if len(sys.argv) > 1 else "the"
SYSTEM = "You are Frost 1.0, a helpful assistant."


def R(x):
    return float(f"{float(x):.5g}")


class Stop(Exception):
    pass


m = Model(verbose=False)
end = m.tok.id_of["<|im_end|>"]
ids = m.tok.encode(chatml([("system", SYSTEM), ("user", word)]))
T = len(ids)
pre = m.tok.encode(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n")
P = len(pre)
print(f"{T} tokens; the word is token #{P}: id {ids[P]} {m.tok.decode([ids[P]])!r}", flush=True)
assert m.tok.decode([ids[P]]) == word

# ---------------------------------------------------------------- 1. the replies
def reply(temp, max_new, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    cache = m.new_cache(T + max_new + 8); lg = m.forward(ids, cache); out = []
    for _ in range(max_new):
        t = m.sample(lg, temp=temp, top_k=20, top_p=0.8, rng=rng)
        if t in (end, m.tok.eos): break
        out.append(t); lg = m.forward([t], cache)
    return m.tok.decode(out), len(out), (len(out) < max_new)


t0 = time.time()
greedy, n_g, done_g = reply(0, 160); print(f"greedy ({n_g} tokens, finished={done_g}, {time.time() - t0:.0f}s):\n{greedy}\n", flush=True)
samples = []
for sd in (1, 2, 3):
    txt, n_s, fin = reply(0.7, 50, sd); samples.append({"seed": sd, "text": txt, "tokens": n_s, "finished": fin}); print(f"sampled (seed {sd}): {txt!r}", flush=True)

# ---------------------------------------------------------------- 2. layer 1 of the word's own token
cap = {}


def hook(l, pt, a):
    if l == 0:
        cap[pt] = np.array(a)
        if pt == "resid_out":
            raise Stop()


cache = m.new_cache(T + 8)
try:
    m.forward(ids, cache, hook)
except Stop:
    pass
Wt = m.W; p0 = "blk.0."; hd, rep = m.hd, m.n_heads // m.n_kv
x, h1 = cap["resid_in"][P], cap["attn_in"][P]
q, k, v = cap["q"][:, P, :], cap["k"][:, P, :], cap["v"][:, P, :]
ctx, ao, xm, h2 = cap["attn_ctx"][P], cap["attn_out"][P], cap["resid_mid"][P], cap["ffn_in"][P]
g, u, act, mo, xo = cap["gate"][P], cap["up"][P], cap["act"][P], cap["mlp_out"][P], cap["resid_out"][P]


def mm(Wm, vec, bias, out, rows=4):
    prod = Wm[:rows].astype(np.float64) * vec[None, :].astype(np.float64)
    for r in range(rows):
        assert abs(prod[r].sum() + float(bias[r]) - float(out[r])) <= 2e-3 * max(1.0, abs(float(out[r]))), (r, prod[r].sum(), out[r])
    return {"W": [[R(a) for a in Wm[r, :8]] for r in range(rows)], "h": [R(a) for a in vec[:8]], "part": [R(prod[r, :8].sum()) for r in range(rows)],
            "rest": [R(prod[r, 8:].sum()) for r in range(rows)], "b": [R(bias[r]) for r in range(rows)], "out": [R(out[r]) for r in range(rows)], "cols": int(Wm.shape[1]), "rows_total": int(Wm.shape[0])}


def receipt(w, vec, out, bias=0.0, n=6):
    prod = w.astype(np.float64) * vec.astype(np.float64)
    assert abs(prod.sum() + float(bias) - float(out)) <= 2e-3 * max(1.0, abs(float(out)))
    return {"terms": [[R(w[i]), R(vec[i]), R(prod[i])] for i in range(n)], "n": int(len(w)), "rest": R(prod[n:].sum()), "bias": R(bias), "out": R(out)}


def norm(vec, gain, out):
    ms = float((vec.astype(np.float64) ** 2).mean()); rms = float(np.sqrt(ms + m.eps)); assert np.allclose(vec / rms * gain, out, rtol=1e-3, atol=1e-4)
    return {"ms": R(ms), "rms": R(rms), "eps": m.eps, "e": [[R(vec[i]), R(gain[i]), R(out[i])] for i in range(8)]}


Wq, Wk, Wv, Wo = Wt[p0 + "attn_q.weight"], Wt[p0 + "attn_k.weight"], Wt[p0 + "attn_v.weight"], Wt[p0 + "attn_output.weight"]
bq, bk, bv = Wt[p0 + "attn_q.bias"], Wt[p0 + "attn_k.bias"], Wt[p0 + "attn_v.bias"]
Wg, Wu, Wd = Wt[p0 + "ffn_gate.weight"], Wt[p0 + "ffn_up.weight"], Wt[p0 + "ffn_down.weight"]
D = {"word": word, "system": SYSTEM, "ids": ids, "pieces": [m.tok.decode([t]) for t in ids], "P": P, "T": T, "dims": {"dim": m.dim, "heads": m.n_heads, "kv": m.n_kv, "hd": hd, "ffn": int(Wg.shape[0])},
     "greedy": {"text": greedy, "tokens": n_g, "finished": done_g}, "samples": samples, "patched": list(m.patched)}
D["emb8"] = [R(a) for a in x[:8]]
D["norm1"] = norm(x, Wt[p0 + "attn_norm.weight"], h1)
D["mm_q"] = mm(Wq, h1, bq, q.reshape(-1)); D["mm_k"] = mm(Wk, h1, bk, k.reshape(-1)); D["mm_v"] = mm(Wv, h1, bv, v.reshape(-1))
# attention for the word's token: it can look at tokens 0..P
pat = cap["attn_pattern"][:, P, :P + 1]
heads = []
for h in range(m.n_heads):
    top = int(np.argmax(pat[h])); heads.append({"h": h, "top": top, "w": R(pat[h][top]), "self": R(pat[h][P])})
head = int(np.argmax([pat[h][1:].max() for h in range(m.n_heads)])); kv = head // rep
qh = m._rope(q[head][None, None, :].copy(), np.array([P], np.float32))[0, 0]
K = cache.k[0, kv, :P + 1]
sc = (K @ qh) / np.sqrt(hd); pr = np.exp(sc - sc.max()); pr /= pr.sum(); assert np.allclose(pr, pat[head], atol=2e-3)
jb = int(np.argmax(pr)); qk = qh.astype(np.float64) * K[jb].astype(np.float64)
Vc = cache.v[0, kv, :P + 1, 0].astype(np.float64); order = np.argsort(-pr)[:6]; cd = float(ctx[head * hd]); assert abs((pr * Vc).sum() - cd) < 3e-3 * max(1, abs(cd))
q0 = q[head]
D["att"] = {"heads": heads, "head": head, "scores": [R(a) for a in sc], "probs": [R(a) for a in pr], "n": P + 1,
            "rope": [{"i": i, "a": R(P * m.inv_freq[i]), "x1": R(q0[i]), "x2": R(q0[i + hd // 2]), "r1": R(qh[i]), "r2": R(qh[i + hd // 2])} for i in (0, 20)],
            "qk": {"j": jb, "terms": [[R(qh[i]), R(K[jb, i]), R(qk[i])] for i in range(6)], "rest": R(qk[6:].sum()), "n": hd, "score": R(sc[jb])},
            "mix": {"d": 0, "terms": [[int(j), R(pr[j]), R(Vc[j]), R(pr[j] * Vc[j])] for j in order], "rest": R((pr * Vc).sum() - sum(pr[j] * Vc[j] for j in order)), "c": R(cd)}}
D["mm_o"] = mm(Wo, ctx, np.zeros(4), ao)
D["res1"] = [[R(x[i]), R(ao[i]), R(xm[i])] for i in range(8)]
D["norm2"] = norm(xm, Wt[p0 + "ffn_norm.weight"], h2)
D["mm_g"] = mm(Wg, h2, np.zeros(4), g); D["mm_u"] = mm(Wu, h2, np.zeros(4), u)
top = np.argsort(-np.abs(act))[:3]
D["neurons"] = [{"n": int(i), "gate": R(g[i]), "up": R(u[i]), "act": R(act[i]), "wg": [R(a) for a in Wg[i, :8]], "wu": [R(a) for a in Wu[i, :8]], "rg": receipt(Wg[i], h2, g[i]), "ru": receipt(Wu[i], h2, u[i])} for i in top]
D["active"] = int((np.abs(act) >= 0.1 * np.abs(act).max()).sum())
D["mm_d"] = mm(Wd, act, np.zeros(4), mo)
D["res2"] = [[R(xm[i]), R(mo[i]), R(xo[i])] for i in range(8)]
json.dump(D, open(os.path.join(ROOT, "model", "frost_work", f"layer1_{word}.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("saved layer 1 data", flush=True)
