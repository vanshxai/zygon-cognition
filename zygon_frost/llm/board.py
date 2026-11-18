"""Record every number of one prompt's trip through the model and write capital_of_france_board.html (animated math board).

    python board.py ["What's the capital of France?"]
The board follows the LAST prompt token (the one that produces the first answer word) through all 36 layers with real weights and real
numbers; the rest of the answer is generated greedily and listed step by step.
"""
import base64, json, os, sys, time
import numpy as np
from engine import Model, chatml, rmsnorm

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
question = sys.argv[1] if len(sys.argv) > 1 else "What's the capital of France?"
SYSTEM = "You are Frost 1.0, a helpful assistant."
OUT = os.path.join(ROOT, "capital_of_france_board.html" if len(sys.argv) < 2 else "custom_board.html")


def R(x):
    return float(f"{float(x):.5g}")


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def enc(v):
    """float vector -> int8 + scale (clipped at the 99.7th percentile so outliers don't wash out the picture)"""
    v = np.asarray(v, np.float32).reshape(-1)
    sc = float(np.percentile(np.abs(v), 99.7)) or float(np.abs(v).max()) or 1.0
    return {"s": R(sc), "mx": R(np.abs(v).max()), "d": b64(np.clip(np.round(v / sc * 127), -127, 127).astype(np.int8))}


def receipt(w, x, out, bias=0.0, n=6):
    prod = w.astype(np.float64) * x.astype(np.float64)
    full = float(prod.sum()) + float(bias)
    assert abs(full - float(out)) <= 2e-3 * max(1.0, abs(float(out))), (full, float(out))        # the printed arithmetic really gives the layer's output
    return {"terms": [[R(w[i]), R(x[i]), R(prod[i])] for i in range(n)], "n": int(len(w)), "rest": R(prod[n:].sum()), "bias": R(bias), "out": R(out)}


m = Model(verbose=False)
ids = m.tok.encode(chatml([("system", SYSTEM), ("user", question)]))
T, NL, hd, rep = len(ids), m.n_layers, m.hd, m.n_heads // m.n_kv
print(f"{T} tokens: {ids}", flush=True)
cap = [dict() for _ in range(NL)]
VEC = ("resid_in", "attn_in", "q", "k", "v", "attn_ctx", "attn_out", "resid_mid", "ffn_in", "gate", "up", "act", "mlp_out", "resid_out")


def make_hook(store):
    def hk(l, pt, a):
        if pt == "attn_pattern":
            store[l][pt] = a[:, -1, :].copy()                                 # (heads, T): last token attends to every token
        elif pt in VEC:
            store[l][pt] = np.asarray(a[:, -1, :] if pt in ("q", "k", "v") else a[-1], np.float32).reshape(-1).copy()
    return hk


hook = make_hook(cap)


cache = m.new_cache(T + 40)
t0 = time.time()
logits = m.forward(ids, cache, hook)
print(f"traced {time.time() - t0:.1f}s", flush=True)
W = m.W
layers = []
for l in range(NL):
    c, p = cap[l], f"blk.{l}."
    x, h1, q, k, v, ctx, ao = c["resid_in"], c["attn_in"], c["q"], c["k"], c["v"], c["attn_ctx"], c["attn_out"]
    xm, h2, g, u, act, mo, xo = c["resid_mid"], c["ffn_in"], c["gate"], c["up"], c["act"], c["mlp_out"], c["resid_out"]
    L = {"l": l}
    ms = float((x.astype(np.float64) ** 2).mean()); rms = float(np.sqrt(ms + m.eps)); g1 = W[p + "attn_norm.weight"]
    assert np.allclose(x / rms * g1, h1, rtol=1e-3, atol=1e-4)
    L["norm1"] = {"ms": R(ms), "rms": R(rms), "eps": m.eps, "e": [[R(x[i]), R(g1[i]), R(h1[i])] for i in range(5)]}
    ms = float((xm.astype(np.float64) ** 2).mean()); rms = float(np.sqrt(ms + m.eps)); g2 = W[p + "ffn_norm.weight"]
    L["norm2"] = {"ms": R(ms), "rms": R(rms), "eps": m.eps, "e": [[R(xm[i]), R(g2[i]), R(h2[i])] for i in range(5)]}
    # projections: one output element each (the largest one) with its exact weight row
    Wq, Wk, Wv, Wo = W[p + "attn_q.weight"], W[p + "attn_k.weight"], W[p + "attn_v.weight"], W[p + "attn_output.weight"]
    rq, rk, rv, ro = int(np.argmax(np.abs(q))), int(np.argmax(np.abs(k))), int(np.argmax(np.abs(v))), int(np.argmax(np.abs(ao)))
    L["q"] = {"row": rq, "rc": receipt(Wq[rq], h1, q[rq], W[p + "attn_q.bias"][rq]), "w": [R(a) for a in Wq[rq, :8]], "blk": [[R(a) for a in Wq[r, :8]] for r in range(max(rq - 2, 0), max(rq - 2, 0) + 5)], "blk0": max(rq - 2, 0)}
    L["k"] = {"row": rk, "rc": receipt(Wk[rk], h1, k[rk], W[p + "attn_k.bias"][rk]), "w": [R(a) for a in Wk[rk, :8]]}
    L["v"] = {"row": rv, "rc": receipt(Wv[rv], h1, v[rv], W[p + "attn_v.bias"][rv]), "w": [R(a) for a in Wv[rv, :8]]}
    L["o"] = {"row": ro, "rc": receipt(Wo[ro], ctx, ao[ro]), "w": [R(a) for a in Wo[ro, :8]]}
    L["v8"] = {nm: [R(a) for a in vec[:8]] for nm, vec in (("x", x), ("h1", h1), ("q", q), ("k", k), ("v", v), ("ctx", ctx), ("ao", ao), ("xm", xm), ("h2", h2), ("g", g), ("u", u), ("act", act), ("mo", mo), ("xo", xo))}
    for key, Wm, hv, bias, ov in (("mm", Wq, h1, W[p + "attn_q.bias"], q), ("mmg", W[p + "ffn_gate.weight"], h2, np.zeros(4, np.float32), g)):
        prod = Wm[:4].astype(np.float64) * hv[None, :].astype(np.float64)
        for r in range(4):
            assert abs(prod[r].sum() + float(bias[r]) - float(ov[r])) <= 2e-3 * max(1.0, abs(float(ov[r])))
        L[key] = {"W": [[R(a) for a in Wm[r, :8]] for r in range(4)], "h": [R(a) for a in hv[:8]], "part": [R(prod[r, :8].sum()) for r in range(4)], "rest": [R(prod[r, 8:].sum()) for r in range(4)],
                  "b": [R(bias[r]) for r in range(4)], "out": [R(ov[r]) for r in range(4)]}
    L["vq"], L["vk"], L["vv"], L["vctx"], L["vao"] = enc(q), enc(k), enc(v), enc(ctx), enc(ao)
    L["vx"], L["vh1"], L["vxm"], L["vh2"], L["vxo"], L["vmo"] = enc(x), enc(h1), enc(xm), enc(h2), enc(xo), enc(mo)
    # attention: pick the head that looks hardest at a token other than the first one
    pat = c["attn_pattern"]
    head = int(np.argmax(pat[:, 1:].max(1))); kv = head // rep
    qh = m._rope(q.reshape(m.n_heads, 1, hd)[head:head + 1], np.array([T - 1], np.float32))[0, 0]
    K = cache.k[l, kv, :T]
    sc = (K @ qh) / np.sqrt(hd)
    pr = np.exp(sc - sc.max()); pr /= pr.sum()
    assert np.allclose(pr, pat[head], atol=2e-3)
    jb = int(np.argmax(pat[head])); qk = qh.astype(np.float64) * K[jb].astype(np.float64)
    assert abs(qk.sum() / np.sqrt(hd) - sc[jb]) < 2e-3 * max(1, abs(sc[jb]))
    d0 = 0; vv = cache.v[l, kv, :T, d0].astype(np.float64); cd = float(ctx[head * hd + d0]); order = np.argsort(-pr)[:6]
    assert abs((pr * vv).sum() - cd) < 3e-3 * max(1, abs(cd))
    q0 = q.reshape(m.n_heads, hd)[head]
    ang = [float((T - 1) * m.inv_freq[i]) for i in (0, 20)]
    L["att"] = {"head": head, "kv": kv, "scores": [R(a) for a in sc], "probs": [R(a) for a in pr],
                "rope": [{"i": i, "a": R(a), "x1": R(q0[i]), "x2": R(q0[i + hd // 2]), "r1": R(qh[i]), "r2": R(qh[i + hd // 2])} for i, a in zip((0, 20), ang)],
                "qk": {"j": jb, "terms": [[R(qh[i]), R(K[jb, i]), R(qk[i])] for i in range(6)], "rest": R(qk[6:].sum()), "n": hd, "score": R(sc[jb])},
                "mix": {"d": d0, "terms": [[int(j), R(pr[j]), R(vv[j]), R(pr[j] * vv[j])] for j in order], "rest": R((pr * vv).sum() - sum(pr[j] * vv[j] for j in order)), "c": R(cd)},
                "pat": b64(np.clip(np.round(pat * 255), 0, 255).astype(np.uint8))}
    # feed-forward: the three most active neurons with their exact weights
    top = np.argsort(-np.abs(act))[:3]
    Wg, Wu, Wd = W[p + "ffn_gate.weight"], W[p + "ffn_up.weight"], W[p + "ffn_down.weight"]
    L["neurons"] = [{"n": int(n), "gate": R(g[n]), "up": R(u[n]), "act": R(act[n]), "wg": [R(a) for a in Wg[n, :8]], "wu": [R(a) for a in Wu[n, :8]],
                     "rg": receipt(Wg[n], h2, g[n]), "ru": receipt(Wu[n], h2, u[n]), "down": [R(a) for a in Wd[:5, n]]} for n in top]
    L["res1"] = [[R(x[i]), R(ao[i]), R(xm[i])] for i in range(5)]
    L["res2"] = [[R(xm[i]), R(mo[i]), R(xo[i])] for i in range(5)]
    L["vact"] = enc(act)
    rd = int(np.argmax(np.abs(mo)))
    L["down"] = {"row": rd, "rc": receipt(Wd[rd], act, mo[rd]), "w": [R(a) for a in Wd[rd, :8]], "active": int((np.abs(act) >= 0.1 * np.abs(act).max()).sum())}
    # logit lens: what would the model answer if it stopped here?
    lg = W["output.weight"] @ rmsnorm(xo, W["output_norm.weight"], m.eps)
    pe = np.exp(lg - lg.max()); pe /= pe.sum(); tk = np.argsort(-lg)[:5]
    L["lens"] = [[m.tok.decode([int(t)]), R(pe[t])] for t in tk]
    layers.append(L)
    if l % 6 == 0: print("  layer", l, flush=True)

# ---- the end of the trip: final norm, logits, softmax, choice
xo = cap[NL - 1]["resid_out"]
ms = float((xo.astype(np.float64) ** 2).mean()); rms = float(np.sqrt(ms + m.eps)); gn = W["output_norm.weight"]
hn = rmsnorm(xo, gn, m.eps)
top10 = np.argsort(-logits)[:10]
pr = np.exp(logits - logits.max()); pr /= pr.sum()
first = int(np.argmax(logits))
E = W["output.weight"]
final = {"ms": R(ms), "rms": R(rms), "e": [[R(xo[i]), R(gn[i]), R(hn[i])] for i in range(5)], "vh": enc(hn), "vx": enc(xo),
         "top": [{"id": int(t), "piece": m.tok.decode([int(t)]), "logit": R(logits[t]), "p": R(pr[t]), "w": [R(a) for a in E[t, :8]]} for t in top10],
         "rc": receipt(E[first], hn, logits[first]), "first": first, "vocab": int(E.shape[0]), "max_logit": R(logits.max()),
         "sum_exp": R(np.exp(logits - logits.max()).sum()), "vl": enc(logits[:2048])}
# ---- the winning path: after the final norm the winner's score is a plain sum over the residual stream's parts
def attribute(cp, w, r, lw, lr):
    gn = W["output_norm.weight"].astype(np.float64); xo = cp[NL - 1]["resid_out"].astype(np.float64)
    rms = float(np.sqrt((xo ** 2).mean() + m.eps)); Ew, Er = E[w].astype(np.float64), E[r].astype(np.float64)
    wv, rv = Ew * gn / rms, Er * gn / rms
    x0 = cp[0]["resid_in"].astype(np.float64)
    names, vecs = ["embedding"], [x0]
    for l in range(NL):
        names += [f"attn {l}", f"mlp {l}"]; vecs += [cp[l]["attn_out"].astype(np.float64), cp[l]["mlp_out"].astype(np.float64)]
    cw = np.array([wv @ v for v in vecs]); cr = np.array([rv @ v for v in vecs])
    assert abs(cw.sum() - lw) <= 5e-3 * max(1, abs(lw)), (cw.sum(), lw)             # the parts really add up to the model's own score
    heads, neus = [], []
    for l in range(NL):
        p_ = f"blk.{l}."; Wo, Wd = W[p_ + "attn_output.weight"], W[p_ + "ffn_down.weight"]
        u = (Wo.T @ wv.astype(np.float32)).astype(np.float64); ctx_ = cp[l]["attn_ctx"].astype(np.float64); pat_ = cp[l]["attn_pattern"]
        ch = np.array([u[h * hd:(h + 1) * hd] @ ctx_[h * hd:(h + 1) * hd] for h in range(m.n_heads)])
        assert abs(ch.sum() - cw[1 + 2 * l]) <= 5e-3 * max(1, abs(cw[1 + 2 * l])) + 1e-3
        for h in np.argsort(-ch)[:3]:
            j = int(np.argmax(pat_[h])); heads.append({"l": l, "h": int(h), "c": R(ch[h]), "j": j, "pj": R(pat_[h][j])})
        u2 = (Wd.T @ wv.astype(np.float32)).astype(np.float64); cn = cp[l]["act"].astype(np.float64) * u2
        for n_ in np.argsort(-cn)[:3]:
            neus.append({"l": l, "n": int(n_), "act": R(cp[l]["act"][n_]), "c": R(cn[n_])})
    heads.sort(key=lambda d: -d["c"]); neus.sort(key=lambda d: -d["c"])
    terms = wv * xo; ist = int(np.argmax(terms)); order = np.argsort(-terms)[:6]
    evo = [R(x0[ist])]
    for l in range(NL):
        evo += [R(cp[l]["resid_mid"][ist]), R(cp[l]["resid_out"][ist])]
    return {"rms": R(rms), "gn_mean": R(gn.mean()), "comp_names": names, "cw": [R(v) for v in cw], "cr": [R(v) for v in cr], "dim": ist,
            "top_terms": [[int(i), R(Ew[i]), R(gn[i]), R(xo[i]), R(terms[i])] for i in order], "rest_terms": R(terms.sum() - terms[order].sum()), "evo": evo,
            "heads": heads[:8], "neurons": neus[:10], "logit_w": R(lw), "logit_r": R(lr), "runner": int(r), "runner_piece": m.tok.decode([int(r)])}


gen, end = [], m.tok.id_of["<|im_end|>"]
lg_now, cp_now, tok_in = logits, cap, ids[-1]
for step in range(24):
    p_ = np.exp(lg_now - lg_now.max()); p_ /= p_.sum(); t5 = np.argsort(-lg_now)[:5]
    t = int(np.argmax(lg_now))
    g = {"id": t, "piece": m.tok.decode([t]), "p": R(p_[t]), "top5": [[int(a), m.tok.decode([int(a)]), R(p_[a])] for a in t5], "pos": T + step, "input_id": int(tok_in), "input_piece": m.tok.decode([int(tok_in)]),
         "path": attribute(cp_now, t, int(t5[1]), float(lg_now[t]), float(lg_now[t5[1]]))}
    g["path"]["emb_w"] = [R(v) for v in E[t, :8]]
    gen.append(g)
    print("  path", step, repr(g["piece"]), flush=True)
    if t == end or t == m.tok.eos: break
    store = [dict() for _ in range(NL)]
    lg_now = m.forward([t], cache, make_hook(store)); cp_now = store; tok_in = t
answer_ids = [g["id"] for g in gen if g["id"] not in (end, m.tok.eos)]
print("answer:", m.tok.decode(answer_ids), flush=True)
pieces = [m.tok.decode([t]) for t in ids]
E0 = W["token_embd.weight"]
D = {"question": question, "system": SYSTEM, "template": chatml([("system", SYSTEM), ("user", question)]), "T": T, "ids": ids, "pieces": pieces,
     "bytes": [list(m.tok.token_bytes(t)) for t in ids], "emb": [[R(a) for a in E0[t, :8]] for t in ids], "emb_norm": [R(np.linalg.norm(E0[t])) for t in ids],
     "dims": {"layers": NL, "dim": m.dim, "heads": m.n_heads, "kv": m.n_kv, "hd": hd, "ffn": int(W["blk.0.ffn_up.weight"].shape[0]), "vocab": int(E.shape[0]), "eps": m.eps},
     "layers": layers, "final": final, "gen": gen, "answer": m.tok.decode(answer_ids), "answer_ids": answer_ids,
     "weights_note": "patched: " + ", ".join(m.patched) if m.patched else "original weights"}
DATA = os.path.join(ROOT, "model", "frost_work", "board_data.json")
json.dump(D, open(DATA, "w", encoding="utf-8"), ensure_ascii=False)
print("data saved", DATA, flush=True)
tpl = open(os.path.join(HERE, "board_template.html"), encoding="utf-8").read()
open(OUT, "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/null", json.dumps(D, ensure_ascii=False).replace("</", "<" + chr(92) + "/")))
print("written", OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB", flush=True)
