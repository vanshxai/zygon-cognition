"""Run one prompt through the model, record every intermediate matrix, write a self-contained viz.html.

    python trace.py "What's the capital of France?"
"""
import sys, json, base64, time
import numpy as np
from engine import Model, chatml, rmsnorm

sys.stdout.reconfigure(encoding="utf-8")
question = sys.argv[1] if len(sys.argv) > 1 else "What's the capital of France?"
SYSTEM = "You are Frost 1.0, a helpful assistant."

m = Model()
ids = m.tok.encode(chatml([("system", SYSTEM), ("user", question)]))
T = len(ids)
print(f"{T} tokens: {ids}")

VEC = ("resid_in", "attn_in", "q", "k", "v", "attn_ctx", "attn_out", "resid_mid",
       "ffn_in", "gate", "up", "act", "mlp_out", "resid_out")
cap = [dict() for _ in range(m.n_layers)]


def hook(l, pt, a):
    if pt == "attn_pattern":
        cap[l][pt] = a.copy()                                   # (heads, T, T)
    elif pt in VEC:
        v = a[:, -1, :] if pt in ("q", "k", "v") else a[-1]    # last token only
        cap[l][pt] = np.asarray(v, np.float32).reshape(-1).copy()


t0 = time.time()
logits = m.forward(ids, m.new_cache(), hook)
print(f"traced in {time.time() - t0:.1f}s")


def b64(arr):
    return base64.b64encode(arr.tobytes()).decode()


def enc(v):
    """float vector -> int8 (clipped at the 99.7th percentile so outliers don't wash out the picture)."""
    mx = float(np.abs(v).max())
    sc = float(np.percentile(np.abs(v), 99.7)) or mx or 1.0
    q = np.clip(np.round(v / sc * 127), -127, 127).astype(np.int8)
    return {"s": sc, "mx": mx, "d": b64(q)}


def receipt(w, x, out_val, bias=0.0, nshow=6):
    prod = w.astype(np.float64) * x.astype(np.float64)
    return {"terms": [[float(w[i]), float(x[i]), float(prod[i])] for i in range(nshow)], "n": int(len(w)),
            "shown_sum": float(prod[:nshow].sum()), "full_sum": float(prod.sum()), "bias": float(bias),
            "out": float(out_val)}


layers = []
for l in range(m.n_layers):
    c, p = cap[l], f"blk.{l}."
    L = {k: enc(c[k]) for k in VEC}
    pat = c["attn_pattern"]
    L["pat"] = b64(np.clip(np.round(pat * 255), 0, 255).astype(np.uint8))
    n = int(np.argmax(np.abs(c["act"])))                       # the most active FFN neuron at the last token
    L["neuron"] = n
    Wq = m.W[p + "attn_q.weight"]
    L["rc_q"] = receipt(Wq[0], c["attn_in"], c["q"][0], m.W[p + "attn_q.bias"][0])
    L["rc_gate"] = receipt(m.W[p + "ffn_gate.weight"][n], c["ffn_in"], c["gate"][n])
    L["rc_down"] = receipt(m.W[p + "ffn_down.weight"][0], c["act"], c["mlp_out"][0])
    L["corner"] = {"W": Wq[:6, :6].tolist(), "x": c["attn_in"][:6].tolist()}
    L["norm_g"] = m.W[p + "attn_norm.weight"][:6].tolist()
    top = np.argsort(-np.abs(c["act"]))[:6]
    L["top_neurons"] = [[int(i), float(c["gate"][i]), float(c["up"][i]), float(c["act"][i])] for i in top]
    lg = m.W["output.weight"] @ rmsnorm(c["resid_out"], m.W["output_norm.weight"], m.eps)
    pr = np.exp(lg - lg.max()); pr /= pr.sum()
    tk = np.argsort(-lg)[:5]
    L["lens"] = [[m.tok.decode([int(t)]), float(pr[t])] for t in tk]
    layers.append(L)

pr = np.exp(logits - logits.max()); pr /= pr.sum()
top = np.argsort(-logits)[:10]
final = [[m.tok.decode([int(t)]), float(logits[t]), float(pr[t])] for t in top]

print("generating the full answer (greedy)...")
answer = "".join(p for _, p in m.generate(ids, max_new=60, temp=0))
print("answer:", answer)

D, F = m.dim, m.W["blk.0.ffn_up.weight"].shape[0]
macs = {"q": D * D, "k": D * 256, "v": D * 256, "o": D * D, "gate": D * F, "up": D * F, "down": F * D}
data = {
    "question": question, "answer": answer, "T": T, "ids": ids,
    "tokens": [m.tok.decode([i]) for i in ids],
    "emb": enc(m.W["token_embd.weight"][ids[-1]]),
    "layers": layers, "final": final, "dim": D, "ffn": F, "heads": m.n_heads, "nlayers": m.n_layers,
    "macs": macs, "macs_layer": sum(macs.values()), "macs_head": 151936 * D,
}
html = open("viz_template.html", encoding="utf-8").read()
blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
out = html.replace("/*__DATA__*/null", blob)
open("viz.html", "w", encoding="utf-8").write(out)
print(f"wrote viz.html ({len(out) / 1e6:.1f} MB)")
