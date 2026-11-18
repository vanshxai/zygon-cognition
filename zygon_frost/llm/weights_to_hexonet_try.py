"""Just a try: can the 3B model's numbers be turned into exponents, and do its vector angles line up with our angles?
    python weights_to_hexonet_try.py   ->  model/weights_try.json
A  exponent form: every weight w = sign * e^x.  How many bits does x need, against the 4.5 bits per weight the model already uses?
B  vector angles: do tokens that are close in the model's embedding space also get close angles from our identity angles (theta_k = 360 * frac(k * ln(id+1)))?
   And how many angles taken from the embedding itself (random 2D projections) does it take to keep a token's 10 nearest neighbours?"""
import json, math, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf
from tokenizer import Tokenizer
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
g = gguf.GGUF(os.path.join(ROOT, "model", "original", "qwen2.5-3b-instruct-q4_k_m.original.gguf")); rng = np.random.default_rng(5); out = {}
t0 = time.perf_counter()

# ---------------------------------------------------------------- A
A = {}
for name in ("blk.10.ffn_down.weight", "blk.20.attn_q.weight", "token_embd.weight"):
    W = gguf.np_tensor(g, name).astype(np.float64).ravel(); n = len(W); samp = W[rng.choice(n, min(n, 3_000_000), replace=False)]
    zero = float(np.mean(samp == 0)); nz = samp[samp != 0]; x = np.log(np.abs(nz)); lo, hi = float(x.min()), float(x.max()); rows = {}
    for bits in (6, 8, 10, 12):
        step = (hi - lo) / (2 ** bits - 1); xq = np.round((x - lo) / step) * step + lo; rec = np.sign(nz) * np.exp(xq); rel = np.abs(rec - nz) / np.abs(nz)
        rows[bits] = {"bits_per_weight_with_sign": bits + 1, "median_rel_error": float(np.median(rel)), "p99_rel_error": float(np.percentile(rel, 99)), "max_rel_error": float(rel.max())}
    A[name] = {"weights": n, "zero_fraction": zero, "x_range": [lo, hi], "by_bits": rows}
    print("A", name, {b: (round(r["median_rel_error"], 5), round(r["p99_rel_error"], 4)) for b, r in rows.items()}, "zeros", round(zero, 4), flush=True)
out["A"] = {"model_bits_per_weight": 4.5, "tensors": A}

# ---------------------------------------------------------------- B
E = gguf.np_tensor(g, "token_embd.weight").astype(np.float32); V = 20000; E = E[:V]; E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
Q = rng.choice(np.arange(500, V), 400, replace=False); S = E[Q] @ E.T; S[np.arange(len(Q)), Q] = -9; truth = np.argpartition(-S, 10, axis=1)[:, :10]


def recall(sim):                                         # sim: (len(Q), V) similarity from the angle scheme; recall of the true 10 neighbours inside its own top 10
    sim = sim.copy(); sim[np.arange(len(Q)), Q] = -9; top = np.argpartition(-sim, 10, axis=1)[:, :10]
    return float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top, truth)]))


ids = np.arange(V)
ours = {}
for H in (1, 3, 8):                                       # our identity angles: theta_k = 360 * frac(k * ln(id + 1))
    th = np.stack([2 * np.pi * (((k + 1) * np.log(ids + 1)) % 1.0) for k in range(H)], 1)
    sim = np.zeros((len(Q), V))
    for k in range(H): sim += np.cos(th[Q][:, [k]] - th[:, k][None, :])
    ours[H] = recall(sim / H)
proj = {}
for m in (1, 3, 8, 16, 32, 64, 128):                       # angles taken from the embedding itself: atan2 of two random directions
    U = rng.standard_normal((E.shape[1], m)).astype(np.float32); Vv = rng.standard_normal((E.shape[1], m)).astype(np.float32)
    th = np.arctan2(E @ U, E @ Vv); sim = np.zeros((len(Q), V), dtype=np.float32)
    for k in range(m): sim += np.cos(th[Q][:, [k]] - th[:, k][None, :])
    proj[m] = recall(sim / m)
chance = 10 / V
print("B recall@10  ours:", {k: round(v, 4) for k, v in ours.items()}, "| from embedding:", {k: round(v, 3) for k, v in proj.items()}, "| chance", round(chance, 5), flush=True)
out["B"] = {"tokens": V, "queries": len(Q), "chance": chance, "our_identity_angles_recall_at_10": ours, "embedding_projection_angles_recall_at_10": proj}

tok = Tokenizer(g.meta); ex = {}
for w in (" heating", " pipe", " duct", " cooling", " pressure"):
    t = tok.encode(w)
    if len(t) == 1 and t[0] < V:
        s = E @ E[t[0]]; s[t[0]] = -9; ex[w.strip()] = [tok.decode([int(i)]).strip() for i in np.argsort(-s)[:8]]
out["B"]["nearest_tokens_in_the_model"] = ex; print("B examples", ex)
out["seconds"] = round(time.perf_counter() - t0, 1)
json.dump(out, open(os.path.join(ROOT, "model", "weights_try.json"), "w", encoding="utf-8"), indent=1)
print("done", out["seconds"], "s")
