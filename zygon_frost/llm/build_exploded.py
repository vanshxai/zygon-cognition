"""Builds the exploded view of the 3B model from its REAL weights.   python build_exploded.py  ->  llm/model_exploded.html (self-contained)
For one block at a time (layers 0, 10, 20, 35) every weight matrix gets: its shape, parameter count, bits, statistics, a histogram, an overview picture (average size of the weights, pooled)
and a raw patch (a real 40 x 40 corner of signed numbers)."""
import base64, io, json, os, sys, time
import numpy as np
from PIL import Image
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
g = gguf.GGUF(os.path.join(ROOT, "model", "original", "qwen2.5-3b-instruct-q4_k_m.original.gguf")); meta = g.meta
LAYERS = tuple(range(36)); rng = np.random.default_rng(3)
PARTS = [("attn_q", "Query matrix", "What each word is looking for. 16 heads of 128 numbers."), ("attn_k", "Key matrix", "What each word offers to be found by. Only 2 key groups, shared by all 16 heads."),
         ("attn_v", "Value matrix", "What a word hands over once it is picked. 2 groups, like the keys."), ("attn_output", "Output matrix", "Mixes the 16 heads' findings back into one 2048-number vector."),
         ("ffn_gate", "Gate matrix", "First of two parallel widening steps, 2048 to 11008. Its result decides how much of the other one passes."), ("ffn_up", "Up matrix", "The second widening step, 2048 to 11008. This is the content that the gate lets through."),
         ("ffn_down", "Down matrix", "Squeezes the 11008 numbers back to 2048. Most of the model's storage sits in these three feed-forward matrices.")]


def png(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG", optimize=True); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def pool(M, rows, cols):
    r = np.linspace(0, M.shape[0], rows + 1).astype(int); c = np.linspace(0, M.shape[1], cols + 1).astype(int)
    A = np.add.reduceat(np.add.reduceat(np.abs(M), r[:-1], axis=0), c[:-1], axis=1); n = np.outer(np.diff(r), np.diff(c)); return A / n


def seq_color(A):                                   # dark blue -> teal -> yellow by size of weight
    lo, hi = np.percentile(A, 1), np.percentile(A, 99.5); t = np.clip((A - lo) / (hi - lo + 1e-12), 0, 1)
    stops = np.array([[12, 20, 60], [10, 108, 128], [250, 220, 90]], dtype=np.float32); x = t * 2; i = np.minimum(x.astype(int), 1); f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def div_color(A, s):                                # blue - white - red by signed value in units of std
    t = np.clip(A / (3 * s), -1, 1)[..., None]; w = np.array([245, 245, 245], dtype=np.float32); red = np.array([200, 50, 40], dtype=np.float32); blue = np.array([30, 90, 190], dtype=np.float32)
    return np.where(t >= 0, w * (1 - t) + red * t, w * (1 + t) + blue * (-t)).astype(np.uint8)


def describe(name, W, qtype):
    rows, cols = W.shape; n = W.size; samp = W.ravel()[rng.choice(n, min(n, 2_000_000), replace=False)]; s = float(samp.std()); h, e = np.histogram(samp, bins=61, range=(-4 * s, 4 * s))
    asp = rows / cols; R = 72 if asp <= 1 else int(min(72 * asp, 400)); C = 72 if asp >= 1 else int(max(72 * asp, 10))
    return {"shape": [rows, cols], "params": int(n), "type": qtype, "mean": float(samp.mean()), "std": s, "min": float(samp.min()), "max": float(samp.max()), "near_zero_pct": float(np.mean(np.abs(samp) < 0.1 * s) * 100),
            "hist": h.tolist(), "hist_range": [-4 * s, 4 * s], "overview": png(seq_color(pool(W, R, C))), "patch": png(div_color(W[:40, :40], s)), "patch_note": "a real 40 x 40 corner of the matrix"}


t0 = time.perf_counter(); out = {"model": {"name": "Qwen2.5-3B-Instruct", "file": "qwen2.5-3b-instruct-q4_k_m.gguf", "layers": meta["qwen2.block_count"], "width": meta["qwen2.embedding_length"], "heads": meta["qwen2.attention.head_count"],
           "kv_heads": meta["qwen2.attention.head_count_kv"], "ffn": meta["qwen2.feed_forward_length"], "context": meta["qwen2.context_length"], "license": meta.get("general.license.name"), "vocab": 151936}, "layers": {}, "parts": [[a, b, c] for a, b, c in PARTS]}
tot = {"embedding": 0, "attention": 0, "feed_forward": 0, "norms_biases": 0}
for n, t in g.tensors.items():
    k = "embedding" if n == "token_embd.weight" else "attention" if ".attn_" in n and n.endswith("weight") and "norm" not in n else "feed_forward" if ".ffn_" in n and "norm" not in n else "norms_biases"
    tot[k] += t.n_elements
out["share"] = tot; out["total_params"] = sum(tot.values())
for L in LAYERS:
    d = {}
    for key, _, _ in PARTS:
        nm = f"blk.{L}.{key}.weight"; W = gguf.np_tensor(g, nm).astype(np.float32); d[key] = describe(key, W, g.tensors[nm].type_name); print("layer", L, key, W.shape, flush=True)
    for key in ("attn_norm", "ffn_norm", "attn_q.bias", "attn_k.bias", "attn_v.bias"):
        nm = f"blk.{L}.{key}" + ("" if key.endswith("bias") else ".weight"); V = gguf.np_tensor(g, nm).astype(np.float32) if g.tensors[nm].type_name != "F32" else np.frombuffer(g.raw(nm), dtype=np.float32)
        d[key] = {"n": int(V.size), "mean": float(V.mean()), "std": float(V.std()), "min": float(V.min()), "max": float(V.max()), "spark": [float(x) for x in np.interp(np.linspace(0, V.size - 1, 120), np.arange(V.size), V)]}
    out["layers"][str(L)] = d
# how the same matrix changes down the stack: std of every layer's Q and FFN down (cheap: sample 200k)
stack = {}
for key in ("attn_q", "ffn_down"):
    stack[key] = []
    for L in range(meta["qwen2.block_count"]):
        W = gguf.np_tensor(g, f"blk.{L}.{key}.weight"); stack[key].append(float(W.ravel()[rng.choice(W.size, 200_000, replace=False)].std()))
out["stack_std"] = stack; out["seconds"] = round(time.perf_counter() - t0, 1)
json.dump(out, open(os.path.join(ROOT, "model", "exploded_data.json"), "w"))
html = open(os.path.join(HERE, "model_exploded_template.html"), encoding="utf-8").read().replace("/*DATA*/null", json.dumps(out)).replace("/*TRACE*/null", open(os.path.join(ROOT, "model", "trace_prompts.json")).read() if os.path.exists(os.path.join(ROOT, "model", "trace_prompts.json")) else "null")
open(os.path.join(HERE, "model_exploded.html"), "w", encoding="utf-8").write(html); print("done", out["seconds"], "s", round(len(html) / 1e6, 2), "MB; total params", out["total_params"], out["share"])
