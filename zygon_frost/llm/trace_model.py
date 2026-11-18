"""Runs real prompts through the 3B model and records which parts of every block did how much work.   python trace_model.py  ->  model/trace_prompts.json
Same maths as engine.py (Qwen2 forward pass in numpy), but the weights are held in half precision and widened one layer at a time, so it fits in about 7 GB.
Per pass (the prompt goes in as ONE pass, then each new word is one more pass) and per layer it keeps: how big the query/key/value numbers were, how hard each of the 16 heads worked and where it looked,
how many of the 11,008 feed-forward neurons had their gate open, the strongest neurons, how much the block changed the running row, and the 5 best next words after the last layer."""
import json, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUF, np_tensor
from tokenizer import Tokenizer
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
PROMPTS = ["hi", "what is hvac"]; MAX_NEW = {"hi": 40, "what is hvac": 40}
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def rms(x, w, eps): return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w
def silu(x): return x / (1.0 + np.exp(-x))


class Tracer:
    def __init__(self):
        g = GGUF(os.path.join(ROOT, "model", "original", "qwen2.5-3b-instruct-q4_k_m.original.gguf")); m = g.meta; self.g = g
        self.L, self.dim, self.nh, self.nkv = m["qwen2.block_count"], m["qwen2.embedding_length"], m["qwen2.attention.head_count"], m["qwen2.attention.head_count_kv"]
        self.hd = self.dim // self.nh; self.eps = m["qwen2.attention.layer_norm_rms_epsilon"]; self.tok = Tokenizer(m); self.W = {}; t0 = time.time()
        for n in g.tensors:
            a = np_tensor(g, n); self.W[n] = a.astype(np.float16) if a.ndim == 2 else a.astype(np.float32)
        self.inv = (1.0 / m["qwen2.rope.freq_base"] ** (np.arange(self.hd // 2) * 2.0 / self.hd)).astype(np.float32); print("loaded in", round(time.time() - t0), "s", flush=True)
        self.eos = {self.tok.id_of["<|im_end|>"], self.tok.eos}

    def w(self, n): a = self.W[n]; return a.astype(np.float32) if a.dtype == np.float16 else a

    def rope(self, x, pos):
        ang = pos[:, None] * self.inv[None, :]; c, s = np.cos(ang), np.sin(ang); h = self.hd // 2; x1, x2 = x[..., :h], x[..., h:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    def forward(self, ids, kc, vc, p0):
        """one pass; returns logits of the last position and the per-layer record"""
        T = len(ids); pos = np.arange(p0, p0 + T, dtype=np.float32); x = self.W["token_embd.weight"][ids].astype(np.float32); rep = self.nh // self.nkv; recs = []; store = []
        for l in range(self.L):
            p = f"blk.{l}."; xin = x; h = rms(x, self.w(p + "attn_norm.weight"), self.eps)
            q = (h @ self.w(p + "attn_q.weight").T + self.W[p + "attn_q.bias"]).reshape(T, self.nh, self.hd).transpose(1, 0, 2)
            k = (h @ self.w(p + "attn_k.weight").T + self.W[p + "attn_k.bias"]).reshape(T, self.nkv, self.hd).transpose(1, 0, 2)
            v = (h @ self.w(p + "attn_v.weight").T + self.W[p + "attn_v.bias"]).reshape(T, self.nkv, self.hd).transpose(1, 0, 2)
            r = {"q": float(np.abs(q).mean()), "k": float(np.abs(k).mean()), "v": float(np.abs(v).mean())}
            q, k = self.rope(q, pos), self.rope(k, pos); kc[l, :, p0:p0 + T], vc[l, :, p0:p0 + T] = k, v
            K, V = np.repeat(kc[l, :, :p0 + T], rep, 0), np.repeat(vc[l, :, :p0 + T], rep, 0)
            s = (q @ K.transpose(0, 2, 1)) / np.float32(np.sqrt(self.hd))
            if T > 1: s = s + np.triu(np.full((T, p0 + T), -np.inf, np.float32), k=p0 + 1)
            s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True); last = s[:, -1, :]
            ent = -(last * np.log(last + 1e-12)).sum(-1); top = last.argmax(-1); topw = last.max(-1)
            ctx = (s @ V).transpose(1, 0, 2); r["head_work"] = np.linalg.norm(ctx, axis=-1).mean(0).round(3).tolist()                 # how hard each of the 16 heads worked (size of what it produced)
            r["head_look"] = [[round(float(ent[i]), 3), int(top[i]), round(float(topw[i]), 3)] for i in range(self.nh)]               # entropy (spread of attention), position it looked at most, how much
            ctxl = ctx[-1].copy(); a = ctx.reshape(T, self.dim) @ self.w(p + "attn_output.weight").T; x = x + a
            r["attn_delta"] = float(np.linalg.norm(a, axis=-1).mean() / np.linalg.norm(xin, axis=-1).mean()); xmid = x
            h = rms(x, self.w(p + "ffn_norm.weight"), self.eps); gt = h @ self.w(p + "ffn_gate.weight").T; up = h @ self.w(p + "ffn_up.weight").T
            hl = h[-1].copy(); sg = silu(gt); act = sg * up; r["gate_open"] = float((sg > 0.1).mean()); r["up_abs"] = float(np.abs(up).mean()); r["gate_abs"] = float(np.abs(sg).mean())
            la = np.abs(act[-1]); idx = np.argsort(-la)[:8]; r["top_neurons"] = [[int(i), round(float(act[-1][i]), 3)] for i in idx]
            srt = np.sort(act[-1] ** 2)[::-1]; r["top1pct_energy"] = float(srt[:110].sum() / (srt.sum() + 1e-12))
            store.append((ctxl, hl, act[-1].copy())); f = act @ self.w(p + "ffn_down.weight").T; x = x + f; r["ffn_delta"] = float(np.linalg.norm(f, axis=-1).mean() / np.linalg.norm(xmid, axis=-1).mean()); recs.append(r)
        kp = getattr(self, "keep", 1); xl = rms(x[-1] if kp == 1 else x[-kp:], self.w("output_norm.weight"), self.eps); logits = self.w("token_embd.weight") @ (xl if kp == 1 else xl.T)
        return logits, recs, self.paths(store, x[-1], logits)

    def paths(self, store, xf, logits):
        """which exact cells pushed the chosen word. u = the output direction of that word; a feed-forward neuron n of block l adds act[n] * down[:, n] to the running row, so its direct push on the
        word's score is act[n] * (u . down[:, n]); a head h adds O[:, head h columns] @ (what the head produced). For the strongest neurons the cells that fed it and the cells it wrote with are listed."""
        t = int(np.argmax(logits)); rmsf = float(np.sqrt((xf * xf).mean() + self.eps)); u = (self.W["token_embd.weight"][t].astype(np.float32) * self.W["output_norm.weight"] / rmsf).astype(np.float32)
        layers, cand, direct = [], [], 0.0
        for l in range(self.L):
            p = f"blk.{l}."; ctxl, hl, actl = store[l]; O = self.w(p + "attn_output.weight"); D = self.w(p + "ffn_down.weight")
            dh = [float(u @ (O[:, h * self.hd:(h + 1) * self.hd] @ ctxl[h])) for h in range(self.nh)]
            wn = u @ D; dn = actl * wn; direct += sum(dh) + float(dn.sum()); top = np.argsort(-np.abs(dn))[:6]; cells = []
            for n in top[:3]:
                n = int(n); g = self.W[p + "ffn_gate.weight"][n].astype(np.float32); up = self.W[p + "ffn_up.weight"][n].astype(np.float32); dc = D[:, n]; ci = actl[n] * dc * u
                jg = np.argsort(-np.abs(g * hl))[:4]; ju = np.argsort(-np.abs(up * hl))[:4]; ii = np.argsort(-np.abs(ci))[:4]
                cells.append({"n": n, "d": round(float(dn[n]), 4), "act": round(float(actl[n]), 4), "gate": [[int(j), round(float(g[j]), 4), round(float(hl[j]), 3)] for j in jg], "up": [[int(j), round(float(up[j]), 4), round(float(hl[j]), 3)] for j in ju],
                              "down": [[int(i), round(float(dc[i]), 4), round(float(ci[i]), 4)] for i in ii]})
            layers.append({"heads": [round(x, 3) for x in dh], "neurons": [[int(n), round(float(dn[n]), 3)] for n in top], "cells": cells})
            for n in top: cand.append((abs(float(dn[n])), l, "ffn", int(n), float(dn[n])))
            for h in range(self.nh): cand.append((abs(dh[h]), l, "head", h, dh[h]))
        cand.sort(reverse=True); lg = float(logits[t])
        return {"token": self.tok.decode([t]), "logit": round(lg, 3), "direct_sum": round(direct, 3), "top": [{"layer": c[1], "kind": c[2], "id": c[3], "d": round(c[4], 3)} for c in cand[:14]], "layers": layers}

    def run(self, prompt, max_new):
        msgs = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"; ids = self.tok.encode(msgs)
        kc = np.zeros((self.L, self.nkv, len(ids) + max_new + 4, self.hd), np.float32); vc = np.zeros_like(kc); passes = []; p0 = 0; feed = ids; reply = []; t0 = time.time()
        for step in range(max_new + 1):
            logits, recs, paths = self.forward(feed, kc, vc, p0); p0 += len(feed)
            pr = np.exp(logits - logits.max()); pr /= pr.sum(); o = np.argsort(-pr)[:5]
            nxt = int(o[0]); passes.append({"kind": "prompt" if step == 0 else "word", "tokens": len(feed), "fed": [self.tok.decode([i]) for i in feed] if step == 0 else [self.tok.decode([feed[0]])], "layers": recs, "paths": paths,
                                            "next": [[self.tok.decode([int(i)]), round(float(pr[i]), 4)] for i in o], "seconds": round(time.time() - t0, 1)})
            print(f"  pass {step}: fed {len(feed)} token(s) -> {self.tok.decode([nxt])!r} p={pr[nxt]:.2f}  ({round(time.time() - t0)} s)", flush=True)
            if nxt in self.eos: break
            reply.append(nxt); feed = [nxt]
        return {"prompt": prompt, "chat_tokens": len(ids), "reply": self.tok.decode(reply), "reply_tokens": len(reply), "passes": passes, "layers_crossed": len(passes) * self.L, "forward_passes": len(passes)}


if __name__ == "__main__":
    T = Tracer(); out = {"model": "Qwen2.5-3B-Instruct", "layers": T.L, "heads": T.nh, "ffn": 11008, "system_prompt": SYSTEM, "runs": []}
    for p in PROMPTS:
        print("prompt:", p, flush=True); out["runs"].append(T.run(p, MAX_NEW[p])); print("  reply:", out["runs"][-1]["reply"], flush=True)
        json.dump(out, open(os.path.join(ROOT, "model", "trace_prompts.json"), "w"))
    print("done")
