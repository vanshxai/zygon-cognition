"""Qwen2.5-3B with every weight rewritten as sign x 2^(step/4)  ("exponent weights"), then run offline.
   python exp_weights.py convert      -> model/exp_weights/*.npy  (one byte per weight: sign bit + 7-bit exponent step; 0 = zero)
   python exp_weights.py run "question" [max_new]
Memory: the bytes are memory-mapped and one matrix is widened at a time (about 90 MB), logits are done in chunks. Fits the 4 GB budget."""
import os, sys, time
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUF, np_tensor
from tokenizer import Tokenizer
import trace_model as TM
sys.stdout.reconfigure(encoding="utf-8")
ROOT = TM.ROOT; OUT = os.path.join(ROOT, "model", "exp_weights"); STEPS = 4; BIAS = 100
GG = os.path.join(ROOT, "model", "original", "qwen2.5-3b-instruct-q4_k_m.original.gguf")
LUT = np.zeros(256, np.float32)
for c in range(256):
    e = c & 127
    if e: LUT[c] = (-1.0 if c >> 7 else 1.0) * 2.0 ** ((e - BIAS) / STEPS)


def encode(W):
    a = np.abs(W); e = np.round(np.log2(np.maximum(a, 1e-30)) * STEPS).astype(np.int32) + BIAS; ok = (e >= 1) & (a > 0); e = np.clip(e, 1, 127)
    return np.where(ok, e | ((W < 0).astype(np.int32) << 7), 0).astype(np.uint8)


def convert():
    os.makedirs(OUT, exist_ok=True); g = GGUF(GG); t0 = time.time(); err = []
    for i, n in enumerate(g.tensors):
        a = np_tensor(g, n)
        if a.ndim == 2:
            W = a.astype(np.float32); C = encode(W); np.save(os.path.join(OUT, n + ".npy"), C)
            s = np.random.default_rng(i).choice(W.size, 200000); err.append(float(np.sqrt(np.mean((LUT[C.ravel()[s]] - W.ravel()[s]) ** 2)) / np.sqrt(np.mean(W.ravel()[s] ** 2))))
        else: np.save(os.path.join(OUT, n + ".npy"), a.astype(np.float32))
        if i % 25 == 0: print(i, "/", len(g.tensors), n, round(time.time() - t0), "s", flush=True)
    print("done", round(time.time() - t0), "s  mean weight rms error", round(float(np.mean(err)), 4), flush=True)


class Chunk:
    def __init__(self, C): self.C = C
    def __matmul__(self, x): return np.concatenate([LUT[self.C[i:i + 20000]] @ x for i in range(0, self.C.shape[0], 20000)])


class Rows:
    def __init__(self, C): self.C = C
    def __getitem__(self, ids): return LUT[self.C[ids]]


class ExpTracer(TM.Tracer):
    def __init__(self):
        g = GGUF(GG); m = g.meta
        self.L, self.dim, self.nh, self.nkv = m["qwen2.block_count"], m["qwen2.embedding_length"], m["qwen2.attention.head_count"], m["qwen2.attention.head_count_kv"]
        self.hd = self.dim // self.nh; self.eps = m["qwen2.attention.layer_norm_rms_epsilon"]; self.tok = Tokenizer(m); self.C = {}; self.W = {}
        for n in g.tensors:
            a = np.load(os.path.join(OUT, n + ".npy"), mmap_mode="r")
            if a.dtype == np.uint8 and a.ndim == 2: self.C[n] = a
            else: self.W[n] = np.asarray(a)
        self.W["token_embd.weight"] = Rows(self.C["token_embd.weight"])
        self.inv = (1.0 / m["qwen2.rope.freq_base"] ** (np.arange(self.hd // 2) * 2.0 / self.hd)).astype(np.float32); self.eos = {self.tok.id_of["<|im_end|>"], self.tok.eos}

    def w(self, n):
        if n == "token_embd.weight": return Chunk(self.C[n])
        return LUT[self.C[n]] if n in self.C else self.W[n]

    def paths(self, *a): return None


def run(q, max_new=60):
    T = ExpTracer(); tok = T.tok
    ids = tok.encode("<|im_start|>system\n" + TM.SYSTEM + "<|im_end|>\n<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n")
    kc = np.zeros((T.L, T.nkv, len(ids) + max_new, T.hd), np.float32); vc = np.zeros_like(kc); out = []; t0 = time.time(); p0 = 0; cur = ids
    for _ in range(max_new):
        lg, _, _ = T.forward(cur, kc, vc, p0); p0 += len(cur); nxt = int(np.argmax(lg))
        if nxt in T.eos: break
        out.append(nxt); cur = [nxt]
    txt = tok.decode(out); print(f"Q: {q}\nA: {txt}\n({len(out)} tokens, {time.time() - t0:.0f} s)", flush=True); return txt


if __name__ == "__main__":
    if sys.argv[1] == "convert": convert()
    else: run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 60)
