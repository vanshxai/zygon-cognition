"""Qwen2 transformer forward pass in plain numpy, with hooks for interpretability experiments.

All weights live in `self.W` (dict name -> float32 ndarray, numpy shape (out, in)); edit them freely.
"""
import os, time
import numpy as np
from gguf import GGUF, np_tensor
from tokenizer import Tokenizer


def rmsnorm(x, w, eps):
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def ftz(a):
    """Flush-to-zero: denormals make CPU matmuls ~100x slower; |v|<1e-20 is numerically irrelevant."""
    a[np.abs(a) < 1e-20] = 0.0
    return a


def silu(x):
    return x / (1.0 + np.exp(-x))


FROST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "frost_1.0")


class KVCache:
    def __init__(self, n_layers, n_kv, hd, max_len):
        self.k = np.zeros((n_layers, n_kv, max_len, hd), np.float32)
        self.v = np.zeros_like(self.k)
        self.len = 0


class Model:
    def __init__(self, path=None, verbose=True, patch=True):
        g = GGUF(path) if path else GGUF()
        self.g, m = g, g.meta
        self.n_layers = m["qwen2.block_count"]
        self.dim = m["qwen2.embedding_length"]
        self.n_heads = m["qwen2.attention.head_count"]
        self.n_kv = m["qwen2.attention.head_count_kv"]
        self.hd = self.dim // self.n_heads
        self.eps = m["qwen2.attention.layer_norm_rms_epsilon"]
        self.theta = m["qwen2.rope.freq_base"]
        self.tok = Tokenizer(m)
        t0 = time.time()
        self.W = {}
        for i, name in enumerate(g.tensors):
            self.W[name] = np_tensor(g, name).astype(np.float32)
            if verbose and i % 60 == 0:
                print(f"\rloading weights {i}/{len(g.tensors)}", end="", flush=True)
        if verbose:
            print(f"\rloaded {len(self.W)} tensors in {time.time() - t0:.1f}s        ")
        # Tied embeddings: no output.weight in this file.
        self.W.setdefault("output.weight", self.W["token_embd.weight"])
        self.patched = []
        if patch and os.path.isdir(FROST_DIR):          # Frost 1.0 = original weights + the tensors saved in this folder
            for fn in sorted(os.listdir(FROST_DIR)):
                if fn.endswith(".npy"):
                    name = fn[:-4]
                    new = np.load(os.path.join(FROST_DIR, fn)).astype(np.float32)
                    assert new.shape == self.W[name].shape, (name, new.shape)
                    self.W[name] = new
                    self.patched.append(name)
            if verbose and self.patched:
                print("Frost 1.0 patch applied:", self.patched)
        half = self.hd // 2
        self.inv_freq = (1.0 / self.theta ** (np.arange(half) * 2.0 / self.hd)).astype(np.float32)

    def new_cache(self, max_len=4096):
        return KVCache(self.n_layers, self.n_kv, self.hd, max_len)

    def _rope(self, x, pos):
        # x: (heads, T, hd); NEOX style = rotate (first half, second half)
        ang = pos[:, None] * self.inv_freq[None, :]
        cos, sin = np.cos(ang), np.sin(ang)
        h = self.hd // 2
        x1, x2 = x[..., :h], x[..., h:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def forward(self, ids, cache, hook=None, all_logits=False):
        """Run `ids` through the model, appending to `cache`. Returns logits for the last position.
        hook(layer, point, array) -> array|None is called at: 'resid_in','attn_in','q','k','v',
        'attn_pattern','attn_ctx','attn_out','resid_mid','ffn_in','gate','up','act','mlp_out','resid_out'. Returning an array replaces that activation."""
        W, T, p0 = self.W, len(ids), cache.len
        pos = np.arange(p0, p0 + T, dtype=np.float32)
        x = W["token_embd.weight"][ids]

        def H(l, pt, a):
            if hook:
                r = hook(l, pt, a)
                return a if r is None else r
            return a

        H(-1, "input_ids", np.asarray(ids))
        rep = self.n_heads // self.n_kv
        for l in range(self.n_layers):
            p = f"blk.{l}."
            x = H(l, "resid_in", x)
            h = H(l, "attn_in", rmsnorm(x, W[p + "attn_norm.weight"], self.eps))
            q = (h @ W[p + "attn_q.weight"].T + W[p + "attn_q.bias"]).reshape(T, self.n_heads, self.hd).transpose(1, 0, 2)
            k = (h @ W[p + "attn_k.weight"].T + W[p + "attn_k.bias"]).reshape(T, self.n_kv, self.hd).transpose(1, 0, 2)
            v = (h @ W[p + "attn_v.weight"].T + W[p + "attn_v.bias"]).reshape(T, self.n_kv, self.hd).transpose(1, 0, 2)
            q, k, v = H(l, "q", q), H(l, "k", k), H(l, "v", v)
            q, k = self._rope(q, pos), self._rope(k, pos)
            cache.k[l, :, p0:p0 + T], cache.v[l, :, p0:p0 + T] = k, v
            K, V = cache.k[l, :, :p0 + T], cache.v[l, :, :p0 + T]
            K, V = np.repeat(K, rep, 0), np.repeat(V, rep, 0)           # GQA: share kv heads
            s = (q @ K.transpose(0, 2, 1)) / np.float32(np.sqrt(self.hd))           # (heads, T, p0+T)
            if T > 1:
                s = s + np.triu(np.full((T, p0 + T), -np.inf, np.float32), k=p0 + 1)
            s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True); ftz(s)
            s = H(l, "attn_pattern", s)
            a = H(l, "attn_ctx", (s @ V).transpose(1, 0, 2).reshape(T, self.dim))
            a = H(l, "attn_out", a @ W[p + "attn_output.weight"].T)
            x = H(l, "resid_mid", x + a)
            h = H(l, "ffn_in", rmsnorm(x, W[p + "ffn_norm.weight"], self.eps))
            g, u = H(l, "gate", h @ W[p + "ffn_gate.weight"].T), H(l, "up", h @ W[p + "ffn_up.weight"].T)
            f = H(l, "act", ftz(silu(g) * u)) @ W[p + "ffn_down.weight"].T
            x = H(l, "resid_out", x + H(l, "mlp_out", f))
        cache.len += T
        if all_logits:                                   # logits for every fed position (used to verify several drafted tokens at once)
            return rmsnorm(x, W["output_norm.weight"], self.eps) @ W["output.weight"].T
        x = H(self.n_layers, "final_norm", rmsnorm(x[-1], W["output_norm.weight"], self.eps))
        return H(self.n_layers, "logits", W["output.weight"] @ x)

    def sample(self, logits, temp=0.7, top_k=20, top_p=0.8, rng=np.random):
        if temp <= 0:
            return int(np.argmax(logits))
        idx = np.argpartition(logits, -top_k)[-top_k:]
        idx = idx[np.argsort(-logits[idx])]
        pr = np.exp((logits[idx] - logits[idx[0]]) / temp); pr /= pr.sum()
        keep = np.cumsum(pr) - pr < top_p
        idx, pr = idx[keep], pr[keep] / pr[keep].sum()
        return int(rng.choice(idx, p=pr))

    def generate(self, ids, max_new=200, cache=None, hook=None, **sk):
        """Yield (token_id, text_piece) for each new token. Stops at EOS (<|im_end|>)."""
        cache = cache or self.new_cache(min(32768, len(ids) + max_new + 8))   # right-sized; 32768 = the model's context window
        if hook is None and len(ids) > 512:
            for i in range(0, len(ids), 512):                                 # chunked prefill: same maths, bounded attention memory
                logits = self.forward(ids[i:i + 512], cache)
        else:
            logits = self.forward(ids, cache, hook)
        dec = []
        for _ in range(max_new):
            t = self.sample(logits, **sk)
            self.last_sampled = t
            if t == self.tok.eos or t == self.tok.id_of["<|im_end|>"]:
                self.stop_reason = "end-of-turn token"
                return
            dec.append(t)
            txt = self.tok.decode(dec)
            if not txt.endswith("�"):          # wait for complete UTF-8 sequences
                dec = []
                yield t, txt
            logits = self.forward([t], cache, hook)
        self.stop_reason = "max new tokens reached"


def chatml(messages):
    s = "".join(f"<|im_start|>{r}\n{c}<|im_end|>\n" for r, c in messages)
    return s + "<|im_start|>assistant\n"
