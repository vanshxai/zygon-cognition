"""Logit lens: decode the residual stream after every layer to watch the answer form."""
import sys, numpy as np
from engine import Model, rmsnorm
sys.stdout.reconfigure(encoding="utf-8")
m = Model(verbose=False)
prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
ids = m.tok.encode(prompt)
seen = {}
def hook(l, pt, a):
    if pt == "resid_out":
        seen[l] = a[-1].copy()
m.forward(ids, m.new_cache(), hook)
print(f"prompt: {prompt!r}\n")
for l in range(m.n_layers):
    x = rmsnorm(seen[l], m.W["output_norm.weight"], m.eps)
    lg = m.W["output.weight"] @ x
    top = np.argsort(-lg)[:3]
    p = np.exp(lg - lg.max()); p /= p.sum()
    print(f"layer {l:2d}  " + "  ".join(f"{m.tok.decode([t])!r}:{p[t]:.2f}" for t in top))
