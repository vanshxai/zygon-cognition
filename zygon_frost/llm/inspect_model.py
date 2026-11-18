"""Dump architecture metadata, tensor inventory, and dequantization sanity stats."""
import sys
from collections import Counter
from gguf import GGUF

g = GGUF()
print(f"GGUF v{g.version}  tensors={len(g.tensors)}  meta keys={len(g.meta)}  data_start={g.data_start}\n")

print("== METADATA (big arrays summarised) ==")
for k, v in g.meta.items():
    if isinstance(v, list) and len(v) > 8:
        print(f"{k}: list[{len(v)}] e.g. {v[:4]}")
    else:
        s = repr(v)
        print(f"{k}: {s if len(s) < 200 else s[:200] + '...'}")

print("\n== TENSOR TYPES ==")
c = Counter(t.type_name for t in g.tensors.values())
for k, v in c.items():
    print(f"{k}: {v} tensors, {sum(t.n_bytes for t in g.tensors.values() if t.type_name == k) / 1e6:.1f} MB")

print("\n== NON-LAYER TENSORS + BLOCK 0 ==")
for name, t in g.tensors.items():
    if not name.startswith("blk.") or name.startswith("blk.0."):
        print(f"{name:28s} shape={str(t.shape):18s} {t.type_name:5s} {t.n_bytes / 1e6:8.2f} MB")

print("\n== DEQUANT SANITY (first row of a few tensors) ==")
for name in ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight", "blk.0.attn_norm.weight"]:
    if name in g.tensors:
        r = g.row(name, 0) if len(g.tensors[name].shape) > 1 else g.dequantize(name)
        mean = sum(r) / len(r)
        std = (sum((x - mean) ** 2 for x in r) / len(r)) ** 0.5
        print(f"{name}: n={len(r)} mean={mean:+.5f} std={std:.5f} min={min(r):+.4f} max={max(r):+.4f}")
