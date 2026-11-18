"""How different are the 36 blocks? Reads the exponent bytes (sampled) and prints per block: weight size per matrix, similarity to block 0 and to the previous block, and the bits per weight an entropy coder would need."""
import json, os, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_weights as EW
ROOT = EW.ROOT; OUT = EW.OUT; LUT = EW.LUT
KEYS = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
rng = np.random.default_rng(1); idx = {}
def sample(k):
    a = np.load(os.path.join(OUT, f"blk.0.{k}.weight.npy"), mmap_mode="r"); n = a.size
    if k not in idx: idx[k] = rng.choice(n, 400000, replace=False)
    return idx[k]
def vals(l, k):
    a = np.load(os.path.join(OUT, f"blk.{l}.{k}.weight.npy"), mmap_mode="r"); return a.reshape(-1)[sample(k)]
def ent(codes):
    c = np.bincount(codes, minlength=256) / len(codes); c = c[c > 0]; return float(-(c * np.log2(c)).sum())
rows = []; allc = []; base = {k: LUT[vals(0, k)] for k in KEYS}; prev = dict(base)
for l in range(36):
    r = {"block": l + 1}; sims0 = []; simsp = []; stds = {}; es = []
    for k in KEYS:
        c = vals(l, k); v = LUT[c]; stds[k] = float(v.std()); es.append(ent(c)); allc.append(c)
        cs = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        sims0.append(cs(v, base[k])); simsp.append(cs(v, prev[k])); prev[k] = v
    r.update(std_attn=round(np.mean([stds[k] for k in KEYS[:4]]), 4), std_ffn=round(np.mean([stds[k] for k in KEYS[4:]]), 4), sim_to_block1=round(float(np.mean(sims0)), 3), sim_to_previous=round(float(np.mean(simsp)), 3), bits=round(float(np.mean(es)), 2))
    rows.append(r)
json.dump(rows, open(os.path.join(ROOT, "model", "block_diff.json"), "w"))
print("block  attn-size ffn-size  alike-to-block1  alike-to-previous  bits/weight")
for r in rows: print(f"{r['block']:5d}  {r['std_attn']:8.4f} {r['std_ffn']:8.4f}  {r['sim_to_block1']:14.3f}  {r['sim_to_previous']:16.3f}  {r['bits']:10.2f}")
print("overall bits per weight (entropy of the exponent byte):", round(ent(np.concatenate(allc)), 2))
