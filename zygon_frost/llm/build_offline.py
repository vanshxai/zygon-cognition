"""Data for the stand-alone exploded view (works without the dashboard).   python build_offline.py -> model/offline_data.json
Embedded: (1) the flipped word table for the 40,000 commonest pieces plus every piece used in the two traces: label, 48 colour bars of its 2048 numbers, 12 look-alikes;
(2) for each of the 252 matrices the exact 64 x 64 corner and a 48 x 48 squeezed map; (3) exact windows (128 numbers) around the cells on the traced paths."""
import base64, json, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf
from tokenizer import Tokenizer
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); t0 = time.time()
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
g = gguf.GGUF(os.path.join(ROOT, "model", "original", "qwen2.5-3b-instruct-q4_k_m.original.gguf")); tok = Tokenizer(g.meta)
tr = json.load(open(os.path.join(ROOT, "model", "trace_prompts.json"))); ex = json.load(open(os.path.join(ROOT, "model", "exploded_data.json")))
KEYS = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
# ---------------------------------------------------------------- slices wanted on the traced paths
want = {}
for r in tr["runs"]:
    for ps in r["passes"]:
        P = ps["paths"]
        for t in P["top"]:
            if t["kind"] != "ffn": continue
            cell = next((c for c in P["layers"][t["layer"]]["cells"] if c["n"] == t["id"]), None)
            if cell: want[(t["layer"], t["id"])] = cell
print(len(want), "neurons on the traced paths", flush=True)
corner, pooled, slices = {}, {}, []
for l in range(36):
    corner[l], pooled[l] = {}, {}
    for k in KEYS:
        A = gguf.np_tensor(g, f"blk.{l}.{k}.weight"); R, C = A.shape; sd = ex["layers"][str(l)][k]["std"]
        corner[l][k] = b64(np.round(A[:64, :64].astype(np.float64), 6).astype(np.float32))
        ri = np.linspace(0, R, 49).astype(int); ci = np.linspace(0, C, 49).astype(int); n = np.outer(np.diff(ri), np.diff(ci))
        ab = np.add.reduceat(np.add.reduceat(np.abs(A), ri[:-1], axis=0), ci[:-1], axis=1) / n; mn = np.add.reduceat(np.add.reduceat(A, ri[:-1], axis=0), ci[:-1], axis=1) / n
        pooled[l][k] = {"abs": b64(np.clip(ab / (4 * sd) * 255, 0, 255).astype(np.uint8)), "mean": b64(np.clip(mn / (3 * sd) * 127, -127, 127).astype(np.int8))}
        for (ll, nn), cell in want.items():
            if ll != l: continue
            if k in ("ffn_gate", "ffn_up"):
                j0 = cell["gate" if k == "ffn_gate" else "up"][0][0]; c0 = max(0, min(C - 128, j0 - 64)); slices.append({"layer": l, "key": k, "r0": nn, "r1": nn + 1, "c0": c0, "c1": c0 + 128, "data": b64(np.round(A[nn, c0:c0 + 128].astype(np.float64), 6).astype(np.float32))})
            if k == "ffn_down":
                i0 = cell["down"][0][0]; r0 = max(0, min(R - 128, i0 - 64)); slices.append({"layer": l, "key": k, "r0": r0, "r1": r0 + 128, "c0": nn, "c1": nn + 1, "data": b64(np.round(A[r0:r0 + 128, nn].astype(np.float64), 6).astype(np.float32))})
    print("block", l + 1, round(time.time() - t0), "s", flush=True)
# ---------------------------------------------------------------- the flipped word table
E = gguf.np_tensor(g, "token_embd.weight"); texts = set()
for r in tr["runs"]:
    for ps in r["passes"]:
        texts.update(ps["fed"]); texts.update(t for t, _ in ps["next"])
extra = {t[0] for tx in texts if len(t := tok.encode(tx)) == 1}
ids = sorted(set(range(40000)) | extra); print(len(ids), "pieces in the flipped table", flush=True)
AP = E[ids].astype(np.float32); sd = float(E[::13, ::7].std()); ci = np.linspace(0, AP.shape[1], 49).astype(int)
strips = np.clip(np.add.reduceat(AP, ci[:-1], axis=1) / np.diff(ci) / (0.9 * sd) * 127, -127, 127).astype(np.int8)
N = AP / (np.linalg.norm(AP, axis=1, keepdims=True) + 1e-9); near = np.zeros((len(ids), 12), np.uint16); sim = np.zeros((len(ids), 12), np.uint8)
for a in range(0, len(ids), 1500):
    S = N[a:a + 1500] @ N.T; S[np.arange(S.shape[0]), np.arange(a, a + S.shape[0])] = -9; top = np.argpartition(-S, 12, axis=1)[:, :12]
    for i in range(S.shape[0]):
        o = top[i][np.argsort(-S[i, top[i]])]; near[a + i] = o; sim[a + i] = np.clip(S[i, o] * 255, 0, 255)
    print("look-alikes", a + S.shape[0], "/", len(ids), round(time.time() - t0), "s", flush=True)
labels = [tok.decode([i]) for i in ids]
out = {"corner": corner, "pooled": pooled, "slices": slices, "flip": {"ids": ids, "labels": labels, "strips": b64(strips), "near": b64(near), "sim": b64(sim), "std": sd}}
json.dump(out, open(os.path.join(ROOT, "model", "offline_data.json"), "w")); print("done", round(time.time() - t0), "s", round(os.path.getsize(os.path.join(ROOT, "model", "offline_data.json")) / 1e6, 1), "MB", len(slices), "slices")
