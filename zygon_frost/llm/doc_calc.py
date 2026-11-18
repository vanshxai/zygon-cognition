"""Writes docs/hi_and_hvac_exact_calculation.txt: the exact numbers of the 3B model reading the prompts "hi" and "what is hvac", in plain text maths (no LaTeX).   python doc_calc.py"""
import json, os, sys, time
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_model import Tracer, rms, silu, SYSTEM
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
f5 = lambda v: f"{float(v):.5g}"
fl = lambda a, n=6: "[" + ", ".join(f5(x) for x in np.asarray(a).ravel()[:n]) + ", ...]"


def walk(T, prompt):
    chat = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"; ids = T.tok.encode(chat); n = len(ids); pos = np.arange(n, dtype=np.float32)
    x = T.W["token_embd.weight"][ids].astype(np.float32); rep = T.nh // T.nkv; out = []; table = []; E = T.W["token_embd.weight"]
    out.append(f"Chat text sent to the model ({n} pieces). Each piece is shown as  position: id = text")
    out.append("  " + "   ".join(f"{i}: {t} = {T.tok.decode([t])!r}" for i, t in enumerate(ids))); out.append("")
    out.append(f"Step 0. Look up the rows.  x[position] = word_table[id]   (each row = 2048 numbers). We follow the LAST position ({n - 1}), because its final row decides the first word of the reply.")
    out.append(f"  x[{n - 1}] = word_table[{ids[-1]}] = {fl(x[-1])}   size |x| = sqrt(sum of squares) = {f5(np.linalg.norm(x[-1]))}"); out.append("")
    for l in range(T.L):
        p = f"blk.{l}."; xin = x; D = l in (0, T.L - 1); L = []
        g1 = T.W[p + "attn_norm.weight"]; r1 = np.sqrt((x * x).mean(-1, keepdims=True) + T.eps); h = x / r1 * g1
        Wq, Wk, Wv = T.w(p + "attn_q.weight"), T.w(p + "attn_k.weight"), T.w(p + "attn_v.weight"); bq, bk, bv = T.W[p + "attn_q.bias"], T.W[p + "attn_k.bias"], T.W[p + "attn_v.bias"]
        q = (h @ Wq.T + bq).reshape(n, T.nh, T.hd).transpose(1, 0, 2); k = (h @ Wk.T + bk).reshape(n, T.nkv, T.hd).transpose(1, 0, 2); v = (h @ Wv.T + bv).reshape(n, T.nkv, T.hd).transpose(1, 0, 2)
        q0, q64 = float(q[0, -1, 0]), float(q[0, -1, 64]); q, k = T.rope(q, pos), T.rope(k, pos)
        K, V = np.repeat(k, rep, 0), np.repeat(v, rep, 0); s = (q @ K.transpose(0, 2, 1)) / np.float32(np.sqrt(T.hd)); s = s + np.triu(np.full((n, n), -np.inf, np.float32), k=1)
        raw = s[0, -1].copy(); s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True); ctx = (s @ V).transpose(1, 0, 2); a = ctx.reshape(n, T.dim) @ T.w(p + "attn_output.weight").T; xmid = x + a
        g2 = T.W[p + "ffn_norm.weight"]; r2 = np.sqrt((xmid * xmid).mean(-1, keepdims=True) + T.eps); h2 = xmid / r2 * g2
        gt = h2 @ T.w(p + "ffn_gate.weight").T; up = h2 @ T.w(p + "ffn_up.weight").T; sg = silu(gt); act = sg * up; Dn = T.w(p + "ffn_down.weight"); f = act @ Dn.T; x = xmid + f
        table.append((l + 1, np.linalg.norm(xin[-1]), 100 * np.linalg.norm(a[-1]) / np.linalg.norm(xin[-1]), 100 * np.linalg.norm(f[-1]) / np.linalg.norm(xmid[-1]), np.linalg.norm(x[-1]), 100 * float((sg[-1] > 0.1).mean())))
        if not D: continue
        L.append(f"================ BLOCK {l + 1} of {T.L}, worked in full for position {n - 1} ================")
        L.append(f"A. Scale the row (RMSNorm).   rms = sqrt(mean(x^2) + 0.000001) = {f5(r1[-1, 0])}   h = x / rms * gain")
        L.append(f"   x    = {fl(xin[-1])}"); L.append(f"   gain = {fl(g1)}   (2048 stored numbers of this block)"); L.append(f"   h    = {fl(h[-1])}")
        for nm, W, b, row in (("query q[head 1, number 0]", Wq, bq, 0), ("key k[group 1, number 0]", Wk, bk, 0), ("value v[group 1, number 0]", Wv, bv, 0)):
            pr = W[row] * h[-1]; L.append(f"B. {nm} = bias + sum over j of W[0][j] * h[j]  (2048 terms)")
            L.append(f"   first terms: " + " + ".join(f"({f5(W[row][j])} * {f5(h[-1][j])})" for j in range(4)) + " + ...")
            L.append(f"   bias {f5(b[row])} + sum of all 2048 products {f5(pr.sum())} = {f5(b[row] + pr.sum())}")
        cs, sn = np.cos((n - 1) * T.inv[0]), np.sin((n - 1) * T.inv[0])
        L.append(f"C. Mark the position (rotary).  angle = position * 1000000^(-2*0/128) = {n - 1}   cos = {f5(cs)}  sin = {f5(sn)}")
        L.append(f"   q'[0] = q[0]*cos - q[64]*sin = {f5(q0)}*{f5(cs)} - {f5(q64)}*{f5(sn)} = {f5(q0 * cs - q64 * sn)}   (the same turn is applied to every pair of numbers of every query and key)")
        o = np.argsort(-s[0, -1])[:4]; L.append(f"D. Head 1 asks: how much does position {n - 1} care about each earlier position?   score[t] = sum(q' * k'[t]) / sqrt(128),  weight[t] = e^score[t] / sum of e^score")
        for t in o: L.append(f"   position {t} {T.tok.decode([ids[t]])!r}: score {f5(raw[t])}  weight {f5(s[0, -1][t])}")
        L.append(f"   head 1 output = sum over t of weight[t] * v[t] = {fl(ctx[-1][0])}   (16 heads like this, then joined: 2048 numbers)")
        L.append(f"E. Mix the heads.  attn_out[i] = sum over j of Wo[i][j] * joined[j] = {fl(a[-1])}   then  x_mid = x + attn_out = {fl(xmid[-1])}")
        L.append(f"F. Scale again.  rms = {f5(r2[-1, 0])}   h2 = x_mid / rms * gain2 = {fl(h2[-1])}")
        nn = int(np.argmax(np.abs(act[-1]))); gw, uw = T.W[p + "ffn_gate.weight"][nn].astype(np.float32), T.W[p + "ffn_up.weight"][nn].astype(np.float32)
        L.append(f"G. The strongest of the {T.W[p + 'ffn_gate.weight'].shape[0]} feed-forward neurons for this position is neuron {nn}.")
        L.append(f"   gate = sum over j of Wgate[{nn}][j] * h2[j] = " + " + ".join(f"({f5(gw[j])} * {f5(h2[-1][j])})" for j in range(3)) + f" + ... = {f5(gt[-1][nn])}")
        L.append(f"   up   = sum over j of Wup[{nn}][j] * h2[j]   = {f5(up[-1][nn])}")
        L.append(f"   silu(gate) = gate / (1 + e^(-gate)) = {f5(gt[-1][nn])} / (1 + e^({f5(-gt[-1][nn])})) = {f5(sg[-1][nn])}")
        L.append(f"   act = silu(gate) * up = {f5(sg[-1][nn])} * {f5(up[-1][nn])} = {f5(act[-1][nn])}")
        L.append(f"   Gates open (silu(gate) > 0.1): {100 * float((sg[-1] > 0.1).mean()):.1f}% of {gt.shape[1]} neurons.  Top 5 act values: " + ", ".join(f"n{int(i)}={f5(act[-1][i])}" for i in np.argsort(-np.abs(act[-1]))[:5]))
        tm = Dn[0] * act[-1]; big = np.argsort(-np.abs(tm))[:3]
        L.append(f"H. Squeeze back.  ffn_out[0] = sum over n of Wdown[0][n] * act[n]  (11008 terms) ; biggest terms: " + ", ".join(f"n{int(i)}: {f5(Dn[0][i])}*{f5(act[-1][i])}={f5(tm[i])}" for i in big) + f" ; total = {f5(tm.sum())}")
        L.append(f"   ffn_out = {fl(f[-1])}   x_out = x_mid + ffn_out = {fl(x[-1])}   size |x_out| = {f5(np.linalg.norm(x[-1]))}"); L.append(""); out += L
        if l == 0: out.append(f"   ... blocks 2 to {T.L - 1} do the same steps A to H with their own stored numbers (see the table below) ...\n")
    out.append("================ ALL 36 BLOCKS, position " + str(n - 1) + " (sizes of the running row) ================")
    out.append("block   |x in|    attention changed the row by   feed-forward changed it by   |x out|    gates open")
    for r in table: out.append(f"{r[0]:>5}   {r[1]:>8.2f}   {r[2]:>10.1f} %                  {r[3]:>10.1f} %               {r[4]:>8.2f}   {r[5]:>6.1f} %")
    xl = x[-1]; rf = float(np.sqrt((xl * xl).mean() + T.eps)); go = T.W["output_norm.weight"]; xn = xl / rf * go; sc = E.astype(np.float32) @ xn; top = np.argsort(-sc)[:5]; mx = sc.max(); Z = float(np.exp(sc - mx).sum())
    out.append(""); out.append("================ THE LAST STEP: choose the first word ================")
    out.append(f"Scale the final row.  rms = sqrt(mean(x^2) + 0.000001) = {f5(rf)}   xn = x / rms * final_gain = {fl(xn)}")
    out.append("Score every one of the 151936 pieces with the SAME word table:  score[piece] = sum over i of word_table[piece][i] * xn[i]   (2048 terms each)")
    out.append("Turn scores into chances:  chance[piece] = e^(score - best) / sum over all 151936 pieces of e^(score - best)")
    for t in top:
        pr = E[t].astype(np.float32) * xn; out.append(f"   piece {int(t)} {T.tok.decode([int(t)])!r}: first terms {f5(E[t][0])}*{f5(xn[0])} + {f5(E[t][1])}*{f5(xn[1])} + ... = score {f5(sc[t])}   chance {float(np.exp(sc[t] - mx) / Z):.4f}")
    out.append(f"The best piece is chosen: {T.tok.decode([int(top[0])])!r}.  Total over all pieces of e^(score - best) = {f5(Z)}"); return "\n".join(out), int(top[0])


if __name__ == "__main__":
    T = Tracer(); tr = json.load(open(os.path.join(ROOT, "model", "trace_prompts.json"))); os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True); t0 = time.time()
    head = """EXACT CALCULATION OF TWO PROMPTS IN THE 3B MODEL (Qwen2.5-3B-Instruct, 36 blocks, 2048 numbers per piece)
Plain text maths. Symbols: * times, / divided by, ^ power, sqrt() square root, e^(x) the exponential, sum over j of ... means add up the term for every j.
Every number below was computed by running the real model (see llm/doc_calc.py and llm/trace_model.py). Numbers are shown with 5 significant digits. [a, b, ...] shows the first numbers of a row of 2048.

HOW ONE PASS WORKS (what each block does to the running row x of 2048 numbers)
  A  h = x / sqrt(mean(x^2) + 0.000001) * gain                     scale the row to a steady size
  B  q = Wq * h + bq,  k = Wk * h + bk,  v = Wv * h + bv            three views of the row; (W * h)[i] = sum over j of W[i][j] * h[j]
  C  q and k are turned by an angle that depends on the position   so the model knows the order of the pieces
  D  score[t] = sum(q * k[t]) / sqrt(128);  weight[t] = e^score[t] / sum of e^score;  head output = sum over t of weight[t] * v[t]
  E  x_mid = x + Wo * (all 16 head outputs joined)                   attention result added back
  F  scale again as in A
  G  gate = Wgate * h2,  up = Wup * h2,  silu(g) = g / (1 + e^(-g)),  act = silu(gate) * up
  H  x_out = x_mid + Wdown * act                                     feed-forward result added back
  After the 36th block:  xn = final scale of x;  score[piece] = sum over i of word_table[piece][i] * xn[i];  the best-scoring piece is the next word.
The prompt is read in ONE pass (all its pieces together). Then every new word costs one more pass through all 36 blocks, feeding the previous word back in.
"""
    parts = [head]
    for r in tr["runs"]:
        text, first = walk(T, r["prompt"])
        parts.append("\n\n" + "#" * 100 + f"\n# PROMPT: {r['prompt']!r}\n" + "#" * 100 + "\n" + text)
        parts.append("\n\n================ THE WHOLE REPLY, pass by pass ================\npass  piece fed in            best next piece (chance)      next 4 best")
        for i, ps in enumerate(r["passes"]):
            fed = ("<whole prompt, " + str(ps["tokens"]) + " pieces>") if i == 0 else repr(ps["fed"][0])
            parts.append(f"{i + 1:>4}  {fed:<24} {ps['next'][0][0]!r} ({ps['next'][0][1]:.4f})".ljust(66) + "  " + ", ".join(f"{t!r} {p_:.3f}" for t, p_ in ps["next"][1:]))
        parts.append(f"\nReply: {r['reply']}\nPasses: {r['forward_passes']}.  Each pass goes through all 36 blocks, so block crossings = 36 * {r['forward_passes']} = {r['layers_crossed']}.")
        print(r["prompt"], "done", round(time.time() - t0), "s", flush=True)
    path = os.path.join(ROOT, "docs", "hi_and_hvac_exact_calculation.txt"); open(path, "w", encoding="utf-8").write("\n".join(parts)); print("written", path, round(os.path.getsize(path) / 1024), "KB")
