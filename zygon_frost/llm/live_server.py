"""Transcript view of the model.   python live_server.py   then open http://127.0.0.1:8800

Ask a question; the server runs the model and APPENDS a plain-text transcript of every stage to one log:
text -> tokens (BPE merges) -> embeddings -> 36 layers of matrix math -> scores -> sampling -> detokenise.
  /              the log page          /weights   raw weight browser
  /api/run (POST), /api/ctl (POST), /api/log?since=N, /api/tensors, /api/weights
"""
import base64, html, json, os, re, struct, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
from engine import Model, chatml
from tokenizer import _PRETOK
import scan

sys.stdout.reconfigure(encoding="utf-8")
PORT = 8800
SYSTEM = "You are Frost 1.0, a helpful assistant."

m = Model()
g, tok = m.g, m.tok
D, F, NL = m.dim, m.W["blk.0.ffn_up.weight"].shape[0], m.n_layers
PER_LAYER = D * D * 2 + D * 256 * 2 + D * F * 3
HEAD = m.W["token_embd.weight"].shape[0] * D
PARAMS = sum(t.n_elements for t in g.tensors.values())
print(f"{PARAMS:,} parameters")


class Abort(Exception):
    pass


lock = threading.Lock()
S = {"log": [], "epoch": 0, "running": False, "neur": []}
R = {}
OPT = {"delay": 0.0, "stop": False, "full": 1, "compact": False}

# ---- formatting helpers ----------------------------------------------------
N = lambda x: f"{int(x):,}"
f = lambda v, d=4: f"{float(v):+.{d}f}"
rowtxt = lambda a, n=8, d=4: " ".join(f(x, d) for x in a[:n])
vis = lambda s: repr(s)


def put(s):
    with lock:
        S["log"].append(s)


def put_open(title, op=False):
    with lock:
        S["log"].append({"o": title, "open": False})


def put_close():
    with lock:
        S["log"].append({"c": 1})


def blk(text, op=False):
    """A collapsible block: first line is the dropdown title, the rest is the body."""
    first, _, rest = text.strip(chr(10)).partition(chr(10))
    put_open(first.strip(), op)
    if rest:
        put(rest.rstrip() + chr(10))
    put_close()


def step_done():
    if OPT["stop"]:
        raise Abort()
    if OPT["delay"] > 0:
        time.sleep(OPT["delay"])


def banner(title, ch="="):
    put(f"\n{ch * 100}\n{title}\n{ch * 100}\n")


# ---- STEP 2: tokenisation, with every BPE merge shown ------------------------
def bpe_explain(text):
    out, final_ids = [], []
    for chunk in tok._split.split(text):
        if not chunk:
            continue
        if chunk in tok.id_of and tok.types[tok.id_of[chunk]] == 3:
            out.append(f"special token {chunk!r:22} -> id {tok.id_of[chunk]}  (looked up directly, never split)")
            final_ids.append(tok.id_of[chunk]); continue
        for piece in _PRETOK.findall(chunk):
            bs = piece.encode("utf-8")
            mapped = "".join(tok.b2u[b] for b in bs)
            out.append(f"pre-token {piece!r:18} bytes {' '.join(f'{b:02x}' for b in bs)}  -> byte-chars {mapped!r}")
            parts = list(mapped)
            while len(parts) > 1:
                pairs = [(tok.ranks.get((parts[i], parts[i + 1]), 1 << 60), i) for i in range(len(parts) - 1)]
                r, i = min(pairs)
                if r == 1 << 60:
                    break
                a_, b_ = parts[i], parts[i + 1]
                parts[i:i + 2] = [a_ + b_]
                out.append(f"      merge #{r:<7} {a_!r} + {b_!r}  =>  {parts}")
            ids_ = [tok.id_of[p] for p in parts]
            out.append(f"      => {len(ids_)} token(s): " + ", ".join(f"{p!r}={i}" for p, i in zip(parts, ids_)))
            final_ids.extend(ids_)
    return out, final_ids


# ---- STEP 4: per-operation logging ------------------------------------------
def worked_example(w, x, y0, bias, full):
    prod = w.astype(np.float64) * x.astype(np.float64)
    cum = np.cumsum(prod)
    n = len(w)
    L = []
    if full:
        for i in range(n):
            L.append(f"      i={i:<5} w={f(w[i], 5)} x={f(x[i], 5)}  w*x={f(prod[i], 6)}  running total={f(cum[i], 6)}")
    else:
        for i in range(4):
            L.append(f"      i={i:<5} w={f(w[i], 5)} x={f(x[i], 5)}  w*x={f(prod[i], 6)}  running total={f(cum[i], 6)}")
        L.append(f"      ... {N(n - 5)} more multiply-adds, same pattern ...")
        i = n - 1
        L.append(f"      i={i:<5} w={f(w[i], 5)} x={f(x[i], 5)}  w*x={f(prod[i], 6)}  running total={f(cum[i], 6)}")
    tot = cum[-1] + bias
    ok = "MATCH" if abs(tot - y0) < 2e-3 * max(1.0, abs(y0)) else "differs (float32 rounding)"
    L.append(f"      sum of all {N(n)} products = {f(cum[-1], 6)}" + (f"  + bias {f(bias, 6)} = {f(tot, 6)}" if bias else "")
             + f"   | model's output = {f(y0, 6)}  -> {ok}")
    return L


def matmul_op(l, op, desc, wname, x, y, bias=0.0, once=False):
    Wm = m.W[wname]
    out, inn = Wm.shape
    Tn = 1 if once else R["T"]
    R["done"] += out * inn * Tn
    qt = g.tensors[wname].type_name
    head = f"[layer {l} | {op}] {desc}"
    cum = f"   (multiply-adds so far this pass: {N(R['done'])} / {N(R['expected'])})"
    if OPT["compact"]:
        put(f"{head}  W={wname} [{out}x{inn}] {qt}  {N(out * inn * Tn)} MACs  y[:4]={rowtxt(y, 4)}\n")
        step_done(); return
    L = [head + cum,
         f"    weight matrix {wname}   {out} rows (outputs) x {inn} cols (inputs)   stored as {qt}   = {N(out * inn)} weights",
         f"    work: {N(out)} x {N(inn)}" + (f" x {Tn} tokens" if Tn > 1 else "") + f" = {N(out * inn * Tn)} multiply-adds",
         f"    input  x (first 8 of {N(inn)}):  {rowtxt(x)}",
         "    W corner (rows 0-3, cols 0-7):"]
    L += [f"        row {r}: {rowtxt(Wm[r])}" for r in range(4)]
    L.append(f"    output y (first 8 of {N(out)}): {rowtxt(y)}")
    full = R["full_left"] > 0
    if full:
        R["full_left"] -= 1
    L.append(f"    worked example - output y[0] = (row 0 of W) . x{' + b[0]' if bias else ''}   ({N(inn)} multiply-adds" + (", EVERY ONE LISTED" if full else ", ends shown") + "):")
    L += worked_example(Wm[0], x, float(y[0]), float(bias), full)
    blk("\n".join(L) + "\n")
    step_done()


def rmsnorm_log(l, which, before, after, gain):
    if OPT["compact"]:
        return
    ms = float((before.astype(np.float64) ** 2).mean())
    inv = 1.0 / np.sqrt(ms + m.eps)
    blk(f"[layer {l} | RMSNorm {which}]  out = x * 1/sqrt(mean(x^2)+eps) * gain\n"
        f"    x (first 8):     {rowtxt(before)}\n"
        f"    mean(x^2) over all {D} = {ms:.6f}   1/sqrt(mean+eps) = {inv:.5f}\n"
        f"    gain (first 8):  {rowtxt(gain)}\n"
        f"    out (first 8):   {rowtxt(after)}\n\n")


def add_log(l, label, a, b, c):
    if OPT["compact"]:
        return
    blk(f"[layer {l} | residual add {label}]  (first 6 numbers)\n" +
        "".join(f"    {f(a[i])} + {f(b[i])} = {f(c[i])}\n" for i in range(6)) + "\n")


def hook(l, pt, a):
    p = f"blk.{l}."
    if pt == "input_ids":
        ids = [int(i) for i in a]
        R["fwd"] += 1; R["T"] = len(ids); R["done"] = 0
        R["expected"] = len(ids) * PER_LAYER * NL + HEAD
        R["pos0"] = len(R["ctx"]); R["ctx"] += ids
        what = (f"PREFILL: all {len(ids)} prompt tokens go through the network together" if len(ids) > 1 else
                f"DECODE: feed back the 1 token just chosen ({ids[0]} = {tok.decode(ids)!r}); the keys/values of the {R['pos0']} earlier tokens are re-used from the cache")
        R["layer_open"] = False
        put_open(f"FORWARD PASS #{R['fwd']}   {what}   |   about {N(R['expected'])} multiply-adds", True)
        nr = {"n": R["fwd"], "pos0": R["pos0"], "T": len(ids), "tokens": [tok.decode([t]) for t in ids],
              "q": np.zeros((NL, len(ids), F), np.int8), "sc": np.zeros((NL, len(ids)), np.float32)}
        R["neur"] = nr
        with lock:
            S["neur"].append(nr)
        return
    if pt == "resid_in":
        if l == 0:
            ids = R["ctx"][R["pos0"]:]
            emb = ["[EMBEDDING LOOKUP]  token_embd.weight is a table of 151,936 rows x %d columns; each token id selects ONE row (no arithmetic)." % D]
            if len(ids) <= 40:
                for i, t in enumerate(ids):
                    emb.append(f"    position {R['pos0'] + i:<3} id {t:<7} {tok.decode([t])!r:20} -> row {t}:  {rowtxt(m.W['token_embd.weight'][t], 6)} ... ({D} numbers)")
            blk(chr(10).join(emb) + chr(10))
        R["resid_in"] = a[-1].copy()
        if R.get("layer_open"):
            put_close()
        put_open(f"LAYER {l} of {NL - 1}   |   residual stream coming in (first 8 of {D}): {rowtxt(a[-1])}")
        R["layer_open"] = True
    elif pt == "attn_in":
        rmsnorm_log(l, "before attention", R["resid_in"], a[-1], m.W[p + "attn_norm.weight"])
        R["x_attn"] = a[-1].copy()
    elif pt in ("q", "k", "v"):
        w = {"q": "attn_q", "k": "attn_k", "v": "attn_v"}[pt]
        names = {"q": "Q = W_q x + b_q   (what am I looking for?)", "k": "K = W_k x + b_k   (what do I contain?)", "v": "V = W_v x + b_v   (what do I hand over?)"}
        matmul_op(l, f"attn_{pt}", names[pt], p + w + ".weight", R["x_attn"], a[:, -1, :].reshape(-1), m.W[p + w + ".bias"][0])
        if pt == "k" and not OPT["compact"]:
            pos = R["pos0"] + R["T"] - 1
            th = m.inv_freq[:3]
            blk(f"[layer {l} | RoPE]  Q and K are rotated by position so attention knows word order. Last token is position {pos}.\n" +
                "".join(f"    pair (q[{i}], q[{i + 64}]): angle = position x theta_{i} = {pos} x {th[i]:.5f} = {pos * th[i]:.4f} rad  -> cos {np.cos(pos * th[i]):+.4f}  sin {np.sin(pos * th[i]):+.4f}\n" for i in range(3)) +
                f"    (K and V of this token are now appended to the cache: 2 KV-heads x 128 x every token so far)\n\n")
    elif pt == "attn_pattern":
        pat = a[:, -1, :]
        ctx = R["ctx"][:pat.shape[1]]
        top = np.argsort(-pat[0])[:5]
        if OPT["compact"]:
            put(f"[layer {l} | attention] head0 top: " + ", ".join(f"{tok.decode([ctx[j]])!r} {pat[0][j]:.2f}" for j in top) + "\n"); step_done(); return
        blk(f"[layer {l} | attention]  for each of 16 heads: score_j = Q . K_j / sqrt(128) for every earlier token j, then softmax -> percentages; head output = sum_j weight_j x V_j\n"
            f"    context = {pat.shape[1]} tokens. Head 0, where the last token looks (top 5 of {pat.shape[1]}):\n" +
            "".join(f"        {tok.decode([ctx[j]])!r:18} position {j:<3} weight {pat[0][j]:.4f}  {'#' * int(round(pat[0][j] * 40))}\n" for j in top) +
            f"    largest weight in each head 0..15: {' '.join(f'{x:.2f}' for x in pat.max(1))}\n\n")
        step_done()
    elif pt == "attn_ctx":
        R["x_o"] = a[-1].copy()
    elif pt == "attn_out":
        matmul_op(l, "attn_output", "attn_out = W_o . (16 heads' blended V, glued together)", p + "attn_output.weight", R["x_o"], a[-1])
        R["attn_out"] = a[-1].copy()
    elif pt == "resid_mid":
        add_log(l, "1: resid_mid = resid_in + attn_out", R["resid_in"], R["attn_out"], a[-1])
        R["resid_mid"] = a[-1].copy()
    elif pt == "ffn_in":
        rmsnorm_log(l, "before MLP", R["resid_mid"], a[-1], m.W[p + "ffn_norm.weight"])
        R["x_ffn"] = a[-1].copy()
    elif pt == "gate":
        matmul_op(l, "ffn_gate", f"g = W_gate x   ({N(F)} neurons)", p + "ffn_gate.weight", R["x_ffn"], a[-1]); R["g"] = a[-1].copy()
    elif pt == "up":
        matmul_op(l, "ffn_up", f"u = W_up x   ({N(F)} neurons)", p + "ffn_up.weight", R["x_ffn"], a[-1]); R["u"] = a[-1].copy()
    elif pt == "act":
        act = a[-1]
        sc = np.abs(a).max(1)
        R["neur"]["sc"][l] = sc
        R["neur"]["q"][l] = np.round(a / np.maximum(sc, 1e-12)[:, None] * 127).astype(np.int8)
        if not OPT["compact"]:
            gg, uu = R["g"], R["u"]
            sil = gg / (1 + np.exp(-gg))
            strong = np.argsort(-np.abs(act))[:3]
            blk(f"[layer {l} | SwiGLU]  act = silu(g) x u  for each of {N(F)} neurons, silu(z) = z / (1+e^-z)\n"
                "    neuron        g          silu(g)         u           act\n" +
                "".join(f"    #{i:<6}  {f(gg[i])}   {f(sil[i])}   {f(uu[i])}   {f(act[i])}{'   <- strongest' if i in strong else ''}\n" for i in list(range(5)) + [int(s) for s in strong]) + "\n")
        R["x_act"] = act.copy()
    elif pt == "mlp_out":
        matmul_op(l, "ffn_down", f"mlp_out = W_down . act   ({N(F)} -> {D})", p + "ffn_down.weight", R["x_act"], a[-1]); R["mlp_out"] = a[-1].copy()
    elif pt == "resid_out":
        add_log(l, "2: resid_out = resid_mid + mlp_out", R["resid_mid"], R["mlp_out"], a[-1])
    elif pt == "final_norm":
        if R.get("layer_open"):
            put_close(); R["layer_open"] = False
        put_open("STEP 5 | AFTER ALL 36 LAYERS: final norm, then score every word in the vocabulary", True)
        rmsnorm_log("final", "(output_norm)", R["resid_final"], a, m.W["output_norm.weight"])
        R["x_final"] = a.copy()
    elif pt == "logits":
        top = np.argsort(-a)[:10]
        pr = np.exp(a - a.max()); pr /= pr.sum()
        best = int(top[0])
        R["neur"]["top5"] = [[tok.decode([int(t)]), float(pr[t])] for t in top[:5]]
        Wm = m.W["output.weight"]
        put(f"[lm_head]  logits = embedding_table . x_final     ({N(len(a))} dot products of length {D} = {N(HEAD)} multiply-adds)\n")
        R["done"] += HEAD
        put(f"    worked example for the highest-scoring token {best} = {tok.decode([best])!r}:  logit = (row {best} of the table) . x_final\n")
        put("\n".join(worked_example(Wm[best], R["x_final"], float(a[best]), 0.0, False)) + "\n\n")
        put("    top 10 scores (logit) and probability = softmax(logits):\n" +
            "".join(f"        id {int(t):<7} {tok.decode([int(t)])!r:18} logit {float(a[t]):+9.4f}   probability {float(pr[t]) * 100:8.4f}%  {'#' * int(round(float(pr[t]) * 40))}\n" for t in top))
        put(f"    forward pass #{R['fwd']} done. weight-matrix multiply-adds this pass: {N(R['done'])}\n")
        put_close(); put_close()
        step_done()


def hook_wrap(l, pt, a):
    if pt == "resid_out" and l == NL - 1:
        R["resid_final"] = a[-1].copy()
    hook(l, pt, a)


def run(prompt, template, max_new, temp):
    with lock:
        S["log"] = []; S["neur"] = []; S["epoch"] += 1; S["running"] = True
        S["meta"] = {"prompt": prompt, "temp": temp, "max_new": max_new, "template": template,
                     "compact": OPT["compact"], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    OPT["stop"] = False
    R.update(fwd=0, T=1, done=0, expected=0, ctx=[], pos0=0, full_left=OPT["full"])
    t0 = time.time()
    try:
        put_open("STEP 1 | YOUR TEXT", True)
        put(f"you typed: {prompt!r}\n")
        text = chatml([("system", SYSTEM), ("user", prompt)]) if template else prompt
        if template:
            put("\nThe model was trained on conversations in this exact format, so your text is wrapped (<|...|> are special marker tokens):\n" +
                "".join("    | " + ln + "\n" for ln in text.split("\n")))
        put_close()
        put_open("STEP 2 | TOKENISATION: text -> pieces -> integer ids (byte-pair encoding, merges listed)")
        lines, ids = bpe_explain(text)
        put("\n".join(lines) + "\n")
        assert ids == tok.encode(text)
        put(f"\nresult: {len(ids)} token ids\n" + "".join(f"    position {i:<3} id {t:<7} {tok.decode([t])!r}\n" for i, t in enumerate(ids)))
        put_close()
        out = ""
        for tid, piece in m.generate(ids, max_new=max_new, hook=hook_wrap, temp=temp):
            out += piece
            banner(f"STEP 6 | CHOOSE A WORD + DETOKENISE   ({'greedy: highest probability' if temp <= 0 else f'sampled with temperature {temp}'})", "-")
            bs = tok.token_bytes(tid)
            put(f"chosen token id {tid} -> table lookup -> bytes {' '.join(f'{b:02x}' for b in bs)} -> utf-8 text {piece!r}\n"
                f"reply so far: {out!r}\n")
            step_done()
        why = getattr(m, "stop_reason", "")
        if why.startswith("end"):
            banner("STEP 6 | STOP", "-")
            put(f"model chose token id {m.last_sampled} = {tok.decode([m.last_sampled])!r}  (end-of-turn marker) -> generation stops\n")
        banner(f"FINAL REPLY   ({len(ids)} prompt tokens in, {len(R['ctx']) - len(ids)} tokens generated, {time.time() - t0:.1f}s, stopped: {why})")
        put(out + "\n")
    except Abort:
        put("\n[stopped by user]\n")
    except Exception as e:
        import traceback; put("\n[ERROR]\n" + traceback.format_exc())
    finally:
        with lock:
            S["running"] = False


RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def render_items(items):
    out, depth, buf = [], 0, []

    def flush():
        if buf:
            out.append("<pre>" + html.escape("".join(buf)) + "</pre>"); buf.clear()
    for it in items:
        if isinstance(it, str):
            buf.append(it)
        elif "o" in it:
            flush(); out.append("<details%s><summary>%s</summary>" % (" open" if it["open"] else "", html.escape(it["o"]))); depth += 1
        else:
            flush(); out.append("</details>"); depth -= 1
    flush()
    return "".join(out) + "</details>" * max(0, depth)


def save_run(name=None):
    with lock:
        if S["running"]:
            return {"error": "run still in progress - wait until it finishes (or press Stop)"}
        meta, items = S.get("meta"), list(S["log"])
    if not meta or not items:
        return {"error": "nothing to save yet"}
    os.makedirs(RUNS_DIR, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", meta["prompt"].lower()).strip("-")[:40] or "run"
    clean = re.sub(r"[^A-Za-z0-9 _.-]+", "-", (name or "")).strip(" .-")[:80]
    path = os.path.abspath(os.path.join(RUNS_DIR, (clean or time.strftime("%Y-%m-%d_%H%M%S") + "_" + slug) + ".html"))
    if os.path.exists(path):
        path = path[:-5] + time.strftime("_%H%M%S") + ".html"
    head = chr(10).join([
        "saved: " + time.strftime("%Y-%m-%d %H:%M:%S"),
        "started: " + meta["started"],
        "prompt: " + repr(meta["prompt"]),
        "temperature: %s   max new tokens: %s   chat format: %s   compact: %s" % (meta["temp"], meta["max_new"], meta["template"], meta["compact"]),
        "model: backup copy of qwen2.5-3b-instruct-q4_k_m.gguf (read-only, unmodified)"])
    css = "body{font:13px/1.35 Consolas,monospace;margin:14px;background:#fff;color:#000}pre{white-space:pre;overflow-x:auto;margin:0;font:inherit}details{margin-left:1.2em}summary{cursor:pointer}"
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>Frost 1.0 run: " + html.escape(meta["prompt"][:60]) + "</title><style>" + css + "</style></head><body>"
            "<h3>Frost 1.0 (based on Qwen2.5-3B-Instruct, Q4_K_M) - full run log</h3><pre>" + html.escape(head) + "</pre>"
            "<p><button onclick=\"document.querySelectorAll('details').forEach(d=>d.open=true)\">expand all</button> "
            "<button onclick=\"document.querySelectorAll('details').forEach(d=>d.open=false)\">collapse all</button></p>"
            + render_items(items) + "</body></html>")
    open(path, "w", encoding="utf-8").write(page)
    return {"ok": True, "path": path, "mb": round(len(page) / 1e6, 2)}


# ---- weight browser --------------------------------------------------------
def tensors():
    out = []
    for n, t in g.tensors.items():
        o, i = (t.shape[1], t.shape[0]) if len(t.shape) == 2 else (1, t.shape[0])
        out.append([n, o, i, t.type_name, t.n_elements])
    return out


def weights(q):
    name = q["name"][0]
    t, Wm = g.tensors[name], m.W[name]
    W2 = Wm.reshape(1, -1) if Wm.ndim == 1 else Wm
    ni = lambda k, d: int(q.get(k, [d])[0])
    r0 = max(0, min(ni("r0", 0), W2.shape[0] - 1)); c0 = max(0, min(ni("c0", 0), W2.shape[1] - 1))
    nr, nc = min(ni("nr", 20), 60), min(ni("nc", 12), 40)
    rows = np.round(W2[r0:r0 + nr, c0:c0 + nc], 6).tolist()
    idx = r0 * W2.shape[1] + c0
    bi = idx // t.block_vals
    start = g.data_start + t.offset + bi * t.block_bytes
    raw = bytes(g.mm[start:start + t.block_bytes])
    info = f"block #{bi} (elements {bi * t.block_vals}..{(bi + 1) * t.block_vals - 1}), {t.block_bytes} bytes, file offset {start}"
    if t.type_name == "Q4_K":
        d, dm = struct.unpack_from("<ee", raw, 0)
        info += f"\nd={d:.6g}  dmin={dm:.6g}  scales(12 bytes, 6-bit packed)={list(raw[4:16])}\nthen 128 bytes = 256 x 4-bit codes;  value = d*scale*code - dmin*min"
    elif t.type_name == "Q6_K":
        info += f"\nd={struct.unpack_from('<e', raw, 208)[0]:.6g}  scales(16 x int8)={list(struct.unpack_from('<16b', raw, 192))}\n128 bytes low-4-bits + 64 bytes high-2-bits = 256 x 6-bit codes;  value = d*scale*(code-32)"
    hexs = "\n".join(f"{o:04x}  " + " ".join(f"{b:02x}" for b in raw[o:o + 16]) for o in range(0, len(raw), 16))
    return {"name": name, "shape": list(W2.shape), "qtype": t.type_name, "n": t.n_elements,
            "r0": r0, "c0": c0, "rows": rows, "info": info, "hex": hexs}


WS = {"state": "idle"}
PR = {"state": "idle", "results": [], "total": 0}
VOCAB = {}
FACTS = {}
if os.path.exists(scan.SCAN_FILE):
    WS.update(state="done", result=json.load(open(scan.SCAN_FILE, encoding="utf-8")))
if os.path.exists(scan.PROBES_FILE):
    PR.update(state="done", results=json.load(open(scan.PROBES_FILE, encoding="utf-8")))
    PR["total"] = len(PR["results"])


def start_wscan():
    if WS["state"] == "running":
        return
    threading.Thread(target=lambda: scan.weight_scan(m, WS), daemon=True).start()


def start_probes():
    if PR["state"] == "running" or S["running"]:
        return False
    threading.Thread(target=lambda: scan.run_probes(m, tok, PR), daemon=True).start()
    return True


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(open("log.html", "rb").read(), ctype="text/html; charset=utf-8")
            elif u.path == "/weights":
                self._send(open("weights.html", "rb").read(), ctype="text/html; charset=utf-8")
            elif u.path == "/capabilities":
                self._send(open("capabilities.html", "rb").read(), ctype="text/html; charset=utf-8")
            elif u.path == "/api/facts":
                if not FACTS:
                    FACTS.update(scan.facts(m, scan.ORIGINAL_PATH))
                FACTS["ram"].update(scan.mem_info())
                self._send(FACTS)
            elif u.path == "/api/vocab":
                if not VOCAB:
                    VOCAB.update(scan.vocab_scan(tok))
                self._send(VOCAB)
            elif u.path == "/api/wscan":
                if WS["state"] == "idle":
                    start_wscan()
                self._send({k: v for k, v in WS.items()})
            elif u.path == "/api/probes":
                self._send({"state": PR["state"], "results": PR["results"], "total": PR.get("total", 0), "busy_with_run": S["running"]})
            elif u.path == "/neurons":
                self._send(open("neurons.html", "rb").read(), ctype="text/html; charset=utf-8")
            elif u.path == "/api/neurons":
                with lock:
                    passes = list(S["neur"])
                if "pass" not in q:
                    self._send({"passes": [{"n": p["n"], "T": p["T"], "pos0": p["pos0"], "tokens": p["tokens"], "top5": p.get("top5")} for p in passes],
                                "layers": NL, "ffn": F, "running": S["running"]})
                else:
                    p = next(x for x in passes if x["n"] == int(q["pass"][0]))
                    pos = int(q["pos"][0])
                    self._send({"scales": [float(v) for v in p["sc"][:, pos]],
                                "d": [base64.b64encode(p["q"][l, pos].tobytes()).decode() for l in range(NL)]})
            elif u.path == "/api/tensors":
                self._send({"tensors": tensors(), "params": PARAMS, "per_layer": PER_LAYER, "head": HEAD, "layers": NL})
            elif u.path == "/api/weights":
                self._send(weights(q))
            elif u.path == "/api/log":
                since = int(q.get("since", [0])[0])
                with lock:
                    chunks, size = [], 0
                    for c in S["log"][since:]:
                        chunks.append(c); size += len(c) if isinstance(c, str) else 60
                        if size > 600_000:
                            break
                    r = {"chunks": chunks, "n": since + len(chunks), "running": S["running"], "epoch": S["epoch"]}
                self._send(r)
            else:
                self._send({"error": "not found"}, 404)
        except Exception as e:
            self._send({"error": repr(e)}, 500)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        u = urlparse(self.path).path
        if u == "/api/run":
            if S["running"]:
                return self._send({"error": "already running"}, 409)
            OPT.update(delay=max(0.0, float(body.get("delay", 0)) / 1000), full=int(body.get("full", 1)), compact=bool(body.get("compact", False)))
            threading.Thread(target=run, daemon=True, args=(
                body["prompt"], body.get("template", True), int(body.get("max_new", 12)), float(body.get("temp", 0)))).start()
            time.sleep(0.05)
            return self._send({"ok": True})
        if u == "/api/ctl":
            if "delay" in body: OPT["delay"] = max(0.0, float(body["delay"]) / 1000)
            if body.get("stop"): OPT["stop"] = True
            return self._send({"ok": True})
        if u == "/api/probes/start":
            return self._send({"ok": start_probes()})
        if u == "/api/save":
            return self._send(save_run(body.get("name")))
        self._send({"error": "not found"}, 404)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"READY  http://127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()
