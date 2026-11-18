"""Experiment: the "exponent + tangent-shot" memory, tested on a real 9-page PDF (bitcoin.pdf).

  part 1  does writing a number as e^x compress it?           (exact arithmetic with Python's decimal module)
  part 2  learn the PDF: sentences -> numbers -> e^x text store -> 3D positions (r = e^x, angle theta, angle phi)
          retrieve with the 3D positions only, compare against the usual methods
  part 3  data for the 3D page (exp_store_3d.html)
python exp_store.py
"""
import json, math, os, re, sys, time
from decimal import Decimal, getcontext
import numpy as np
from gguf import GGUF, np_tensor
from tokenizer import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
EXP = os.path.join(ROOT, "model", "exp_store")
RES = {}


# =============================================================================== part 1: does e^x compress?
def part1():
    getcontext().prec = 420
    import random
    random.seed(1)
    digits = "".join(random.choice("0123456789") for _ in range(300))
    v = Decimal("1." + digits); x = v.ln()
    rows = []
    for d in (3, 6, 10, 20, 50, 100, 200, 300):
        xs = x.quantize(Decimal(1).scaleb(-d)); vv = xs.exp()
        a, b = str(v)[2:], str(vv)[2:]; ok = 0
        while ok < len(a) and ok < len(b) and a[ok] == b[ok]: ok += 1
        rows.append({"x_digits": d, "x_text_chars": len(str(xs)), "correct_digits_of_v": ok})
    big = Decimal(10) ** 7 * Decimal(10).ln()               # 10^(10,000,000) = e^big
    rows2 = []
    for d in (0, 2, 4, 6, 8, 10, 14, 20):
        xs = big.quantize(Decimal(1).scaleb(-d)); rel = ((xs - big).exp() - 1).copy_abs()
        corr = int(-rel.log10()) if rel > 0 else 999
        rows2.append({"x_text": "e^" + str(xs), "bytes": len("e^" + str(xs)), "correct_leading_digits": corr})
    RES["part1"] = {"random_300_digit_number": {"v_text_chars": len(str(v)), "rows": rows}, "ten_to_ten_million": {"exact_exponent": str(big)[:34], "total_digits_of_number": 10_000_001, "rows": rows2}}
    print("part 1: a random 300-digit number: digits of x needed vs correct digits of v:")
    for r in rows: print("   ", r)
    print("part 1: 10^10,000,000 = e^x:")
    for r in rows2: print("   ", r)


# =============================================================================== part 2: learn the PDF
STOP_AFTER = "References"
HEAD = re.compile(r"^(\d{1,2})\.\s+([A-Z][A-Za-z\- ]+)$")
WORD = re.compile(r"[a-z0-9][a-z0-9\-']*")


def load_sentences():
    lines = open(os.path.join(EXP, "bitcoin.txt"), encoding="utf-8").read().split("\n")
    secs = [("Abstract", [])]; begun = False
    for ln in lines:
        s = ln.strip()
        if s == STOP_AFTER: break
        if not begun:                                                  # skip title / author / e-mail lines
            if s.startswith("Abstract"): begun = True; secs[0][1].append(s)
            continue
        m = HEAD.match(s)
        if m: secs.append((f"{m.group(1)}. {m.group(2).strip()}", [])); continue
        secs[-1][1].append(s)
    out = []
    for si, (name, body) in enumerate(secs):
        text = re.sub(r"\s+", " ", " ".join(body).replace("-\n", "")).strip()
        text = re.sub(r"^Abstract\.\s*", "", text)
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", text):
            w = WORD.findall(s.lower()); letters = sum(c.isalpha() for c in s)
            if len(w) >= 6 and letters >= 0.6 * len(s): out.append({"sec": si, "text": s})
    return [n for n, _ in secs], out


def words_of(s):
    return WORD.findall(s.lower())


def part2():
    t_start = time.time()
    secs, sents = load_sentences()
    N = len(sents)
    print(f"part 2: {N} sentences in {len(secs)} sections; section sizes {[sum(s['sec'] == i for s in sents) for i in range(len(secs))]}", flush=True)
    g = GGUF(); tok = Tokenizer(g.meta)
    E = np_tensor(g, "token_embd.weight").astype(np.float32)             # (151936, 2048): one row per word-piece
    vocab = sorted({w for s in sents for w in words_of(s["text"])})
    df = {w: 0 for w in vocab}
    for s in sents:
        for w in set(words_of(s["text"])): df[w] += 1
    idf = {w: math.log((N + 1) / (df[w] + 0.5)) for w in vocab}
    W = np.zeros((len(vocab), E.shape[1]), np.float32)                      # word -> mean of its word-piece embeddings
    for i, w in enumerate(vocab):
        W[i] = E[tok.encode(" " + w)].mean(0)
    widx = {w: i for i, w in enumerate(vocab)}
    del E

    def svec(text):
        ws = [w for w in words_of(text) if w in widx]
        if not ws: return None
        a = np.array([idf[w] for w in ws], np.float32); return (a[:, None] * W[[widx[w] for w in ws]]).sum(0) / a.sum()
    S = np.stack([svec(s["text"]) for s in sents]); mu = S.mean(0)
    secid = np.array([s["sec"] for s in sents])
    t_learn_embed = time.time() - t_start
    # PCA: the 3D (and k-D) frame
    U, sv, Vt = np.linalg.svd(S - mu, full_matrices=False)
    P = Vt.T                                                                   # (2048, n) directions, strongest first
    expl = (sv ** 2) / (sv ** 2).sum()
    W3 = (W - mu) @ P[:, :3]; S3 = (S - mu) @ P[:, :3]
    sc = float(np.percentile(np.abs(S3), 95)); W3n, S3n = W3 / sc, S3 / sc
    # --- polar form of every sentence position: r = e^(lnr), tan(theta) = y/x, phi = elevation
    r = np.linalg.norm(S3n, axis=1) + 1e-9; theta = np.arctan2(S3n[:, 1], S3n[:, 0]); phi = np.arctan2(S3n[:, 2], np.hypot(S3n[:, 0], S3n[:, 1]))
    lnr = np.log(r)

    # --- the e^x text store: every one of the 2048 numbers of a sentence = sign + ln|value|, quantised
    A = np.abs(S); nz = A[A > 0]
    xmin, xmax = float(np.log(np.percentile(nz, 0.5))), float(np.log(nz.max()))

    def enc_log(v, bits):
        top = (1 << (bits - 1)) - 1                                            # codes 1..top, 0 = "zero"
        x = np.log(np.maximum(np.abs(v), 1e-30)); c = np.clip(np.round((x - xmin) / (xmax - xmin) * (top - 1)) + 1, 1, top).astype(np.int64)
        c[np.abs(v) < math.exp(xmin) * 0.5] = 0
        return c | ((v < 0).astype(np.int64) << (bits - 1))

    def dec_log(c, bits):
        top = (1 << (bits - 1)) - 1; neg = (c >> (bits - 1)) & 1; m = c & top
        val = np.exp(xmin + (m - 1) / (top - 1) * (xmax - xmin)); val[m == 0] = 0
        return np.where(neg == 1, -val, val).astype(np.float32)
    C8 = np.stack([enc_log(v, 8) for v in S]).astype(np.uint8); C16 = np.stack([enc_log(v, 16) for v in S]).astype(np.uint16)
    # text store file (real text bytes) and read it back
    q16 = lambda a, lo, hi: np.clip(np.round((a - lo) / (hi - lo) * 65535), 0, 65535).astype(int)
    lo_r, hi_r = float(lnr.min()), float(lnr.max())
    lines = ["# EXP-STORE v1  (text).  decoder for the 2048 numbers: value = (-1 if code>=128 else 1) * exp(xmin + (code%128 - 1)/126 * (xmax - xmin)); code 0 = zero",
             f"# xmin={xmin:.6f} xmax={xmax:.6f}   position decoder: P = e^lnr * (cos(phi)cos(theta), cos(phi)sin(theta), sin(phi)); lnr in [{lo_r:.6f},{hi_r:.6f}] theta in [-pi,pi] phi in [-pi/2,pi/2] as 16-bit codes"]
    for i, s in enumerate(sents):
        lines.append(f"S{i}|sec{s['sec']}|{q16(lnr[i], lo_r, hi_r):04x}{q16(theta[i], -math.pi, math.pi):04x}{q16(phi[i], -math.pi / 2, math.pi / 2):04x}|next S{i + 1 if i + 1 < N else i}|{s['text']}|{C8[i].tobytes().hex()}")
    store_path = os.path.join(EXP, "store.txt"); open(store_path, "w", encoding="utf-8").write("\n".join(lines))
    # ---- read the file back and rebuild everything from the text
    back = open(store_path, encoding="utf-8").read().split("\n")[2:]
    S_back = np.zeros_like(S); pos_back = np.zeros((N, 3), np.float32)
    for i, ln in enumerate(back):
        f = ln.split("|"); code = bytes.fromhex(f[-1]); S_back[i] = dec_log(np.frombuffer(code, np.uint8).astype(np.int64), 8)
        h = f[2]; a, b, c = (int(h[k:k + 4], 16) / 65535 for k in (0, 4, 8))
        lr_, th_, ph_ = lo_r + a * (hi_r - lo_r), -math.pi + b * 2 * math.pi, -math.pi / 2 + c * math.pi
        rr = math.exp(lr_); pos_back[i] = [rr * math.cos(ph_) * math.cos(th_), rr * math.cos(ph_) * math.sin(th_), rr * math.sin(ph_)]
    pos_err = float(np.abs(pos_back - S3n).max())
    S_16 = np.stack([dec_log(c, 16) for c in C16.astype(np.int64)])
    # sizes
    dec_txt = float(np.mean([sum(len(f"{x:.8g}") + 1 for x in v) for v in S[:20]])); ex_txt = float(np.mean([sum(len(f"e^{math.log(abs(x) + 1e-30):.6f}") + 2 for x in v) for v in S[:20]]))
    lin8 = np.stack([np.round(v / np.abs(v).max() * 127) / 127 * np.abs(v).max() for v in S]).astype(np.float32)
    cosv = lambda a, b: float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-20)))
    rel = lambda a, b: float(np.median(np.abs(a - b)[np.abs(b) > 0] / np.abs(b)[np.abs(b) > 0]))
    size = {"numbers_per_sentence": int(S.shape[1]), "float32_bytes": 4 * S.shape[1], "float64_bytes": 8 * S.shape[1], "decimal_text_8digits_bytes": round(dec_txt), "exponent_text_e^x_6decimals_bytes": round(ex_txt),
            "log8_binary_bytes": S.shape[1], "log8_hex_text_bytes": 2 * S.shape[1], "log16_binary_bytes": 2 * S.shape[1], "position_only_bytes_binary": 6,
            "cos_restored_vs_original": {"log8": cosv(S_back, S), "log16": cosv(S_16, S), "linear_int8": cosv(lin8, S)},
            "median_relative_error": {"log8": rel(S_back, S), "log16": rel(S_16, S), "linear_int8": rel(lin8, S)}, "store_file_bytes": os.path.getsize(store_path),
            "text_of_sentences_bytes": sum(len(s["text"].encode()) for s in sents), "position_roundtrip_max_error": pos_err}
    RES["sizes"] = size
    print("part 2: sizes", json.dumps(size, indent=1), flush=True)

    # --- retrieval test: queries made from the sentences themselves
    rng = np.random.default_rng(7); queries = []
    for i, s in enumerate(sents):
        w = words_of(s["text"])
        if len(w) < 10: continue
        a = int(rng.integers(0, len(w) - 5)); queries.append(("6 consecutive words", " ".join(w[a:a + 6]), i))
        queries.append(("4 random keywords", " ".join(rng.choice(w, 4, replace=False)), i))
    allq = np.stack([svec(q[1]) if svec(q[1]) is not None else np.zeros(S.shape[1], np.float32) for q in queries]); Qc = allq - mu
    Sc, S8c, S16c = S - mu, S_back - mu, S_16 - mu

    def cos_rank(Qm, Sm):
        a = Qm / (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-12); b = Sm / (np.linalg.norm(Sm, axis=1, keepdims=True) + 1e-12); return np.argsort(-(a @ b.T), axis=1)

    def euc_rank(Qm, Sm):
        d = ((Qm[:, None, :] - Sm[None, :, :]) ** 2).sum(-1); return np.argsort(d, axis=1)
    inv = {}
    for i, s in enumerate(sents):
        for w in set(words_of(s["text"])): inv.setdefault(w, []).append(i)

    def sparse_rank(q):
        sc_ = np.zeros(N)
        for w in set(words_of(q)):
            for i in inv.get(w, []): sc_[i] += idf[w]
        return np.argsort(-sc_)
    methods = {"plain word index (idf, no neural numbers)": np.stack([sparse_rank(q[1]) for q in queries]), "all 2048 numbers, cosine": cos_rank(Qc, Sc),
               "2048 numbers stored as e^x 8-bit text": cos_rank(Qc, S8c), "2048 numbers stored as e^x 16-bit": cos_rank(Qc, S16c)}
    for k in (128, 32, 8, 3):
        methods[f"only {k} numbers (PCA), cosine"] = cos_rank(Qc @ P[:, :k], Sc @ P[:, :k])
    Q3 = Qc @ P[:, :3] / sc
    methods["3D position store (shot to nearest point)"] = euc_rank(Q3, S3n)
    rows = []
    for name, rk in methods.items():
        top1 = float(np.mean([rk[j][0] == q[2] for j, q in enumerate(queries)])); top5 = float(np.mean([q[2] in rk[j][:5] for j, q in enumerate(queries)]))
        sec1 = float(np.mean([secid[rk[j][0]] == secid[q[2]] for j, q in enumerate(queries)]))
        by = {}
        for kind in ("6 consecutive words", "4 random keywords"):
            idx = [j for j, q in enumerate(queries) if q[0] == kind]; by[kind] = float(np.mean([rk[j][0] == queries[j][2] for j in idx]))
        rows.append({"method": name, "top1": top1, "top5": top5, "right_section_top1": sec1, "top1_by_query_type": by})
    RES["retrieval"] = {"n_queries": len(queries), "n_sentences": N, "chance_top1": 1 / N, "chance_section": float(sum((np.mean(secid == i)) ** 2 for i in range(len(secs)))), "rows": rows}
    print("retrieval:", flush=True)
    for r in rows: print(f"   {r['method']:48s} top1 {100 * r['top1']:5.1f}%  top5 {100 * r['top5']:5.1f}%  right section {100 * r['right_section_top1']:5.1f}%", flush=True)

    # --- do similar topics sit together in 3D?  nearest neighbour (other sentence) in the same section
    def nn_same(Xm, metric):
        rk = euc_rank(Xm, Xm) if metric == "euc" else cos_rank(Xm, Xm)
        return float(np.mean([secid[rk[i][1]] == secid[i] for i in range(N)]))
    RES["clusters"] = {"nn_same_section_3D": nn_same(S3n, "euc"), "nn_same_section_2048": nn_same(Sc, "cos"), "chance": RES["retrieval"]["chance_section"],
                       "pca_variance_first3": [float(e) for e in expl[:3]], "pca_variance_first10_sum": float(expl[:10].sum())}
    print("clusters:", RES["clusters"], flush=True)
    RES["timing"] = {"learn_seconds": round(time.time() - t_start, 1), "embed_seconds": round(t_learn_embed, 1)}

    # ------------------------------------------------------------------ data for the 3D page
    pts = []
    for i, s in enumerate(sents):
        pts.append({"i": i, "sec": s["sec"], "p": [round(float(a), 4) for a in S3n[i]], "lnr": round(float(lnr[i]), 4), "th": round(float(theta[i]), 4), "ph": round(float(phi[i]), 4), "text": s["text"], "nw": len(words_of(s["text"])), "ex8": [("-" if v < 0 else "+") + "e^" + f"{math.log(abs(v) + 1e-30):.3f}" for v in S[i][:8]]})
    cents = []
    for si, name in enumerate(secs):
        m = secid == si
        cents.append({"name": name, "n": int(m.sum()), "c": [round(float(a), 3) for a in S3n[m].mean(0)] if m.any() else [0, 0, 0], "first": next((s["text"] for s in sents if s["sec"] == si), "")})
    words = {w: [round(float(a), 4) for a in W3n[i]] + [round(idf[w], 3)] for w, i in widx.items()}
    data = {"sections": secs, "points": pts, "centers": cents, "words": words, "results": RES, "decoder": {"xmin": xmin, "xmax": xmax}, "pdf": {"name": "bitcoin.pdf", "source": "https://bitcoin.org/bitcoin.pdf", "bytes": os.path.getsize(os.path.join(EXP, "bitcoin.pdf")), "pages": 9}}
    tpl = open(os.path.join(HERE, "exp_store_template.html"), encoding="utf-8").read() if os.path.exists(os.path.join(HERE, "exp_store_template.html")) else None
    json.dump(data, open(os.path.join(EXP, "viz_data.json"), "w", encoding="utf-8"), ensure_ascii=False)
    if tpl:
        open(os.path.join(ROOT, "exp_store_3d.html"), "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<" + chr(92) + "/")))
        print("written exp_store_3d.html", flush=True)


if __name__ == "__main__":
    part1(); part2()
    json.dump(RES, open(os.path.join(EXP, "results.json"), "w", encoding="utf-8"), indent=1)
