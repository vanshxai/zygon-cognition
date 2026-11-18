"""Code side of the Neuron Path Locker: index the functions of the 10 programs and the neurons each one fires.

stage 1 (no model, seconds):   python code_index.py parse   -> functions, common vs unique, token ids   (model/path_locker/code_index.json + .npz)
stage 2 (model, ~25 min):      python code_index.py trace   -> for every function the neurons it fires more than average code does
For neuron (layer L, id i) the weights that belong to it are: blk.L.ffn_gate.weight[i, :], blk.L.ffn_up.weight[i, :] (rows) and blk.L.ffn_down.weight[:, i] (column).
"""
import hashlib, json, os, re, sys, time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
SW = os.path.join(ROOT, "software"); OUT = os.path.join(ROOT, "model", "path_locker")
os.makedirs(OUT, exist_ok=True)
PROGRAMS = [("01_hello", "hello.cs"), ("02_calculator", "calc.cs"), ("03_todo", "todo.cs"), ("04_tictactoe", "tictactoe.cs"), ("05_notepad", "notepad.cs"),
            ("06_diskusage", "diskusage.cs"), ("07_notes_server", "notes_server.cs"), ("08_taskboard", "taskboard.cs"), ("09_chat_server", "chat_server.cs"),
            ("10_miniwiki", "miniwiki.cs")]
SYS = "You are a helpful assistant."
FP_K = 48
MIN_TOK = 12
KW = {"return", "new", "else", "throw", "using", "lock", "if", "foreach", "while", "for", "switch", "catch", "case", "yield", "await", "do", "try", "var", "in", "out", "ref", "is", "as"}
SIG = re.compile(r"^\s*(?:(?:public|private|internal|protected|static|override|virtual|abstract|sealed|async|extern|unsafe)\s+)*([A-Za-z_][\w<>\[\],\.\?]*)\s+([A-Za-z_]\w*)\s*\(([^;]*)\)\s*(\{.*)?$")
JSF = re.compile(r"function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{")
JSV = re.compile(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\s*\([^)]*\)|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)\s*\{")


def match_brace(s, i, js=False):
    """index just after the } that closes the { at s[i]; skips strings, chars and comments"""
    d, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "/" and s.startswith("//", i):
            i = s.find("\n", i); i = n if i < 0 else i; continue
        if c == "/" and s.startswith("/*", i):
            i = s.find("*/", i + 2); i = n if i < 0 else i + 2; continue
        if c == "@" and s.startswith('@"', i):
            i += 2
            while i < n:
                if s[i] == '"':
                    if s.startswith('""', i): i += 2; continue
                    break
                i += 1
            i += 1; continue
        if c == '"' or (c == "'" or (js and c == "`")):
            q = c; i += 1
            while i < n and s[i] != q:
                if s[i] == "\\": i += 1
                elif s[i] == "\n" and q != "`": break
                i += 1
            i += 1; continue
        if c == "{": d += 1
        elif c == "}":
            d -= 1
            if d == 0: return i + 1
        i += 1
    return -1


def parse_program(src):
    """-> list of {name, lang, start, end (char offsets), sig}"""
    out = []
    a, z = src.find("// <<WEB-BEGIN>>"), src.find("// <<WEB-END>>")
    web = (a, z) if a >= 0 and z > a else (len(src), len(src))
    # ---- C# methods (outside the embedded web page)
    pos = 0
    for line in src.split("\n"):
        ls, le = pos, pos + len(line); pos = le + 1
        if web[0] <= ls < web[1]: continue
        m = SIG.match(line)
        if not m or m.group(1) in KW or line.rstrip().endswith(";") or "=>" in line or "=" in line.split("(")[0]: continue
        b = src.find("{", ls) if m.group(4) is None else ls + line.index("{")
        if b < 0: continue
        if m.group(4) is None:                                          # signature alone on the line: the { must be the very next thing
            if src[le:b].strip(): continue
        e = match_brace(src, b)
        if e > 0: out.append({"name": m.group(2), "lang": "csharp", "start": ls, "end": e, "sig": line.strip()[:120]})
    # ---- JavaScript functions inside the embedded page
    if web[0] < web[1]:
        page = src[web[0]:web[1]]
        for rx in (JSF, JSV):
            for m in rx.finditer(page):
                b = web[0] + m.end() - 1
                e = match_brace(src, b, js=True)
                if e > 0 and e <= web[1]:
                    ls = src.rfind("\n", 0, web[0] + m.start()) + 1
                    out.append({"name": m.group(1), "lang": "javascript", "start": web[0] + m.start(), "end": e, "sig": src[web[0] + m.start():b].strip()[:120]})
    out.sort(key=lambda f: (f["start"], -f["end"]))
    return out


def load_tokenizer():
    from gguf import GGUF
    from tokenizer import Tokenizer
    return Tokenizer(GGUF().meta)


def parse():
    tok = load_tokenizer()
    fns, toks = [], []
    for folder, fn in PROGRAMS:
        src = open(os.path.join(SW, folder, fn), encoding="utf-8").read().replace("\r\n", "\n")
        for f in parse_program(src):
            body = src[f["start"]:f["end"]]
            if f["lang"] == "csharp":                                   # take the full line(s) the signature starts on
                body = src[src.rfind("\n", 0, f["start"]) + 1 if False else f["start"]:f["end"]]
            ids = tok.encode(body)
            norm = re.sub(r"\s+", " ", body).strip()
            f.update(program=folder, file=fn, lines=body.count("\n") + 1, chars=len(body), tokens_n=len(ids), body_hash=hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12], text=body)
            fns.append(f); toks.append(np.array(ids, np.int32))
    # common = same name in 2+ programs; identical = same normalised body in 2+ programs
    by_name, by_hash = {}, {}
    for f in fns:
        by_name.setdefault((f["lang"], f["name"]), set()).add(f["program"])
        by_hash.setdefault(f["body_hash"], set()).add(f["program"])
    for i, f in enumerate(fns):
        f["id"] = i
        progs = by_name[(f["lang"], f["name"])]
        f["kind"] = "common" if len(progs) >= 2 else "unique"
        f["shared_with"] = sorted(progs - {f["program"]})
        f["identical_in"] = sorted(by_hash[f["body_hash"]] - {f["program"]})
    off = np.concatenate([[0], np.cumsum([len(t) for t in toks])]).astype(np.int64)
    json.dump({"functions": fns, "programs": [p for p, _ in PROGRAMS]}, open(os.path.join(OUT, "code_index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(OUT, "code_index.npz"), tokens=np.concatenate(toks), offsets=off)
    common = sum(f["kind"] == "common" for f in fns)
    print(f"{len(fns)} functions in {len(PROGRAMS)} programs: {common} common (same name in 2+ programs), {len(fns) - common} unique;  {int(off[-1]):,} tokens")
    names = {}
    for f in fns:
        if f["kind"] == "common": names.setdefault(f["name"], set()).add(f["program"])
    print("most common:", sorted(((len(v), k) for k, v in names.items()), reverse=True)[:12])


def trace():
    from engine import Model
    t0 = time.time()
    J = json.load(open(os.path.join(OUT, "code_index.json"), encoding="utf-8")); fns = J["functions"]
    m = Model(verbose=False, patch=False)
    NL, F = m.n_layers, m.W["blk.0.ffn_up.weight"].shape[0]
    end = m.tok.id_of["<|im_end|>"]
    SUM = np.zeros((len(fns), NL, F), np.float32); CNT = np.zeros(len(fns), np.int64)
    ALL = np.zeros((NL, F), np.float64); NALL = 0
    for folder, fn in PROGRAMS:
        src = open(os.path.join(SW, folder, fn), encoding="utf-8").read().replace("\r\n", "\n")
        head = "```csharp\n"
        p = m.tok.encode(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nWrite a complete, working Windows program in C#. Give the full source code.<|im_end|>\n<|im_start|>assistant\n")
        r = m.tok.encode(head + src + "\n```") + [end]
        ids, s0 = p + r, len(p)
        # token -> function (smallest enclosing function)
        spans = sorted([(f["id"], len(head.encode()) + len(src[:f["start"]].encode()), len(head.encode()) + len(src[:f["end"]].encode())) for f in fns if f["program"] == folder], key=lambda x: x[2] - x[1])
        tf, off = np.full(len(r), -1, np.int32), 0
        for i, t in enumerate(r):
            nb = len(m.tok.token_bytes(t)); mid = off + nb // 2; off += nb
            for fid, a, b in spans:
                if a <= mid < b: tf[i] = fid; break
        st = {"c0": 0}

        def hook(l, pt, a):
            nonlocal NALL
            if pt != "act": return
            g = st["c0"] + np.arange(len(a)) - s0
            ok = g >= 0
            if not ok.any(): return
            ab = np.abs(a[ok]); gg = g[ok]
            ALL[l] += ab.sum(0)
            if l == 0: NALL += int(ok.sum())
            fid = tf[gg]
            for q in np.unique(fid):
                if q < 0: continue
                SUM[q, l] += ab[fid == q].sum(0)
                if l == 0: CNT[q] += int((fid == q).sum())
        t1 = time.time()
        cache = m.new_cache(len(ids) + 4)
        for c0 in range(0, len(ids), 512):
            st["c0"] = c0
            m.forward(ids[c0:c0 + 512], cache, hook)
        print(f"  {folder:16s} {len(ids):>6,} tokens  {time.time() - t1:.0f}s", flush=True)
    mu_all = (ALL / max(NALL, 1)).astype(np.float32)
    ids_a = np.zeros((len(fns), NL, FP_K), np.uint16); lift_a = np.zeros((len(fns), NL, FP_K), np.float16); ok = CNT >= MIN_TOK
    for f in range(len(fns)):
        if not ok[f]: continue
        mu = SUM[f] / CNT[f]
        d = mu - mu_all
        top = np.argsort(-d, axis=1)[:, :FP_K]
        ids_a[f] = top; lift_a[f] = np.take_along_axis(d, top, 1)
    # does the same function name light up the same neurons in different programs?
    def ov(a, b):
        return float(np.mean([len(set(ids_a[a, l]) & set(ids_a[b, l])) / FP_K for l in range(NL)]))
    rng = np.random.default_rng(0)
    same, diff = [], []
    byn = {}
    for f in fns:
        if ok[f["id"]]: byn.setdefault((f["lang"], f["name"]), []).append(f["id"])
    for k, v in byn.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)): same.append(ov(v[i], v[j]))
    good = [f["id"] for f in fns if ok[f["id"]]]
    for _ in range(1500):
        a, b = rng.choice(good, 2, replace=False)
        if (fns[a]["lang"], fns[a]["name"]) != (fns[b]["lang"], fns[b]["name"]): diff.append(ov(a, b))
    res = {"same_name_overlap": float(np.mean(same)) if same else None, "same_name_pairs": len(same), "different_name_overlap": float(np.mean(diff)), "chance": FP_K / F,
           "traced_functions": int(ok.sum()), "min_tokens": MIN_TOK, "seconds": round(time.time() - t0)}
    for f in fns:
        i = f["id"]; f["traced"] = bool(ok[i]); f["tokens_traced"] = int(CNT[i])
        if ok[i]:
            lay = (lift_a[i].astype(np.float32).sum(1)); f["top_layers"] = [int(x) for x in np.argsort(-lay)[:5]]
            flat = [(float(lift_a[i, l, k]), l, int(ids_a[i, l, k])) for l in range(NL) for k in range(5)]
            flat.sort(reverse=True); f["top_neurons"] = [{"layer": l, "neuron": n, "lift": round(v, 3)} for v, l, n in flat[:8]]
    J["trace"] = res
    json.dump(J, open(os.path.join(OUT, "code_index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    z = np.load(os.path.join(OUT, "code_index.npz"))
    np.savez_compressed(os.path.join(OUT, "code_index.npz"), tokens=z["tokens"], offsets=z["offsets"], fp_ids=ids_a, fp_lift=lift_a, traced=ok)
    print("TRACE DONE", json.dumps(res), flush=True)


if __name__ == "__main__":
    {"parse": parse, "trace": trace}[sys.argv[1]]()
