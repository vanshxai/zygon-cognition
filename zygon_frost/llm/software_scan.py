"""Which neurons fire when the model reads/writes 10 real, working programs (smallest -> biggest)?  Full CPU, original weights.

For each program the model is given "Write <description> as a Windows program in C#" and the program's actual source code as the reply
(teacher-forced). For every reply token we record which MLP neurons are triggered (>= 10% of the layer's loudest neuron).
Tokens are labelled C# / HTML / CSS / JavaScript (the web programs embed a whole web page in the C# file).
Outputs: model/frost_work/software_scan.json, software_sets.npz and ../software_neuron_map.html
"""
import base64, json, os, sys, time
import numpy as np
from engine import Model

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(ROOT, "model", "frost_work"); SW = os.path.join(ROOT, "software")
SYS = "You are a helpful assistant."
PROGRAMS = [("01_hello", "hello.cs", "a console program that prints a greeting"), ("02_calculator", "calc.cs", "a command-line calculator"),
            ("03_todo", "todo.cs", "a console todo list that saves its tasks to a file"), ("04_tictactoe", "tictactoe.cs", "a tic-tac-toe game against an unbeatable computer"),
            ("05_notepad", "notepad.cs", "a Windows text editor with menus, find and save"), ("06_diskusage", "diskusage.cs", "a Windows disk usage analyzer with a graphical window"),
            ("07_notes_server", "notes_server.cs", "a full-stack notes web app: HTTP backend with a JSON API, an HTML front-end and file storage"),
            ("08_taskboard", "taskboard.cs", "a full-stack Kanban task board web app with a REST API, drag-and-drop front-end, search and CSV export"),
            ("09_chat_server", "chat_server.cs", "a full-stack multi-room chat web app with long polling, rate limiting and saved history"),
            ("10_miniwiki", "miniwiki.cs", "a full-stack wiki platform with accounts, sessions, page versions, markdown, search and an admin page")]
SEGS = ["csharp", "html", "css", "javascript"]


def segments(src):
    """per-byte label array (0 csharp, 1 html, 2 css, 3 javascript) for the source text"""
    b = src.encode("utf-8"); lab = np.zeros(len(b), np.int8)
    a, z = b.find(b"// <<WEB-BEGIN>>"), b.find(b"// <<WEB-END>>")
    if a < 0 or z < 0:
        return lab
    lab[a:z] = 1
    for tag, code in ((b"<style>", 2), (b"<script>", 3)):
        s = a
        while True:
            i = b.find(tag, s, z)
            if i < 0: break
            j = b.find(b"</" + tag[1:], i, z)
            if j < 0: break
            lab[i:j] = code; s = j
    return lab


def pack(a, dt):
    return base64.b64encode(np.ascontiguousarray(a, dt).tobytes()).decode()


def main():
    t0 = time.time()
    m = Model(verbose=False, patch=False)
    NL, F = m.n_layers, m.W["blk.0.ffn_up.weight"].shape[0]
    print(f"model loaded {time.time() - t0:.0f}s", flush=True)
    end = m.tok.id_of["<|im_end|>"]
    prog = []                       # per-program results
    SUM = [np.zeros((len(SEGS), NL, F), np.float32) for _ in PROGRAMS]; CNT = [np.zeros(len(SEGS), np.int64) for _ in PROGRAMS]
    for k, (folder, fn, desc) in enumerate(PROGRAMS):
        src = open(os.path.join(SW, folder, fn), encoding="utf-8").read().replace("\r\n", "\n")
        head = "```csharp\n"
        p = m.tok.encode(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nWrite {desc} as a complete, working Windows program in C#. Give the full source code.<|im_end|>\n<|im_start|>assistant\n")
        r = m.tok.encode(head + src + "\n```") + [end]
        ids, s0 = p + r, len(p)
        # token -> segment (by byte offsets inside the reply)
        lab = np.concatenate([np.zeros(len(head.encode()), np.int8), segments(src), np.zeros(5, np.int8)])
        off, seg = 0, []
        for t in r:
            nb = len(m.tok.token_bytes(t)); mid = min(off + nb // 2, len(lab) - 1)
            seg.append(int(lab[mid]) if off < len(lab) else 0); off += nb
        seg = np.array(seg, np.int8)
        union = np.zeros((NL, F), bool); per_tok = np.zeros(len(ids), np.int64)
        st = {"c0": 0}

        def hook(l, pt, a):
            if pt != "act":
                return
            ab = np.abs(a)
            mask = ab >= 0.1 * ab.max(1, keepdims=True)
            g = st["c0"] + np.arange(len(a))
            per_tok[g] += mask.sum(1)
            keep = g >= s0
            if not keep.any():
                return
            union[l] |= mask[keep].any(0)
            sg = seg[g[keep] - s0]
            abk = ab[keep]
            for s in range(len(SEGS)):
                rows = sg == s
                if rows.any():
                    SUM[k][s, l] += abk[rows].sum(0)
        t1 = time.time()
        cache = m.new_cache(len(ids) + 4)
        for c0 in range(0, len(ids), 512):
            st["c0"] = c0
            m.forward(ids[c0:c0 + 512], cache, hook)
        for s in range(len(SEGS)):
            CNT[k][s] = int((seg == s).sum())
        n_reply = len(r)
        res = {"k": k + 1, "program": folder, "description": desc, "source_lines": src.count("\n") + 1, "source_chars": len(src), "reply_tokens": n_reply,
               "tokens_by_part": {SEGS[s]: int(CNT[k][s]) for s in range(len(SEGS))}, "avg_triggered_per_token": float(per_tok[s0:].mean()),
               "union_triggered": int(union.sum()), "seconds": round(time.time() - t1, 1)}
        prog.append(res); res["_union"] = union
        print(f"  {k + 1}/10 {folder:16s} {res['source_lines']:>4} lines {n_reply:>6,} tokens | avg {res['avg_triggered_per_token']:>7,.0f} per token | union {res['union_triggered']:>7,} ({100 * res['union_triggered'] / (NL * F):.0f}%) | {res['seconds']}s", flush=True)

    # ------------------------------------------------ neuron sets
    z = np.load(os.path.join(WORK, "text_code_means.npz"))
    mu_text = z["neur_text"].mean(0)                                  # earlier text replies (10)
    ts = json.load(open(os.path.join(WORK, "text_scan.json")))
    text_set = np.zeros((NL, F), bool)
    for l in range(NL):
        text_set[l, ts["text_neuron_ids"][str(l)]] = True
    ca = np.load(os.path.join(WORK, "category_assignment.npz"), allow_pickle=True)
    code_cat_set = ca["assigned"] == list(ca["cats"]).index("Code") if "Code" in list(ca["cats"]) else np.zeros((NL, F), bool)

    def mean_act(idx):                                                # mean |act| over all code tokens of the chosen programs
        s = sum(SUM[i].sum(0) for i in idx); n = sum(int(CNT[i].sum()) for i in idx)
        return s / max(n, 1)
    allp = list(range(10))
    mu_code = mean_act(allp)
    q75 = np.quantile(mu_code, 0.75, axis=1, keepdims=True)
    sel = (mu_code - mu_text) / (mu_code + mu_text + 1e-9)
    stable = np.ones_like(sel, bool)
    for half in ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]):
        mh = mean_act(half)
        stable &= ((mh - mu_text) / (mh + mu_text + 1e-9)) >= 0.5
    coding = (sel >= 0.5) & stable & (mu_code >= q75)
    web = [6, 7, 8, 9]
    seg_sets = {}
    mus = {}
    for s in range(len(SEGS)):
        n = sum(int(CNT[i][s]) for i in web)
        mus[SEGS[s]] = (sum(SUM[i][s] for i in web) / max(n, 1)) if n >= 200 else None
    for s in SEGS:
        if mus[s] is None: continue
        others = [mus[o] for o in SEGS if o != s and mus[o] is not None]
        rest = np.mean(others, 0)
        seg_sets[s] = ((mus[s] - rest) / (mus[s] + rest + 1e-9) >= 0.5) & (mus[s] >= np.quantile(mus[s], 0.75, axis=1, keepdims=True)) & coding
    tot = NL * F
    out = {"programs": [], "coding_neurons": int(coding.sum()), "text_reply_set": int(text_set.sum()), "code_category_set_earlier": int(code_cat_set.sum()),
           "overlap_coding_text": int((coding & text_set).sum()), "overlap_coding_code_category": int((coding & code_cat_set).sum()),
           "segment_sets": {s: int(v.sum()) for s, v in seg_sets.items()},
           "segment_overlap": {f"{a}&{b}": int((seg_sets[a] & seg_sets[b]).sum()) for i, a in enumerate(seg_sets) for b in list(seg_sets)[i + 1:]},
           "coding_by_layer": [int(coding[l].sum()) for l in range(NL)], "total_neurons": tot,
           "union_all_programs": int(np.logical_or.reduce([r["_union"] for r in prog]).sum())}
    for r in prog:
        u = r.pop("_union")
        r["coding_set_triggered_union"] = int((u & coding).sum()); r["text_set_triggered_union"] = int((u & text_set).sum())
        out["programs"].append(r)
    json.dump(out, open(os.path.join(WORK, "software_scan.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(WORK, "software_sets.npz"), coding=coding, text=text_set, code_cat=code_cat_set, **{f"seg_{s}": v for s, v in seg_sets.items()})
    print(f"\nCODING neurons (full-stack software): {int(coding.sum()):,} of {tot:,} ({100 * coding.sum() / tot:.2f}%)", flush=True)
    print("  overlap with text-reply set:", out["overlap_coding_text"], "| with earlier 'Code' category set:", out["overlap_coding_code_category"], f"of {out['code_category_set_earlier']}", flush=True)
    print("  by language part (web programs 7-10):", out["segment_sets"], "| overlaps:", out["segment_overlap"], flush=True)
    t = out["coding_by_layer"]
    print("  by layer thirds (0-11 / 12-23 / 24-35):", sum(t[:12]), sum(t[12:24]), sum(t[24:]), flush=True)

    # ------------------------------------------------ 3D map data: the 8,000 most engaged neurons per program (mean |act| over its code tokens)
    rows = []
    for i, r in enumerate(out["programs"]):
        mu = SUM[i].sum(0) / max(int(CNT[i].sum()), 1)
        flat = mu.reshape(-1); top = np.argpartition(-flat, 8000)[:8000]; top.sort()
        L = [top[(top // F) == l] % F for l in range(NL)]
        vals = np.concatenate([np.round(mu[l, L[l]] / max(mu[l].max(), 1e-9) * 127) for l in range(NL)]).astype(np.int8)
        rows.append({**r, "_top_idx": pack(np.concatenate(L), np.uint16), "_top_cnt": [len(x) for x in L], "_top_val": pack(vals, np.int8)})

    def ids_of(mask):
        L = [np.where(mask[l])[0].astype(np.uint16) for l in range(NL)]
        return {"idx": pack(np.concatenate(L), np.uint16), "cnt": [len(x) for x in L]}
    data = {"layers": NL, "ffn": F, "rows": rows, "coding": ids_of(coding), "text": ids_of(text_set), "segs": {s: ids_of(v) for s, v in seg_sets.items()},
            "summary": {k: v for k, v in out.items() if k != "programs"}}
    tpl = open(os.path.join(HERE, "software_map_template.html"), encoding="utf-8").read()
    open(os.path.join(ROOT, "software_neuron_map.html"), "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<" + chr(92) + "/")))
    print(f"DONE in {time.time() - t0:.0f}s -> software_neuron_map.html", flush=True)


if __name__ == "__main__":
    main()
