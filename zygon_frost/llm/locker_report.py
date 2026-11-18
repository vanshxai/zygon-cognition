"""Live, plain-text report pages for the Neuron Path Locker (served by frost_app.py)."""
import html, json, os, time
import numpy as np
import path_locker as PL

STYLE = ("<style>body{font:13px/1.45 Consolas,monospace;margin:14px;background:#fff;color:#000}pre{margin:2px;white-space:pre-wrap}"
         "details{margin-left:1.2em}summary{cursor:pointer}h3{font:700 16px system-ui,sans-serif}</style>")


def det(title, body, op=False):
    return f"<details{' open' if op else ''}><summary>{html.escape(title)}</summary><pre>{html.escape(body)}</pre></details>"


def ago(t):
    d = time.time() - t
    return f"{d / 86400:.1f} d" if d > 86400 else f"{d / 3600:.1f} h" if d > 3600 else f"{d / 60:.0f} min" if d > 60 else f"{d:.0f} s"


def paths_page(LK):
    st, recs = LK.stats, sorted(LK.rec.values(), key=lambda r: -r["last_used"])
    locked = [r for r in recs if r["status"] == "locked"]
    head = (f"records {len(recs)} (limit {LK.max_records})   locked {len(locked)}   candidates {len(recs) - len(locked)}   disk {LK.disk_mb():.2f} MB (limit {LK.max_mb})\n"
            f"locked-path replays {st['lock_hits']}   branch runs {st['branch_runs']}   new runs {st['new_runs']}   tokens replayed without running a layer {st['tokens_replayed']:,}   erased to save memory {st['evicted']}\n\n"
            "locked    : approved reply. The same ask (same content words, same conversation context) returns it from disk in milliseconds.\n"
            "candidate : discovered but not approved. Used only as a draft source for similar asks; erased first when the index is full.\n"
            "branch    : a changed ask. The model followed an older path, generated the changed part itself and rejoined the old path when the text lined up again.\n"
            "erasing   : when over the limit, the path with the lowest (2*hits + 3 if locked + 1) / (1 + days since last use) is erased first.")
    rows = []
    for r in recs:
        br = r.get("branches", [])
        followed = r["n_tokens"] - sum(b["end"] - b["start"] for b in br) if r.get("parent") else 0
        t = f"{r['status']:9s} {r['kind']:4s} hits {r['hits']:>3}  {r['n_tokens']:>4} tokens  used {ago(r['last_used']):>8} ago  {'branch of ' + r['parent'] if r.get('parent') else 'new path':28s} {r['ask'][:60]}"
        body = f"id         : {r['id']}\nask        : {r['ask']}\ncontext    : {r['ctx'] or '(first message of a conversation)'}\ncontent key: {r['key']}\n"
        if r.get("parent"):
            body += f"followed older path for {followed} of {r['n_tokens']} tokens; fresh (branch) token ranges: {[(b['start'], b['end']) for b in br]}\n"
        if r.get("stats"):
            body += "run stats  : " + json.dumps(r["stats"]) + "\n"
        if r.get("fingerprint"):
            try:
                z = np.load(LK._npz(r["id"]))
                lay = z["fp_cnt"].astype(np.int64).sum(1)
                body += f"neuron path: {r.get('neurons_triggered_ever', 0):,} neurons triggered at least once over the reply; top {PL.FP_K} per layer stored.\n  busiest layers: {[int(i) for i in np.argsort(-lay)[:6]]}\n"
                body += "  layer 0 / 17 / 35 top neurons: " + " | ".join(str([int(x) for x in z['fp_ids'][l][:6]]) for l in (0, 17, 35)) + "\n"
                if "br_ids" in z.files:
                    body += "  branch-only neurons (layer 17 top 6): " + str([int(x) for x in z["br_ids"][17][:6]]) + "\n"
            except Exception as e:
                body += f"neuron path: unreadable ({e})\n"
        else:
            body += "neuron path: not traced yet (traced in the background when you approve the reply)\n"
        body += "\n--- reply ---\n" + r["reply"]
        rows.append(det(t, body))
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Neuron path locker</title>{STYLE}</head><body><h3>Neuron path locker - locked and discovered paths</h3>"
            + det("How it works and totals", head, True) + ("".join(rows) or "<pre>no paths yet - chat with Frost; every reply is stored as a candidate path, press Approve under a reply to lock it.</pre>") + "</body></html>")


def code_page(LK):
    f = os.path.join(LK.dir, "code_index.json")
    if not os.path.exists(f):
        return "<!doctype html><meta charset='utf-8'><pre>code index not built yet: python code_index.py parse ; python code_index.py trace</pre>"
    J = json.load(open(f, encoding="utf-8")); fns = J["functions"]; tr = J.get("trace")
    common = [x for x in fns if x["kind"] == "common"]
    head = f"{len(fns)} functions in {len(J['programs'])} programs: {len(common)} common (same name in 2+ programs), {len(fns) - len(common)} unique.\n"
    if tr:
        head += (f"neurons traced for {tr['traced_functions']} functions (needs >= {tr['min_tokens']} tokens).\n"
                 f"Do functions with the same name fire the same neurons?  top-{PL.FP_K}-per-layer overlap: same name {100 * tr['same_name_overlap']:.1f}% ({tr['same_name_pairs']} pairs) vs different names {100 * tr['different_name_overlap']:.1f}% vs pure chance {100 * tr['chance']:.2f}%\n")
    else:
        head += "neurons not traced yet (python code_index.py trace).\n"
    head += ("\nFor neuron (layer L, id i) the weights are: blk.L.ffn_gate.weight[i,:], blk.L.ffn_up.weight[i,:] and blk.L.ffn_down.weight[:,i].\n"
             "A function's neurons = those that fire more than average code does on its tokens (largest excess, 48 per layer, stored in code_index.npz).\n"
             "The function bodies are also used as drafts: when Frost writes code that matches one, the model verifies many of its tokens per pass.")
    byname = {}
    for x in common:
        byname.setdefault((x["lang"], x["name"]), []).append(x)
    crow = []
    for (lang, name), v in sorted(byname.items(), key=lambda kv: -len(kv[1])):
        body = "\n".join(f"{x['program']:16s} {x['lines']:>3} lines {x['tokens_n']:>4} tokens" + (f"  top layers {x['top_layers']}  top neurons {[(n['layer'], n['neuron']) for n in x.get('top_neurons', [])[:3]]}" if x.get("traced") else "") for x in v)
        crow.append(det(f"{name}   ({lang}, in {len(v)} programs)", body))
    urow = []
    for x in fns:
        if x["kind"] == "unique":
            t = f"{x['program']:16s} {x['name']:22s} {x['lang']:10s} {x['lines']:>3} lines {x['tokens_n']:>4} tokens"
            body = x["sig"] + "\n" + (f"top layers {x['top_layers']}\ntop neurons (layer, id, excess): " + ", ".join(f"({n['layer']},{n['neuron']},{n['lift']})" for n in x.get("top_neurons", [])) if x.get("traced") else "not traced (too short)")
            urow.append(det(t, body))
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Code function index</title>{STYLE}</head><body><h3>Neuron path locker - code function index</h3>"
            + det("Summary", head, True) + "<details><summary>Common functions (same name in several programs)</summary>" + "".join(crow) + "</details>"
            + "<details><summary>Unique functions (one program only)</summary>" + "".join(urow) + "</details></body></html>")
