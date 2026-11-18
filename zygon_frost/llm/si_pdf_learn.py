"""ZYGON learns a PDF on its own (no 3B model, no embedding table, no matrices) and everything is recorded.
    python si_pdf_learn.py   -> model/si_pdf/results.json, si_pdf_report.html (root)
Stages timed: PDF text extraction, sentence split, tokenising, learning (with a growth checkpoint every 10%), memory (estimated bytes, real Python memory with tracemalloc, bytes on disk,
text store export), reply speed (word search, wheel completion, ask), quality (retrieval top-1/top-5, sentence completion from its first 5 tokens, held-out next-token accuracy).
"""
import html, json, math, os, platform, shutil, statistics, sys, time, tracemalloc
import numpy as np
import pypdf
from gguf import GGUF
from tokenizer import Tokenizer
import angle_ai as A
import path_locker as PL
import exp_store

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
EXP = os.path.join(ROOT, "model", "exp_store"); OUT = os.path.join(ROOT, "model", "si_pdf")
PDF = os.path.join(EXP, "bitcoin.pdf")
PR = lambda *a: print(*a, flush=True)


def fresh(name):
    d = os.path.join(ROOT, "model", name); shutil.rmtree(d, ignore_errors=True); return d


def learn_all(ai, sents_ids, texts, checkpoints=True):
    t0 = time.perf_counter(); curve = []; n = len(texts); step = max(1, n // 10); toks = 0
    for i, (txt, ids) in enumerate(zip(texts, sents_ids)):
        ai.learn_qa(txt, ids, "pdf", ended=True, pull=False); toks += len(ids) + 1
        if checkpoints and ((i + 1) % step == 0 or i + 1 == n):
            m = ai.memory(); curve.append({"sentences": i + 1, "tokens": toks, "seconds": time.perf_counter() - t0, "wheels": m["wheels"], "sectors": m["sectors"], "hooks": m["hooks"], "memory_bytes": m["total"]})
    return time.perf_counter() - t0, toks, curve


def next_acc(ai, ids_list):
    hit = tot = oov = 0; known = ai.cell_of
    for ids in ids_list:
        hist = [A.BOS] * ai.K
        for t in ids + [A.END]:
            cand = ai._turn(hist, []); tot += 1
            if t not in known: oov += 1
            if cand and max(cand, key=cand.get) == t: hit += 1
            hist.append(t)
    return {"accuracy": hit / max(1, tot), "tokens": tot, "never_seen_token_share": oov / max(1, tot)}


def main():
    info = {"python": platform.python_version(), "cpu": platform.processor(), "cores": os.cpu_count(), "platform": platform.platform()}
    tok = Tokenizer(GGUF().meta); R = {"machine": info}
    # ---------------------------------------------------------------- 1. read the PDF, split, tokenise
    t = time.perf_counter(); rd = pypdf.PdfReader(PDF); text = "\n".join((pg.extract_text() or "") for pg in rd.pages); R["extract_s"] = time.perf_counter() - t
    open(os.path.join(EXP, "bitcoin.txt"), "w", encoding="utf-8").write(text)
    t = time.perf_counter(); secs, sents = exp_store.load_sentences(); R["split_s"] = time.perf_counter() - t
    texts = [s["text"] for s in sents]; sec_of = [s["sec"] for s in sents]
    t = time.perf_counter(); ids_list = [tok.encode(x) for x in texts]; R["tokenise_s"] = time.perf_counter() - t
    ntok = sum(len(i) for i in ids_list)
    R["source"] = {"pdf": "bitcoin.pdf", "pages": len(rd.pages), "pdf_bytes": os.path.getsize(PDF), "text_chars": len(text), "text_bytes": len(text.encode()), "sentences": len(texts), "sections": len(secs), "tokens": ntok,
                   "unique_tokens": len({t for i in ids_list for t in i}), "learned_text_bytes": sum(len(x.encode()) for x in texts)}
    PR(f"PDF: {R['source']}  extract {R['extract_s']:.2f}s split {R['split_s']:.3f}s tokenise {R['tokenise_s']:.2f}s")
    # ---------------------------------------------------------------- 2. learning (timed run, no tracing)
    shutil.rmtree(OUT, ignore_errors=True); ai = A.AngleAI(tok, OUT)
    learn_s, toks, curve = learn_all(ai, ids_list, texts)
    m = ai.memory(); ai.settings["memory_threshold_mb"] = 50.0
    R["learn"] = {"seconds": learn_s, "tokens": toks, "tokens_per_s": toks / learn_s, "sentences_per_s": len(texts) / learn_s, "ms_per_sentence": 1000 * learn_s / len(texts), "growth": curve}
    store_bytes = ai.export_store_text(os.path.join(OUT, "store_text.txt"))
    R["memory"] = {"estimated_total_bytes": m["total"], "parts": m["parts"], "wheels": m["wheels"], "sectors": m["sectors"], "hooks": m["hooks"], "pulls": m["pulls"], "paths": m["paths"],
                   "disk_bytes_log_and_metrics": ai.disk_bytes() - store_bytes, "text_store_bytes": store_bytes, "bytes_per_token": m["total"] / toks, "tokens_known": len(ai.cell_of),
                   "vs_pdf_bytes": m["total"] / R["source"]["pdf_bytes"], "vs_text_bytes": m["total"] / R["source"]["learned_text_bytes"]}
    # ---------------------------------------------------------------- 3. the same learning with real Python memory measured
    tracemalloc.start(); d2 = fresh("si_pdf_mem"); base = tracemalloc.get_traced_memory()[0]; ai_m = A.AngleAI(tok, d2); learn_all(ai_m, ids_list, texts, checkpoints=False)
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop(); shutil.rmtree(d2, ignore_errors=True)
    R["memory"]["python_ram_bytes_after_learning"] = cur - base; R["memory"]["python_ram_peak_bytes"] = peak - base; del ai_m
    PR(f"learned {toks} tokens in {learn_s:.3f}s ({toks / learn_s:,.0f} tokens/s); estimated memory {m['total']:,} B; python RAM {cur - base:,} B; text store {store_bytes:,} B")
    # ---------------------------------------------------------------- 4. reply speed + quality
    rng = np.random.default_rng(7); queries = []
    for i, tx in enumerate(texts):
        w = exp_store.words_of(tx)
        if len(w) < 10: continue
        a = int(rng.integers(0, len(w) - 5)); queries.append(("6 consecutive words", " ".join(w[a:a + 6]), i)); queries.append(("4 random keywords", " ".join(rng.choice(w, 4, replace=False)), i))
    lat, t1, t5, sec1, kind = [], 0, 0, 0, {}
    for k, q, i in queries:
        t = time.perf_counter(); res = ai.search(q, 5); lat.append(1000 * (time.perf_counter() - t)); ranked = [pid - 1 for _, pid in res]      # path ids start at 1
        ok1 = bool(ranked) and ranked[0] == i; t1 += ok1; t5 += i in ranked; sec1 += bool(ranked) and sec_of[ranked[0]] == sec_of[i]; kind.setdefault(k, []).append(ok1)
    R["search"] = {"queries": len(queries), "top1": t1 / len(queries), "top5": t5 / len(queries), "right_section_top1": sec1 / len(queries), "by_kind": {k: sum(v) / len(v) for k, v in kind.items()},
                   "mean_ms": statistics.mean(lat), "median_ms": statistics.median(lat), "p95_ms": float(np.percentile(lat, 95)), "max_ms": max(lat)}
    # sentence completion from the first 5 tokens (wheels only)
    comp, exact, ratio, per_tok, samples = [], 0, [], [], []
    for i, ids in enumerate(ids_list):
        if len(ids) < 12: continue
        prefix, rest = ids[:5], ids[5:] + [A.END]; ptxt = tok.decode(prefix)
        t = time.perf_counter(); out = ai.complete(ptxt, len(rest) + 8); dt = time.perf_counter() - t
        out_ids = out + ([A.END] if len(out) < len(rest) + 8 and True else [])
        n_ok = 0
        for a_, b_ in zip(out, rest[:-1]):
            if a_ == b_: n_ok += 1
            else: break
        full = out == rest[:-1]; exact += full; ratio.append(n_ok / max(1, len(rest) - 1)); per_tok.append(1000 * dt / max(1, len(out))); comp.append(len(out))
        if len(samples) < 8 and i % 14 == 0: samples.append({"start": ptxt, "si": tok.decode(out), "real": tok.decode(rest[:-1]), "exact": bool(full)})
    R["completion"] = {"sentences_tested": len(comp), "exact_rebuilt": exact / len(comp), "mean_correct_prefix_share": float(np.mean(ratio)), "ms_per_token": statistics.mean(per_tok), "tokens_per_s": 1000 / statistics.mean(per_tok), "samples": samples}
    # ask() speed on the live rules, on a copy so the evaluation does not change the learned store
    d3 = fresh("si_pdf_ask"); shutil.copytree(OUT, d3); aa = A.AngleAI(tok, d3); srcs, lats = {}, {}
    for k, q, i in queries[:120]:
        t = time.perf_counter(); r = aa.ask(q); ms = 1000 * (time.perf_counter() - t); s_ = r.get("source", "needs the 3B model"); srcs[s_] = srcs.get(s_, 0) + 1; lats.setdefault(s_, []).append(ms)
    shutil.rmtree(d3, ignore_errors=True)
    R["ask"] = {"queries": 120, "answered_by": srcs, "mean_ms": {k: statistics.mean(v) for k, v in lats.items()}}
    # held-out: learn the first 80% of the sentences, test the last 20%
    cut = int(len(texts) * 0.8); d4 = fresh("si_pdf_holdout"); ah = A.AngleAI(tok, d4); hs, ht, _ = learn_all(ah, ids_list[:cut], texts[:cut], checkpoints=False)
    R["heldout"] = {"learned_sentences": cut, "learn_seconds": hs, "test_sentences": len(texts) - cut, "on_unseen_sentences": next_acc(ah, ids_list[cut:]), "on_learned_sentences": next_acc(ah, ids_list[:cut])}
    shutil.rmtree(d4, ignore_errors=True)
    R["cube"] = ai.cube()
    json.dump({k: v for k, v in R.items() if k != "cube"}, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1, default=float)
    PR(json.dumps({k: R[k] for k in ("search", "completion", "ask", "heldout")}, indent=1, default=float)[:2500])
    page(R)


def page(R):
    pc = lambda x: f"{100 * x:.1f}%"; S, L, M, Sr, Co, Ak, Ho = R["source"], R["learn"], R["memory"], R["search"], R["completion"], R["ask"], R["heldout"]
    kb = lambda b: f"{b / 1024:.1f} KB" if b < 1048576 else f"{b / 1048576:.2f} MB"
    g = L["growth"]; W, H = 560, 200; mx = max(x["memory_bytes"] for x in g); mt = max(x["tokens"] for x in g)
    pts = " ".join(f"{20 + x['tokens'] / mt * (W - 40):.1f},{H - 20 - x['memory_bytes'] / mx * (H - 40):.1f}" for x in g)
    svg = f"<svg width='{W}' height='{H}' style='background:#fff;border:1px solid #999'><polyline fill='none' stroke='#0A6C80' stroke-width='2' points='{pts}'/><text x='24' y='14' font-size='11'>memory (bytes) against tokens learned: 0 to {mt:,} tokens, 0 to {kb(mx)}</text></svg>"
    row = lambda *c: "<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>"
    growth = "".join(row(x["sentences"], f"{x['tokens']:,}", f"{x['seconds'] * 1000:.0f} ms", f"{x['wheels']:,}", f"{x['sectors']:,}", f"{x['hooks']:,}", f"{x['memory_bytes']:,} B") for x in g)
    samples = "".join(row(html.escape(s["start"]), html.escape(s["si"][:170]), html.escape(s["real"][:170]), "yes" if s["exact"] else "no") for s in Co["samples"])
    cube = json.dumps(R["cube"], ensure_ascii=False).replace("</", "<\\/"); js = open(os.path.join(HERE, "cube_view.js"), encoding="utf-8").read()
    verdict = [
        f"Learning: {S['sentences']} sentences ({L['tokens']:,} tokens) were learned in {L['seconds'] * 1000:.0f} ms, which is {L['tokens_per_s']:,.0f} tokens per second, with no 3B model and no matrix. Reading the PDF text took {R['extract_s']:.2f} s and tokenising {R['tokenise_s']:.2f} s, so reading and tokenising cost more than learning.",
        f"Memory: {M['estimated_total_bytes']:,} bytes estimated for the learned structures ({M['bytes_per_token']:.1f} bytes per token), {M['python_ram_bytes_after_learning']:,} bytes of real Python memory, and {M['text_store_bytes']:,} bytes as the exponent-and-angle text store. The PDF is {S['pdf_bytes']:,} bytes and its learned text {S['learned_text_bytes']:,} bytes.",
        f"Finding a sentence again from 4 to 6 of its words: {pc(Sr['top1'])} top-1 and {pc(Sr['top5'])} top-5, in {Sr['mean_ms']:.2f} ms on average. This is the same job a plain word index does.",
        f"Writing a sentence from its first 5 tokens with the wheels only: {pc(Co['exact_rebuilt'])} rebuilt exactly, {pc(Co['mean_correct_prefix_share'])} of each sentence correct before the first slip, at {Co['ms_per_token']:.3f} ms per token ({Co['tokens_per_s']:,.0f} tokens per second).",
        f"Generalising: trained on the first {pc(Ho['learned_sentences'] / S['sentences'])} of the sentences, the wheels guessed the next token right {pc(Ho['on_unseen_sentences']['accuracy'])} of the time on sentences they had never seen (on the ones they had learned: {pc(Ho['on_learned_sentences']['accuracy'])}). Of the unseen sentences' tokens, {pc(Ho['on_unseen_sentences']['never_seen_token_share'])} had never appeared in training.",
        "So after one reading ZYGON remembers this document almost word for word, and can find and replay it fast, but its guesses on text it has not read are weak. Fluency on new text needs a lot more reading."]
    out = f"""<!doctype html><html><head><meta charset="utf-8"><title>ZYGON learns a PDF</title><style>
body{{font:14px/1.5 Consolas,monospace;margin:16px;max-width:1150px;background:#fff;color:#111}}h2{{font:700 17px system-ui;margin:22px 0 6px;border-bottom:1px solid #999}}table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #999;padding:3px 7px;vertical-align:top;text-align:left}}th{{background:#e6f2f4}}
#cvw{{position:relative;width:100%;height:520px;background:#06232B;border-radius:6px}}#cv{{width:100%;height:100%;display:block;cursor:grab}}#tip{{position:absolute;display:none;pointer-events:none;background:#000c;color:#fff;border:1px solid #1B6272;padding:5px 8px;font-size:12px;border-radius:5px}}
#info{{background:#e6f2f4;padding:6px 10px;margin-top:6px;min-height:40px}}button{{padding:3px 10px}}</style></head><body>
<h1 style="font:700 20px system-ui">ZYGON learns a PDF on its own: everything recorded</h1>
<p>ZYGON (exponent identities, angle wheels, hooks) read <b>{S['pdf']}</b> ({S['pages']} pages, {S['pdf_bytes']:,} bytes) with no 3B model, no embedding table and no matrices. Machine: {R['machine']['cores']} cores, Python {R['machine']['python']}, {html.escape(R['machine']['platform'])}.</p>
<h2>1. Source</h2><table>{row('pages', S['pages'])}{row('text extracted', f"{S['text_chars']:,} characters, {S['text_bytes']:,} bytes")}{row('sentences learned', f"{S['sentences']} in {S['sections']} sections")}{row('tokens', f"{S['tokens']:,} ({S['unique_tokens']:,} different)")}</table>
<h2>2. Time</h2><table><tr><th>stage</th><th>time</th></tr>{row('read the PDF text (pypdf)', f"{R['extract_s']:.2f} s")}{row('split into sentences', f"{R['split_s'] * 1000:.1f} ms")}{row('tokenise', f"{R['tokenise_s']:.2f} s")}{row('<b>learn</b>', f"<b>{L['seconds'] * 1000:.0f} ms</b> ({L['tokens_per_s']:,.0f} tokens/s, {L['ms_per_sentence']:.2f} ms per sentence)")}</table>
<h2>3. Memory</h2><table><tr><th>what</th><th>size</th></tr>{row('estimated, all learned structures', f"{M['estimated_total_bytes']:,} B ({kb(M['estimated_total_bytes'])}), {M['bytes_per_token']:.1f} B per token")}{''.join(row('&nbsp;&nbsp;' + k, f"{v:,} B") for k, v in M['parts'].items())}{row('real Python memory after learning (tracemalloc)', f"{M['python_ram_bytes_after_learning']:,} B (peak {M['python_ram_peak_bytes']:,} B)")}{row('text store (exponents + angles)', f"{M['text_store_bytes']:,} B")}{row('log + metrics on disk', f"{M['disk_bytes_log_and_metrics']:,} B")}{row('counts', f"{M['wheels']:,} wheels, {M['sectors']:,} sectors, {M['hooks']:,} hooks, {M['paths']} stored sentences, {M['tokens_known']:,} token identities")}{row('against the PDF', f"{M['vs_pdf_bytes']:.2f} x the PDF file, {M['vs_text_bytes']:.2f} x its text")}</table>
<p>Growth while reading (every 10% of the sentences):</p>{svg}
<table><tr><th>sentences</th><th>tokens</th><th>learning time so far</th><th>wheels</th><th>sectors</th><th>hooks</th><th>memory</th></tr>{growth}</table>
<h2>4. Reply speed</h2><table><tr><th>what</th><th>speed</th></tr>{row('find a sentence from 4 to 6 words (hooks)', f"{Sr['mean_ms']:.3f} ms mean, {Sr['median_ms']:.3f} median, {Sr['p95_ms']:.3f} p95, {Sr['max_ms']:.3f} max ({Sr['queries']} queries)")}{row('write words with the wheels', f"{Co['ms_per_token']:.3f} ms per token = {Co['tokens_per_s']:,.0f} tokens/s")}{row('for comparison, the 3B model on this PC', 'about 560 ms per token = 1.8 tokens/s')}{row('answer a question (ask, live rules)', ', '.join(f"{k}: {v} answers, {Ak['mean_ms'][k]:.3f} ms mean" for k, v in Ak['answered_by'].items()))}</table>
<h2>5. Quality</h2><table><tr><th>test</th><th>result</th></tr>{row('find the right sentence: top-1 / top-5', f"{pc(Sr['top1'])} / {pc(Sr['top5'])} (6 consecutive words {pc(Sr['by_kind']['6 consecutive words'])}, 4 random keywords {pc(Sr['by_kind']['4 random keywords'])}); right section {pc(Sr['right_section_top1'])}")}{row('earlier embedding experiment for comparison', 'word index 97.8% top-1, 3D position 9.7% top-1')}{row('rebuild a sentence from its first 5 tokens', f"{pc(Co['exact_rebuilt'])} exactly ({Co['sentences_tested']} sentences), {pc(Co['mean_correct_prefix_share'])} correct before the first slip")}{row(f"next word on unseen sentences (trained on {Ho['learned_sentences']} of {S['sentences']})", f"{pc(Ho['on_unseen_sentences']['accuracy'])} right ({Ho['on_unseen_sentences']['tokens']} tokens, {pc(Ho['on_unseen_sentences']['never_seen_token_share'])} of them never seen in training)")}{row('next word on the sentences it learned', pc(Ho['on_learned_sentences']['accuracy']))}</table>
<p>Sentences written by the wheels from their first 5 tokens:</p><table><tr><th>start</th><th>ZYGON wrote</th><th>real sentence</th><th>exact</th></tr>{samples}</table>
<h2>6. The exponent cube (10 x 10 x 10)</h2><p>Every token's exponent is in its own cell (cells numbered in the order tokens were first read). The box is only a picture; nothing computes with it. Drag to rotate, wheel to zoom, click a cell to see the next words after it. Colour = how often the token was read (blue rare, red frequent). {R['cube']['shown']} of {R['cube']['tokens_known']:,} tokens are in the box.</p>
<div id="cvw"><canvas id="cv"></canvas><div id="tip"></div></div><div id="info">click a cell</div><button id="b1">Stop spin</button> <button id="b2">Hide relations</button> <button id="b3">Reset view</button>
<h2>7. Verdict</h2><ul>{''.join(f'<li>{html.escape(v)}</li>' for v in verdict)}</ul>
<script>{js}
const cube = makeCube(document.getElementById('cv'), document.getElementById('tip'), document.getElementById('info')); cube.setData({cube});
document.getElementById('b1').onclick = e => {{ e.target.textContent = cube.toggleSpin() ? 'Stop spin' : 'Spin'; }}; document.getElementById('b2').onclick = e => {{ e.target.textContent = cube.toggleRel() ? 'Hide relations' : 'Show relations'; }}; document.getElementById('b3').onclick = () => cube.reset();
</script></body></html>"""
    open(os.path.join(ROOT, "si_pdf_report.html"), "w", encoding="utf-8").write(out)
    PR("written si_pdf_report.html")


if __name__ == "__main__":
    main()
