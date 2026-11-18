"""Directional gate experiment. Builds a store from downloaded books, asks questions made from their own sentences,
and scores each way of choosing how many sentences to return against the book's own paragraph breaks.
    python dir_experiment.py [number_of_books] [questions]  ->  model/dir_experiment.json and one row in model/lang_solo/runs.jsonl"""
import json, os, random, re, sys, time, itertools, tempfile
sys.stdout.reconfigure(encoding="utf-8")
import lang
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 60; NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 400
random.seed(11)


def body(t):
    a = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", t); b = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", t)
    return t[a.end():b.start()] if a and b else t


D = os.path.join(ROOT, "model", "books", "text"); files = sorted(os.listdir(D))[:NB]
L = lang.Lang(tempfile.mkdtemp()); para = []; pid = 0; t0 = time.perf_counter(); words = 0
for f in files:
    for p in re.split(r"\n\s*\n", body(open(os.path.join(D, f), encoding="utf-8", errors="replace").read())):
        ss = lang.split_sentences(p)
        if not ss: continue
        pid += 1
        for s in ss: L._add(s, f); para.append(pid); words += len(s.split())
        L.pend.add(len(L.sents) - 1)
learn_s = time.perf_counter() - t0
print(f"{len(files)} books, {len(L.sents)} sentences, {words} words, learned in {learn_s:.1f} s", flush=True)
N = len(L.sents); lim = 0.08 * N
# questions: two of the rarest words of a random sentence in a paragraph of 3+ sentences
byp = {}
for i, p in enumerate(para): byp.setdefault(p, []).append(i)
qs = []
for i in random.sample(range(N), min(N, NQ * 6)):
    if len(byp[para[i]]) < 3: continue
    ws = [w for w in set(lang.WORD.findall(L.sents[i].lower())) if 4 <= len(w) and len(L.post[w]) <= lim]
    ws.sort(key=lambda w: len(L.post[w]))
    if len(ws) >= 2 and len(L.post[ws[0]]) >= 1: qs.append((i, f"what is {ws[0]} {ws[1]}", ws[:2]))
    if len(qs) >= NQ: break
print(len(qs), "questions", flush=True)


def score(params):
    L.dirp = {**L.dirp, "para": False, "beta": 0.0, "back": False, **params}; n = 0; tot = {"found": 0, "words": 0, "para_precision": 0.0, "f1": 0.0, "both_words": 0, "ends_whole": 0}
    for i, q, ws in qs:
        r = L.associate(q)
        if not r.get("ids"): continue
        n += 1; ids = r["ids"]; a = r["anchor"]; pa = para[a]; same = [j for j in ids if para[j] == pa]
        target = [j for j in byp[pa] if a - 0 <= j <= a + 3] or [a]
        tp = len(set(ids) & set(target)); prec = tp / len(ids); rec = tp / len(target); f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        tot["para_precision"] += len(same) / len(ids); tot["f1"] += f1; tot["words"] += sum(len(L.sents[j].split()) for j in ids)
        tot["both_words"] += all(w in L.sents[a].lower() for w in ws); tot["found"] += (para[a] == para[i])
    return {"n": n, **{k: round(v / n, 3) for k, v in tot.items()}}


rows = []
rows.append(("single sentence", {"mode": "single"}, score({"mode": "single"})))
rows.append(("pair (old rule)", {"mode": "pair"}, score({"mode": "pair"})))
for tau, theta, beta in itertools.product((1.0, 2.0, 4.0), (0.15, 0.25, 0.35), (0.0, 0.2, 0.4)):
    p = {"mode": "dir", "tau": tau, "theta": theta, "beta": beta, "para": False}; rows.append((f"gate tau={tau} theta={theta} beta={beta}", p, score(p)))
bestg = max((r for r in rows if r[1]["mode"] == "dir"), key=lambda r: r[2]["f1"])
p = dict(bestg[1], para=True); rows.append(("best gate + paragraph ends (uses the book's layout)", p, score(p)))
print(f"{'variant':42} {'words':>6} {'in-para':>8} {'F1':>6}")
for name, p, r in rows: print(f"{name:42} {r['words']:6.1f} {r['para_precision']:8.3f} {r['f1']:6.3f}")
best = bestg
out = {"books": len(files), "sentences": len(L.sents), "words": words, "learn_s": learn_s, "questions": len(qs), "rows": [{"variant": n, "params": p, **r} for n, p, r in rows], "best": best[0]}
json.dump(out, open(os.path.join(ROOT, "model", "dir_experiment.json"), "w", encoding="utf-8"), indent=1)
print("best gate:", best[0], best[2])
