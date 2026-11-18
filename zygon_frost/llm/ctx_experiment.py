"""Chat-context experiment. Two-turn chats made from real paragraphs, with the chat thread off and on.
    python ctx_experiment.py [books] [cases]  ->  model/ctx_experiment.json and one row in model/lang_solo/runs.jsonl
 A  follow-up with one ambiguous word:   "tell me about w1 w2"  then  "and what about w3?"   right = reply is in the same paragraph as turn 1
 A2 same with a pronoun:                 "what did it say about w3?"
 B  "tell me more" / "and then?":        right = the reply starts after where turn 1 stopped, in the same book, within 3 sentences
 C  topic switch (no harm test):         turn 1 about one thing, turn 2 about something else; right = reply is in the paragraph that turn 2's words came from"""
import json, os, random, re, sys, tempfile, time
sys.stdout.reconfigure(encoding="utf-8")
import lang
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 100; NC = int(sys.argv[2]) if len(sys.argv) > 2 else 250
random.seed(21)


def body(t):
    a = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", t); b = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", t)
    return t[a.end():b.start()] if a and b else t


D = os.path.join(ROOT, "model", "books", "text"); files = sorted(os.listdir(D))[:NB]
L = lang.Lang(tempfile.mkdtemp()); para = []; pid = 0
for f in files:
    for p in re.split(r"\n\s*\n", body(open(os.path.join(D, f), encoding="utf-8", errors="replace").read())):
        ss = lang.split_sentences(p)
        if not ss: continue
        pid += 1
        for s in ss: L._add(s, f); para.append(pid)
        L.pend.add(len(L.sents) - 1)
N = len(L.sents); lim = 0.08 * N; byp = {}
for i, p in enumerate(para): byp.setdefault(p, []).append(i)
print(len(files), "books", N, "sentences", flush=True)
W = lambda i: set(lang.WORD.findall(L.sents[i].lower()))
rare = lambda i: sorted((w for w in W(i) if 4 <= len(w) and len(L.post[w]) <= lim and w not in L.FOLLOW and w not in L.PRON), key=lambda w: len(L.post[w]))

cases = []
for p, ids in random.sample(list(byp.items()), len(byp)):
    if len(ids) < 4: continue
    i = ids[0]; r1 = rare(i)
    if len(r1) < 2 or len(L.post[r1[0]]) > 300: continue
    w3s = [w for k in ids[1:] for w in rare(k) if w not in W(i) and 15 <= len(L.post[w]) <= 400]       # a word that is ambiguous on its own
    if not w3s: continue
    other = random.choice(range(N))
    r2 = rare(other)
    if len(r2) < 2 or para[other] == p: continue
    cases.append({"para": p, "i": i, "w1w2": r1[:2], "w3": w3s[0], "other": other, "w4w5": r2[:2]})
    if len(cases) >= NC: break
print(len(cases), "cases", flush=True)


def run(on):
    L.ctx_on = on; out = {k: {"n": 0, "right": 0, "same_book": 0} for k in ("A", "A2", "B", "B2", "C", "C2")}; t0 = time.perf_counter(); turns = 0
    def start(c):
        L.reset_chat(); r = L.associate("tell me about " + " ".join(c["w1w2"])); return r
    for c in cases:
        r1 = start(c)
        if not r1.get("ids") or para[r1["anchor"]] != c["para"]: continue                 # turn 1 must find the paragraph, otherwise the case says nothing about context
        end1 = r1["ids"][-1]; snap = (dict(L.ctx), L.last_end, set(L.said)); src = L.src[r1["anchor"]]
        def restore(): L.ctx, L.last_end, L.said = dict(snap[0]), snap[1], set(snap[2])
        for key, q in (("A", f"and what about {c['w3']}?"), ("A2", f"what did it say about {c['w3']}?")):
            restore(); r = L.associate(q); turns += 1; o = out[key]; o["n"] += 1
            if r.get("ids"): o["right"] += para[r["anchor"]] == c["para"]; o["same_book"] += L.src[r["anchor"]] == src
        for key, q in (("B", "tell me more"), ("B2", "and then?")):
            restore(); r = L.associate(q); turns += 1; o = out[key]; o["n"] += 1
            if r.get("ids"): a = r["ids"][0]; ok = L.src[a] == src and end1 < a <= end1 + 3; o["right"] += ok; o["same_book"] += L.src[a] == src
        restore(); w = c["w4w5"][0]; r = L.associate("what is " + w); turns += 1; o = out["C2"]; o["n"] += 1       # one new word: the reply must still be about that word
        if r.get("ids"): o["right"] += w in L.sents[r["anchor"]].lower(); o["same_book"] += w in r["text"].lower()
        restore(); r = L.associate("what is " + " ".join(c["w4w5"])); turns += 1; o = out["C"]; o["n"] += 1
        if r.get("ids"): o["right"] += para[r["anchor"]] == para[c["other"]]; o["same_book"] += L.src[r["anchor"]] == L.src[c["other"]]
    ms = (time.perf_counter() - t0) * 1000 / max(turns, 1)
    return {k: {**v, "right_rate": round(v["right"] / v["n"], 3) if v["n"] else None, "same_book_rate": round(v["same_book"] / v["n"], 3) if v["n"] else None} for k, v in out.items()}, ms


if len(sys.argv) > 3 and sys.argv[3] == "sweep":
    cases = cases[:100]; import itertools; table = []
    for gain, kappa, rho in itertools.product((0.3,), (0.8, 1.2, 2.0), (3.0, 5.0, 8.0)):
        L.ctxp.update(gain=gain, pron_gain=max(gain, 0.6), kappa=kappa, rho=rho); r, _ = run(True)
        table.append({"gain": gain, "kappa": kappa, "rho": rho, **{k: r[k]["right_rate"] for k in ("A", "A2", "C")}}); print(table[-1], flush=True)
    json.dump(table, open(os.path.join(ROOT, "model", "ctx_sweep2.json"), "w"), indent=1); sys.exit()
off, ms_off = run(False); on, ms_on = run(True)
names = {"A": "follow-up, one ambiguous word", "A2": "follow-up with 'it'", "B": "'tell me more'", "B2": "'and then?'", "C": "topic switch, two words (no harm)", "C2": "topic switch, one word (no harm)"}
print(f"{'test':34} {'n':>4} {'chat off':>9} {'chat on':>9}   (same book off/on)")
for k in names: print(f"{names[k]:34} {on[k]['n']:4} {off[k]['right_rate']:9} {on[k]['right_rate']:9}   {off[k]['same_book_rate']} / {on[k]['same_book_rate']}")
print(f"mean ms per turn: off {ms_off:.1f}, on {ms_on:.1f}")
res = {"books": len(files), "sentences": N, "cases": len(cases), "off": off, "on": on, "ms_off": ms_off, "ms_on": ms_on}
json.dump(res, open(os.path.join(ROOT, "model", "ctx_experiment.json"), "w", encoding="utf-8"), indent=1)
