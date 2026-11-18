"""Does Frost Lang reply to anything? Sends many kinds of input to the running app (port 8840) and records everything.
    python lang_eval.py  ->  model/lang_solo/eval.json and one line in runs.jsonl
"""
import collections, json, os, random, re, string, sys, time, urllib.request

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
B = "http://127.0.0.1:8840"; F = os.path.join(ROOT, "model", "lang_solo")
random.seed(5)


def post(u, b): return json.load(urllib.request.urlopen(urllib.request.Request(B + u, data=json.dumps(b).encode(), method="POST"), timeout=60))


# the book's words
book = [json.loads(l) for l in open(os.path.join(F, "learned.jsonl"), encoding="utf-8") if '"t": "doc"' in l]
sents = [s for e in book for s in e["sentences"]]; cnt = collections.Counter(w for s in sents for w in re.findall(r"[a-z]+", s.lower()))
vocab = [w for w, c in cnt.items() if len(w) >= 4]; common = [w for w, _ in cnt.most_common(40)]; rare = [w for w in vocab if cnt[w] <= 2]


def typo(w):
    i = random.randrange(1, len(w) - 1); k = random.choice(("drop", "swap", "double", "sub"))
    return w[:i] + w[i + 1:] if k == "drop" else w[:i] + w[i + 1] + w[i] + w[i + 2:] if k == "swap" and i + 2 <= len(w) else w[:i] + w[i] + w[i:] if k == "double" else w[:i] + random.choice(string.ascii_lowercase) + w[i + 1:]


groups = {
    "words from the book": random.sample(vocab, 80), "very common words": common[:30], "rare words (seen once or twice)": random.sample(rare, 40), "typing mistakes in book words": [typo(w) for w in random.sample(vocab, 60)],
    "made-up words": ["blorp", "zxqv", "wumpus", "florbitz", "qwerty", "snorkelwhack", "grumblefitz", "xylothrop", "vleeb", "plunkaroo"] * 2,
    "punctuation and keyboard symbols": list(string.punctuation), "digits and numbers": list("0123456789") + ["42", "2024", "3.14", "1000000"],
    "other characters": ["é", "ñ", "ü", "€", "£", "✓", "😀", "你好", "я", "ß", "\t", "  "], "short questions": [f"what is {w}" for w in random.sample(vocab, 15)] + [f"tell me about {w}" for w in random.sample(vocab, 10)],
    "everyday phrases": ["hey", "hello", "how are you", "thanks", "bye", "who are you", "what can you do", "i am sad", "good morning", "help"]}
res, rows, total_ms = {}, [], 0.0
for g, inputs in groups.items():
    kinds = collections.Counter(); ms = []; rel = 0; replied = 0; ex = []
    for q in inputs:
        if not q.strip() and g != "other characters": continue
        t = time.perf_counter(); r = post("/api/ask", {"q": q}); wall = (time.perf_counter() - t) * 1000; ms.append(r.get("ms") or wall)
        kind = r["kind"]; kinds[kind] += 1; text = r.get("text") or ""
        if text: replied += 1
        if g in ("words from the book", "rare words (seen once or twice)") and kind == "lang" and q.lower() in text.lower(): rel += 1
        if len(ex) < 3: ex.append({"typed": q, "kind": kind, "reply": text[:140]})
    n = len(inputs); res[g] = {"typed": n, "replied_with_text": replied, "kinds": dict(kinds), "mean_ms": sum(ms) / len(ms), "max_ms": max(ms), "reply_contains_the_typed_word": rel if g in ("words from the book", "rare words (seen once or twice)") else None, "examples": ex}; total_ms += sum(ms)
tot = sum(v["typed"] for v in res.values()); ans = sum(v["replied_with_text"] for v in res.values())
out = {"inputs": tot, "replied_with_a_sentence": ans, "asked_where_to_learn": tot - ans, "mean_ms": total_ms / tot, "book_words": len(cnt), "book_sentences": len(sents), "groups": res}
json.dump(out, open(os.path.join(F, "eval.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
with open(os.path.join(F, "runs.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "engine": "Frost Lang", "kind": "reply test", "name": "always-reply test", "metrics": {k: v for k, v in out.items() if k != "groups"}, "test": {g: {k: v for k, v in d.items() if k != "examples"} for g, d in res.items()}, "after": {}}, ensure_ascii=False) + "\n")
print(f"{tot} inputs: {ans} got a sentence, {tot - ans} got a question about where to read; mean {out['mean_ms']:.2f} ms")
for g, d in res.items(): print(f"  {g:36} typed {d['typed']:3} -> text {d['replied_with_text']:3} {dict(d['kinds'])} {('contains word ' + str(d['reply_contains_the_typed_word'])) if d['reply_contains_the_typed_word'] is not None else ''} mean {d['mean_ms']:.2f} ms")
