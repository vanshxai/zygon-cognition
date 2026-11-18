"""Feeds all downloaded books into a fresh Frost Lang store, together, and records every metric.
    python feed_books.py   (the old lang_solo store is moved to model/lang_solo_before_500)"""
import json, os, shutil, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import lang, si_lessons
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..")); F = os.path.join(ROOT, "model", "lang_solo"); B = os.path.join(ROOT, "model", "books")
if os.path.exists(F):
    old = F + "_before_500"
    if os.path.exists(old): shutil.rmtree(old)
    shutil.move(F, old)
cat = {b["id"]: b for b in json.load(open(os.path.join(B, "catalogue_science.json"), encoding="utf-8"))["books"]}
def body(t):
    import re
    a = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", t); b = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", t)
    return t[a.end():b.start()] if a and b else t
L = lang.Lang(F); t_all = time.perf_counter()
t0 = time.perf_counter()
for c, utts, reps in si_lessons.LESSONS: L.learn_lesson(c, utts, reps)
for par in si_lessons.READING: L.learn_text(par, "reading")
eng_s = time.perf_counter() - t0; eng_mem = L.memory()["total"]
learn_s = read_s = 0.0; words = 0; done = 0; per = []
for i, f in enumerate(sorted(os.listdir(os.path.join(B, "text")), key=lambda x: int(x.split(".")[0]))):
    bid = int(f.split(".")[0]); t = time.perf_counter(); txt = body(open(os.path.join(B, "text", f), encoding="utf-8", errors="replace").read()); read_s += time.perf_counter() - t
    t = time.perf_counter(); w = L.learn_text(txt, f"{cat[bid]['title'][:70]} (Gutenberg {bid})" if bid in cat else f"Gutenberg {bid}"); dt = time.perf_counter() - t; learn_s += dt; words += w; done += 1; per.append(dt)
    if done % 50 == 0: print(done, f"{time.perf_counter() - t_all:.0f}s", words, flush=True)
wall = time.perf_counter() - t_all; m = L.memory(); disk = sum(os.path.getsize(os.path.join(F, x)) for x in os.listdir(F))
doc = {"name": f"{done} science books (Gutenberg)", "pages": None, "pdf_bytes": sum(os.path.getsize(os.path.join(B, "text", x)) for x in os.listdir(os.path.join(B, "text"))), "words_pdf": words, "words_learned": words, "sentences": m["sentences"], "tokens": words, "unit": "words", "learn_s": learn_s,
       "words_per_s": words / learn_s, "tokens_per_s": words / learn_s, "extract_s": read_s, "tokenise_s": 0.0, "elapsed_s": wall, "memory_added": m["total"] - eng_mem, "mode": "book batch", "books": done}
json.dump([{"name": "everyday English", "pages": None, "words_pdf": 17154, "words_learned": 17154, "sentences": 219, "tokens": 3680, "unit": "words", "learn_s": eng_s, "mode": "full speed", "lessons": len(si_lessons.LESSONS)}, doc], open(os.path.join(F, "docs.json"), "w"), indent=1)
rec = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "engine": "Frost Lang", "kind": "book batch", "name": doc["name"], "metrics": {**doc, "english_learn_s": eng_s, "book_learn_s_total": learn_s, "file_read_s": read_s, "slowest_book_s": max(per), "mean_book_s": sum(per) / len(per), "wall_s": wall, "books_text_bytes": doc["pdf_bytes"]}, "test": None,
       "after": {"memory_bytes": m["total"], "memory_parts": m["parts"], "sentences": m["sentences"], "vocabulary": m["words"], "postings": m["postings"], "disk_bytes": disk}}
open(os.path.join(F, "runs.jsonl"), "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
print(json.dumps({k: rec["metrics"][k] for k in ("books", "words_learned", "sentences", "book_learn_s_total", "file_read_s", "wall_s", "words_per_s", "memory_added")}), "mem", m["total"], "disk", disk, m["words"], "vocab")
