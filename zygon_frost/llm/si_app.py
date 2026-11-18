"""ZYGON app (minimal chat + settings).
    python si_app.py --engine 1 --port 8820     ZYGON       (word pieces)      store: model/si_solo
    python si_app.py --engine 2 --port 8830     ZYGON 2.0   (letters)          store: model/si2_solo
Both: no 3B model. Unknown questions are looked up on the web, one source after another, and nothing is remembered until you approve it.
Every reading, teaching, quick check and remembered answer is written with all its numbers to <store>/runs.jsonl.
"""
import argparse, collections, json, os, re, shutil, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import pypdf, subprocess, html as htmllib
import si_lessons, web_search

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
ap = argparse.ArgumentParser(); ap.add_argument("--engine", type=int, default=1); ap.add_argument("--port", type=int, default=0); ARGS = ap.parse_args()
PORT = ARGS.port or {1: 8820, 2: 8830, 3: 8840, 4: 8850}[ARGS.engine]
try:
    import scan
except Exception:
    scan = None


def ram():
    try: return scan.mem_info()["process_working_set"] if scan else None
    except Exception: return None


# ================================================================== engines (the only part that differs between the two versions)
class Engine1:
    name = "ZYGON"

    def __init__(self):
        import angle_ai as A
        from gguf import GGUF
        from tokenizer import Tokenizer
        self.folder = os.path.join(ROOT, "model", "si_solo"); seed = os.path.join(ROOT, "model", "si_pdf")
        if not os.path.exists(os.path.join(self.folder, "learned.jsonl")): shutil.copytree(seed, self.folder, dirs_exist_ok=True)
        self.tok = Tokenizer(GGUF().meta); self.ai = A.AngleAI(self.tok, self.folder); self.units = "tokens"

    def prepare(self, texts):
        t = time.perf_counter(); ids = [self.tok.encode(x) for x in texts]; return ids, time.perf_counter() - t, sum(len(i) for i in ids)

    def add_sentence(self, text, prep, source, name): self.ai.learn_qa(text, prep, source, ended=True, pull=False)
    def add_lesson(self, cat, utts, reps): self.ai.learn_lesson(cat, utts, reps, "lesson: " + cat)
    def add_reading(self, par): return self.ai.learn_text(par, "reading")
    def lesson_tokens(self, reps): return sum(len(self.tok.encode(r)) + 1 for r in reps)
    def converse(self, q): r = self.ai.converse(q); return {"cat": r["cat"]} if r else None

    def chat(self, q):
        r = self.ai.converse(q)
        if r: return {"kind": "chat", "text": r["text"], "ms": r["ms"]}
        t0 = time.perf_counter(); s = self.ai.saved_answer(q)
        if s: return {"kind": "saved", "text": s["text"], "from": s["source"], "ms": (time.perf_counter() - t0) * 1000}
        c = self.ai.compose(q, min_cover=0.6)
        if c.get("found"): return {"kind": "doc", "text": c["text"], "ms": c["ms"], "segments": [{"text": g["text"], "source": ("bitcoin.pdf" if g["source"] in ("", "pdf") else g["source"])} for g in c["segments"]]}
        return None

    def remember(self, q, text, source): return f"p{self.ai.learn_qa(q, self.tok.encode(text), 'web: ' + source, ended=True, pull=True)}"
    def items(self, q): return self.ai.learned_items(q)
    def delete(self, i): return self.ai.delete_item(i)
    def memory_bytes(self): return self.ai.memory()["total"]
    def disk_bytes(self): return self.ai.disk_bytes()

    def reply_stats(self):
        a, c = self.ai.agg.get("reply"), self.ai.agg.get("reply_tokens"); nf = self.ai.agg.get("reply_none")
        return {"n": a[0] if a else 0, "not_found": nf[0] if nf else 0, "mean_ms": (a[1] / a[0]) if a else None, "fastest_ms": a[2] if a else None, "slowest_ms": a[3] if a else None, "units_per_s": (c[1] / (a[1] / 1000)) if a and c and a[1] else None, "unit": "tokens"}

    def counts(self):
        saved = sum(1 for p in self.ai.paths.values() if p["source"].startswith("web:"))
        return {"sentences": len(self.ai.paths) - saved, "lessons": len(self.ai.groups), "saved": saved}

    def snapshot(self):
        m = self.ai.memory(); return {"memory_bytes": m["total"], "memory_parts": m["parts"], "wheels": m["wheels"], "sectors": m["sectors"], "hooks": m["hooks"], "identities": len(self.ai.cell_of)}

    def timing_ok(self): return True


class Engine2:
    name = "ZYGON 2.0"

    def __init__(self):
        import si2
        self.folder = os.path.join(ROOT, "model", "si2_solo"); self.ai = si2.SI2(self.folder); self.units = "letters"

    def prepare(self, texts): return None, 0.0, sum(len(si2_clean(x)) for x in texts)
    def add_sentence(self, text, prep, source, name): self.ai.learn_sentence(text, source, name)
    def add_lesson(self, cat, utts, reps): self.ai.learn_lesson(cat, utts, reps)
    def add_reading(self, par): return self.ai.learn_text(par, "reading")
    def lesson_tokens(self, reps): return sum(len(r) for r in reps)
    def converse(self, q): r = self.ai.converse(q); return {"cat": r["cat"]} if r else None

    def chat(self, q):
        r = self.ai.converse(q)
        if r: return {"kind": "chat", "text": r["text"], "ms": r["ms"]}
        t0 = time.perf_counter(); s = self.ai.saved_answer(q)
        if s: return {"kind": "saved", "text": s["text"], "from": s["source"], "ms": (time.perf_counter() - t0) * 1000}
        c = self.ai.compose(q)
        if c.get("found"): return {"kind": "doc", "text": c["text"], "ms": c["ms"], "segments": [{"text": g["text"], "source": g["source"]} for g in c["segments"]]}
        return None

    def remember(self, q, text, source): return self.ai.remember(q, text, "web: " + source)
    def items(self, q): return self.ai.learned_items(q)
    def delete(self, i): return self.ai.delete_item(i)
    def memory_bytes(self): return self.ai.memory()["total"]
    def disk_bytes(self): return sum(os.path.getsize(os.path.join(self.folder, f)) for f in os.listdir(self.folder) if os.path.isfile(os.path.join(self.folder, f)))
    def reply_stats(self): return self.ai.reply_stats()

    def counts(self):
        its = [i for i in self.ai.items.values() if i["alive"]]
        return {"sentences": sum(1 for i in its if i["kind"] == "document"), "lessons": sum(1 for g in self.ai.groups.values() if g["alive"]), "saved": sum(1 for i in its if i["kind"] == "saved answer")}

    def snapshot(self):
        m = self.ai.memory(); return {"memory_bytes": m["total"], "memory_parts": m["parts"], "letters": m["letters"], "index_entries": m["index_entries"], "triplets": m["triplets"]}


class Engine3:
    name = "Frost Lang"

    def __init__(self):
        import lang
        self.folder = os.path.join(ROOT, "model", "lang_solo"); self.ai = lang.Lang(self.folder); self.units = "words"

    def prepare(self, texts): return None, 0.0, sum(len(t.split()) for t in texts)
    def add_sentence(self, text, prep, source, name): self.ai.add_sentence(text, name or source.split(" p.")[0])
    def add_lesson(self, cat, utts, reps): self.ai.learn_lesson(cat, utts, reps)
    def add_reading(self, par): return self.ai.learn_text(par, "reading")
    def lesson_tokens(self, reps): return sum(len(r.split()) for r in reps)
    def converse(self, q): r = self.ai.lesson(q); return {"cat": r["cat"]} if r else None
    def chat(self, q): return self.ai.chat(q)
    def remember(self, q, text, source): return self.ai.remember(q, text, source)
    def items(self, q): return self.ai.learned_items(q)
    def delete(self, i): return self.ai.delete_item(i)
    def memory_bytes(self): return self.ai.memory()["total"]
    def disk_bytes(self): return sum(os.path.getsize(os.path.join(self.folder, f)) for f in os.listdir(self.folder) if os.path.isfile(os.path.join(self.folder, f)))
    def reply_stats(self): return self.ai.reply_stats()
    def counts(self): return {"sentences": self.ai.memory()["sentences"], "lessons": len(self.ai.groups), "saved": len(self.ai.saved)}
    def snapshot(self): m = self.ai.memory(); return {"memory_bytes": m["total"], "memory_parts": m["parts"], "sentences": m["sentences"], "vocabulary": m["words"], "postings": m["postings"]}


class Engine4(Engine3):
    name = "Frost Code"

    def __init__(self):
        import codeeng
        self.folder = os.path.join(ROOT, "model", "code_solo"); os.makedirs(self.folder, exist_ok=True); self.ai = codeeng.Code(self.folder); self.units = "words"


def si2_clean(x):
    import si2
    return si2.clean(x)


E = {1: Engine1, 2: Engine2, 3: Engine3, 4: Engine4}.get(ARGS.engine, Engine3)()
RUNS = os.path.join(E.folder, "runs.jsonl"); DOCS_FILE = os.path.join(E.folder, "docs.json")
SETTINGS_FILE = os.path.join(E.folder, "app_settings.json")
SETTINGS = {"sources": [n for n, _ in web_search.PROVIDERS], "notice_mb": 50.0}
if os.path.exists(SETTINGS_FILE): SETTINGS.update(json.load(open(SETTINGS_FILE, encoding="utf-8")))


def save_settings(): json.dump(SETTINGS, open(SETTINGS_FILE, "w", encoding="utf-8"))


def load_docs():
    if os.path.exists(DOCS_FILE): return json.load(open(DOCS_FILE, encoding="utf-8"))
    docs = []
    seedr = os.path.join(ROOT, "model", "si_pdf", "results.json")
    if ARGS.engine == 1 and os.path.exists(seedr):
        R = json.load(open(seedr, encoding="utf-8")); s = R["source"]
        docs.append({"name": "bitcoin.pdf", "pages": s["pages"], "words_pdf": len(open(os.path.join(ROOT, "model", "exp_store", "bitcoin.txt"), encoding="utf-8").read().split()), "words_learned": None, "sentences": s["sentences"], "tokens": s["tokens"], "learn_s": R["learn"]["seconds"], "mode": "first run"})
        json.dump(docs, open(DOCS_FILE, "w", encoding="utf-8"), indent=1)
    return docs


DOCS = load_docs()
JOB = {"state": "idle", "name": "", "stop": False}
WEB = {"id": 0, "query": "", "state": "idle", "results": [], "total": 0, "started": 0}
LOCK = threading.Lock()


def log_run(kind, name, metrics, test=None):
    rec = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "engine": E.name, "kind": kind, "name": name, "metrics": metrics, "test": test,
           "after": {**E.snapshot(), "process_ram": ram(), "disk_bytes": E.disk_bytes(), "replies_so_far": E.reply_stats()["n"], "reply_mean_ms": E.reply_stats()["mean_ms"], "counts": E.counts()}}
    with open(RUNS, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
    return rec


# ================================================================== reading a PDF / teaching English / quick check
def clean_pages(pages):
    cnt = collections.Counter()
    for p in pages:
        for ln in {re.sub(r"\d+", "#", l.strip()) for l in p.split("\n") if l.strip()}: cnt[ln] += 1
    lim = max(4, int(0.25 * len(pages))); out = []
    for p in pages:
        keep = [l.strip() for l in p.split("\n") if l.strip() and cnt[re.sub(r"\d+", "#", l.strip())] < lim and not re.search(r"\[Page \d+\]\s*$", l) and not re.fullmatch(r"\d{1,4}", l.strip())]
        out.append(re.sub(r"-\n(?=[a-z])", "", "\n".join(keep)))
    return out


def sentences_of(pages_clean):
    full, starts, off = "", [], 0
    for i, p in enumerate(pages_clean): starts.append((off, i + 1)); full += re.sub(r"\s*\n\s*", " ", p) + " "; off = len(full)
    out, last = [], 0
    bounds = [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z“\"\[(])", full)] + [len(full)]
    for b in bounds:
        s = full[last:b].strip(); pg = next(pg for st, pg in reversed(starts) if st <= last); last = b
        for part in (re.split(r"(?<=;)\s+", s) if len(s.split()) > 90 else [s]):
            if len(part.split()) >= 6 and sum(c.isalpha() or c == " " for c in part) >= 0.7 * len(part): out.append({"text": part, "page": pg})
    return out


def run_pdf(path, watch):
    J = JOB; name = os.path.basename(path); J.clear()
    J.update(state="reading", kind="pdf", name=name, stop=False, pdf_bytes=os.path.getsize(path), pages_total=0, pages_read=0, words_pdf=0, words_done=0, units=0, units_total=0, sentences=0, sentences_total=0, extract_s=0.0, prepare_s=0.0,
             learn_s=0.0, wall_s=0.0, watch=watch, memory_before=E.memory_bytes(), started=time.time())
    try:
        rd = pypdf.PdfReader(path); n = len(rd.pages); J["pages_total"] = n; pages = []
        for i, pg in enumerate(rd.pages):
            if J["stop"]: raise InterruptedError
            t = time.perf_counter(); txt = pg.extract_text() or ""; J["extract_s"] += time.perf_counter() - t; pages.append(txt); J["pages_read"] = i + 1; J["words_pdf"] += len(txt.split())
        J["state"] = "preparing"; sents = sentences_of(clean_pages(pages)); J["sentences_total"] = len(sents)
        prep, J["prepare_s"], J["units_total"] = E.prepare([s["text"] for s in sents]); J["words_total"] = sum(len(s["text"].split()) for s in sents)
        delay = (35.0 / max(1, len(sents))) if watch else 0.0; J["state"] = "adding"; t_wall = time.perf_counter()
        for i, s in enumerate(sents):
            if J["stop"]: raise InterruptedError
            t = time.perf_counter(); E.add_sentence(s["text"], prep[i] if prep else None, f"{name} p.{s['page']}", name); J["learn_s"] += time.perf_counter() - t
            J["sentences"] = i + 1; J["words_done"] += len(s["text"].split()); J["units"] += (len(prep[i]) if prep else len(s["text"])); J["wall_s"] = time.perf_counter() - t_wall; J["page_now"] = s["page"]
            if delay: time.sleep(delay)
        wall = time.perf_counter() - t_wall; mem = E.memory_bytes()
        doc = {"name": name, "pages": n, "pdf_bytes": J["pdf_bytes"], "words_pdf": J["words_pdf"], "words_learned": J["words_done"], "sentences": J["sentences"], "tokens": J["units"], "unit": E.units, "learn_s": J["learn_s"],
               "tokens_per_s": J["units"] / max(J["learn_s"], 1e-9), "words_per_s": J["words_done"] / max(J["learn_s"], 1e-9), "extract_s": J["extract_s"], "tokenise_s": J["prepare_s"], "elapsed_s": wall, "memory_added": mem - J["memory_before"],
               "mode": "watch (slowed on purpose)" if watch else "full speed"}
        DOCS.append(doc); json.dump(DOCS, open(DOCS_FILE, "w", encoding="utf-8"), indent=1); J["doc"] = doc; J["state"] = "done"; log_run("pdf", name, doc)
    except InterruptedError: J["state"] = "stopped"
    except Exception as e:
        import traceback; traceback.print_exc(); J["state"] = "error"; J["error"] = repr(e)


def run_tests():
    ok, fails, ms = 0, [], []
    for q, want in si_lessons.TESTS:
        t = time.perf_counter(); r = E.converse(q); ms.append(1000 * (time.perf_counter() - t))
        if r and r["cat"] == want: ok += 1
        else: fails.append({"q": q, "wanted": want, "got": (r["cat"] if r else None)})
    doc_q = ["what is the longest chain", "what is a cache", "what does a 404 status mean", "how does proof of work stop double spending", "what is a proxy server", "who is satoshi nakamoto"]
    none_q = ["blorp frazzle wumpus", "what is the capital of spain", "who won the cricket world cup in 2011", "translate this into french", "how tall is mount everest"]
    typos = [("helo", "greeting"), ("hii there", "greeting"), ("gud morning", "greeting morning"), ("how r u", "how are you"), ("thanx", "thanks"), ("wat is a nown", "grammar noun"), ("what is a verbb", "grammar verb"), ("goodbye frnd", "farewell"), ("whats ur name", "who are you"), ("pls help me", "please")]
    ty = sum(1 for q, w in typos if (r := E.converse(q)) and r["cat"] == w)
    return {"taught_wordings_tested": len(si_lessons.TESTS), "matched_right_topic": ok, "accuracy": ok / max(1, len(si_lessons.TESTS)), "failures": fails, "mean_ms": sum(ms) / len(ms), "max_ms": max(ms),
            "document_questions_wrongly_answered_as_chat": [q for q in doc_q if E.converse(q)], "unknown_questions_wrongly_answered": [q for q in none_q if E.converse(q)], "document_questions_tested": len(doc_q), "unknown_questions_tested": len(none_q),
            "typo_tests": len(typos), "typo_matched": ty}


def run_lessons(watch):
    J = JOB; J.clear()
    J.update(state="adding", kind="lessons", name="everyday English", stop=False, pdf_bytes=None, pages_total=0, pages_read=0, words_pdf=0, words_done=0, units=0, units_total=0, sentences=0, sentences_total=len(si_lessons.LESSONS) + len(si_lessons.READING),
             extract_s=0.0, prepare_s=0.0, learn_s=0.0, wall_s=0.0, watch=watch, memory_before=E.memory_bytes(), started=time.time(), wordings=0, replies=0)
    try:
        n = J["sentences_total"]; delay = (20.0 / n) if watch else 0.0; t_wall = time.perf_counter(); i = 0
        for cat, utts, reps in si_lessons.LESSONS:
            if J["stop"]: raise InterruptedError
            t = time.perf_counter(); E.add_lesson(cat, utts, reps); J["learn_s"] += time.perf_counter() - t
            i += 1; J["sentences"] = i; J["wordings"] += len(utts); J["replies"] += len(reps); J["words_done"] += sum(len(u.split()) for u in utts) + sum(len(r.split()) for r in reps); J["words_pdf"] = J["words_done"]; J["units"] += E.lesson_tokens(reps); J["wall_s"] = time.perf_counter() - t_wall
            if delay: time.sleep(delay)
        for par in si_lessons.READING:
            if J["stop"]: raise InterruptedError
            t = time.perf_counter(); k = E.add_reading(par); J["learn_s"] += time.perf_counter() - t; i += 1; J["sentences"] = i; J["units"] += k; J["words_done"] += len(par.split()); J["words_pdf"] = J["words_done"]; J["wall_s"] = time.perf_counter() - t_wall
            if delay: time.sleep(delay)
        wall = time.perf_counter() - t_wall; mem = E.memory_bytes(); J["state"] = "checking"; test = run_tests(); J["test"] = test
        doc = {"name": "everyday English", "pages": None, "pdf_bytes": None, "words_pdf": J["words_done"], "words_learned": J["words_done"], "sentences": n, "tokens": J["units"], "unit": E.units, "learn_s": J["learn_s"], "tokens_per_s": J["units"] / max(J["learn_s"], 1e-9),
               "words_per_s": J["words_done"] / max(J["learn_s"], 1e-9), "extract_s": 0.0, "tokenise_s": 0.0, "elapsed_s": wall, "memory_added": mem - J["memory_before"], "mode": "watch (slowed on purpose)" if watch else "full speed",
               "lessons": len(si_lessons.LESSONS), "wordings": J["wordings"], "replies": J["replies"], "reading_paragraphs": len(si_lessons.READING)}
        DOCS.append(doc); json.dump(DOCS, open(DOCS_FILE, "w", encoding="utf-8"), indent=1); J["doc"] = doc; J["state"] = "done"; log_run("lessons", "everyday English", doc, test)
    except InterruptedError: J["state"] = "stopped"
    except Exception as e:
        import traceback; traceback.print_exc(); J["state"] = "error"; J["error"] = repr(e)


# ================================================================== web look-ups
def run_web(query):
    WEB.update(id=WEB["id"] + 1, query=query, state="running", results=[], total=len([n for n, _ in web_search.PROVIDERS if n in SETTINGS["sources"]]), started=time.time(), my_id=WEB["id"] + 1)
    my = WEB["id"]
    def on(r):
        if WEB["id"] == my: WEB["results"].append(r)
    web_search.search_all(query, set(SETTINGS["sources"]), on)
    if WEB["id"] == my:
        WEB["state"] = "done"; WEB["seconds"] = round(time.time() - WEB["started"], 1)
        log_run("web", query, {"query": query, "total_seconds": WEB["seconds"], "sources": [{"source": r["source"], "ok": r["ok"], "seconds": r.get("seconds"), "note": r.get("note", "")[:120]} for r in WEB["results"]]})


PREFIX = re.compile(r"^\s*(please\s+)?(can you\s+)?(find me|find|search for|search|look up|look for|lookup)\s+(for\s+)?", re.I)
PENDING = {"topic": None, "question": None}
ALIASES = {"wikipedia": "Wikipedia", "wiki": "Wikipedia", "duckduckgo": "DuckDuckGo", "duck duck go": "DuckDuckGo", "ddg": "DuckDuckGo", "youtube": "YouTube", "yt": "YouTube", "wiktionary": "Wiktionary", "dictionary": "Wiktionary",
           "open library": "Open Library", "openlibrary": "Open Library", "books": "Open Library", "stack overflow": "Stack Overflow", "stackoverflow": "Stack Overflow", "hacker news": "Hacker News", "hn": "Hacker News", "arxiv": "arXiv", "papers": "arXiv"}
HOW = "You can name a source (Wikipedia, DuckDuckGo, Stack Overflow, YouTube, Open Library, Hacker News), paste a web address, give the path of a PDF on this PC, or say a Project Gutenberg book number."


def parse_source(text):
    t = text.strip(); low = t.lower()
    m = re.search(r"https?://\S+", t)
    if m: return ("url", m.group(0).rstrip(".,)"))
    m = re.search(r'([A-Za-z]:\\[^"<>|?*\n]+?\.(?:pdf|txt))', t, re.I)
    if m and os.path.isfile(m.group(1)): return ("pdf" if m.group(1).lower().endswith(".pdf") else "file", m.group(1))
    m = re.search(r"(?:gutenberg|book)\s*(?:number|no\.?|#|id)?\s*(\d{1,6})", low)
    if m: return ("gutenberg", m.group(1))
    for alias in sorted(ALIASES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", low): return ("provider", ALIASES[alias])
    return None


def curl_text(url, limit=4_000_000):
    out = subprocess.run(["curl", "-sSL", "-m", "40", "--max-filesize", str(limit), "-A", "FrostLang/1.0 (personal research)", url], capture_output=True, timeout=60)
    if out.returncode != 0 or not out.stdout: raise RuntimeError("could not download it")
    return out.stdout.decode("utf-8", "replace")


def html_to_text(h):
    h = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header|aside|form)[^>]*>.*?</\1>", " ", h)
    paras = re.findall(r"(?is)<(?:p|li|h[1-4])[^>]*>(.*?)</(?:p|li|h[1-4])>", h)
    txt = "\n".join(htmllib.unescape(re.sub(r"<[^>]+>", " ", p)) for p in paras) if paras else htmllib.unescape(re.sub(r"<[^>]+>", " ", h))
    return re.sub(r"[ \t]+", " ", txt)


def strip_gutenberg(t):
    a = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", t); b = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", t)
    title = (re.search(r"^Title:\s*(.+)$", t, re.M) or [None, "book"])[1].strip()
    return title, t[a.end():b.start()] if a and b else t


def add_document_text(name, text):
    import lang
    if hasattr(E, "ai") and hasattr(E.ai, "learn_text"):                 # Frost Lang / Code keep where each paragraph ends
        n0 = len(E.ai.sents); E.ai.learn_text(text, name); return len(E.ai.sents) - n0
    sents = lang.split_sentences(text); prep, _, _ = E.prepare(sents)
    for i, st in enumerate(sents): E.add_sentence(st, prep[i] if prep else None, name, name)
    return len(sents)


from uploads import extract_upload


def run_upload(name, raw):
    t0 = time.time(); mem0 = E.memory_bytes(); t1 = time.perf_counter(); text, pages = extract_upload(name, raw); ex = time.perf_counter() - t1
    if len(text.split()) < 5: return {"kind": "told", "text": f"I could not find readable text in {name}. If it is a scan, I cannot read pictures of words."}
    t1 = time.perf_counter(); n = add_document_text(name, text); ls = time.perf_counter() - t1; w = len(text.split())
    doc = {"name": name, "pages": pages, "pdf_bytes": len(raw), "words_pdf": w, "words_learned": w, "sentences": n, "tokens": w, "unit": E.units, "learn_s": ls, "words_per_s": w / max(ls, 1e-9), "tokens_per_s": w / max(ls, 1e-9), "extract_s": ex, "tokenise_s": 0.0, "elapsed_s": ex + ls, "memory_added": E.memory_bytes() - mem0, "mode": "upload"}
    DOCS.append(doc); json.dump(DOCS, open(DOCS_FILE, "w", encoding="utf-8"), indent=1); log_run("upload", name, doc)
    return {"kind": "learned", "text": f"I read {name} and saved {n:,} sentences ({w:,} words) in {ex + ls:.1f} s.", "preview": "", "item": "d:" + name}


def learn_from(spec, topic, question):
    kind, val = spec; t0 = time.time(); mem0 = E.memory_bytes()
    try:
        if kind == "provider":
            fn = dict(web_search.PROVIDERS)[val]; r = fn(topic); text = (r.get("text") or r.get("summary") or "").strip()
            if not r["ok"] or not text: return {"kind": "told", "text": f"{val} had nothing useful about {topic}. Where else can I look? {HOW}"}
            item = E.remember(question, text, val); log_run("remember", f"{val}: {question}", {"question": question, "source": val, "words": len(text.split()), "remember_ms": (time.time() - t0) * 1000, "memory_added": E.memory_bytes() - mem0})
            return {"kind": "learned", "text": f"I read about {topic} on {val} and saved it.", "preview": text[:420], "item": item, "then": True}
        if kind == "url":
            h = curl_text(val); title = (re.search(r"(?is)<title[^>]*>(.*?)</title>", h) or [0, val])[1]; name = "web page: " + " ".join(htmllib.unescape(title).split())[:60]
            n = add_document_text(name, html_to_text(h)); log_run("web page", name, {"url": val, "sentences": n, "seconds": time.time() - t0, "memory_added": E.memory_bytes() - mem0})
            return {"kind": "learned", "text": f"I read the web page and saved {n} sentences from it.", "preview": "", "item": "d:" + name, "then": True} if n else {"kind": "told", "text": "I could not find readable sentences on that page."}
        if kind == "gutenberg":
            title, body = strip_gutenberg(curl_text(f"https://www.gutenberg.org/cache/epub/{val}/pg{val}.txt")); name = f"{title} (Gutenberg {val})"; n = add_document_text(name, body)
            doc = {"name": name, "pages": None, "words_pdf": len(body.split()), "words_learned": len(body.split()), "sentences": n, "tokens": len(body.split()), "unit": E.units, "learn_s": time.time() - t0, "mode": "book"}
            DOCS.append(doc); json.dump(DOCS, open(DOCS_FILE, "w", encoding="utf-8"), indent=1); log_run("book", name, {**doc, "memory_added": E.memory_bytes() - mem0})
            return {"kind": "learned", "text": f"I read {title} and saved {n:,} sentences from it.", "preview": "", "item": "d:" + name, "then": True}
        if kind in ("pdf", "file"):
            if kind == "file":
                name = os.path.basename(val); body = open(val, encoding="utf-8", errors="replace").read(); n = add_document_text(name, body); log_run("file", name, {"sentences": n, "seconds": time.time() - t0})
                return {"kind": "learned", "text": f"I read {name} and saved {n:,} sentences.", "preview": "", "item": "d:" + name, "then": True}
            if JOB.get("state") in ("reading", "preparing", "adding", "checking"): return {"kind": "told", "text": "I am busy reading something else. Ask again in a moment."}
            threading.Thread(target=run_pdf, args=(val, False), daemon=True).start()
            return {"kind": "told", "text": f"Reading {os.path.basename(val)} now. The right side shows how far I am; ask me again when it is done."}
    except Exception as e:
        return {"kind": "told", "text": f"I could not read from there ({type(e).__name__}). Where else can I look?"}


def ask_text(topic): return f"I have not read about {topic} yet. Where can I learn it from? {HOW}"


def ask(q):
    m = PREFIX.match(q)
    if m and q[m.end():].strip():
        query = q[m.end():].strip(); threading.Thread(target=run_web, args=(query,), daemon=True).start(); return {"kind": "search", "query": query, "id": WEB["id"] + 1}
    if PENDING["topic"]:
        spec = parse_source(q)
        if spec:
            topic, question = PENDING["topic"], PENDING["question"]; r = learn_from(spec, topic, question)
            if r.get("then"): PENDING.update(topic=None, question=None); r["answer"] = E.chat(question)
            return r
    r = E.chat(q)
    if r is None or r.get("kind") == "ask":
        topic = (r or {}).get("topic") or q.strip(); PENDING.update(topic=topic, question=q); return {"kind": "ask", "text": ask_text(topic)}
    if r.get("ask"): PENDING.update(topic=r.get("topic") or q.strip(), question=q); r["follow"] = f"Want a better answer about {r.get('topic') or q.strip()}? Tell me where to read about it. {HOW}"
    return r


def remember(body):
    r = WEB["results"][int(body["index"])]; text = (body.get("text") or r.get("text") or r.get("summary") or "").strip(); q = (body.get("question") or WEB["query"]).strip()
    if not text: return {"error": "there is nothing to remember in that card"}
    t = time.perf_counter(); mem0 = E.memory_bytes(); E.remember(q, text, r["source"]); ms = (time.perf_counter() - t) * 1000
    log_run("remember", f"{r['source']}: {q}", {"question": q, "source": r["source"], "words": len(text.split()), "characters": len(text), "remember_ms": ms, "memory_added": E.memory_bytes() - mem0})
    return {"ok": True}


# ================================================================== state
EXAMPLES = {4: ["Write a python program to reverse a string", "Write a c program to print hello world", "Build a todo web app", "Make a blue notes app called Study notes", "Stop all apps"]}


def run_codelessons():
    """teaches the coding lessons, then checks wordings and runs every Python snippet; one row in runs.jsonl"""
    import code_data, subprocess, tempfile
    J = JOB; J.clear(); J.update(state="adding", kind="lessons", name="coding lessons", stop=False, pdf_bytes=None, pages_total=0, pages_read=0, words_pdf=0, words_done=0, units=0, units_total=0, sentences=0, sentences_total=68,
                                   extract_s=0.0, prepare_s=0.0, learn_s=0.0, wall_s=0.0, watch=False, memory_before=E.memory_bytes(), started=time.time(), wordings=0, replies=0)
    try:
        t_wall = time.perf_counter(); r = E.ai.teach_code(); J.update(learn_s=r["seconds"], sentences=r["lessons"], wordings=r["wordings"], words_done=r["words"], words_pdf=r["words"], units=r["words"])
        J["state"] = "checking"; ok = 0; miss = []
        for q, want in code_data.CODE_TESTS:
            m = E.ai.lesson(q)
            if m and m["cat"] == want: ok += 1
            else: miss.append([q, want, m["cat"] if m else None])
        ran = bad = 0
        with tempfile.TemporaryDirectory() as tmp:
            for lang_, phrases, why, code, out in code_data.TASKS:
                if lang_ != code_data.PY or out is None: continue
                pr = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20, cwd=tmp, encoding="utf-8")
                if pr.returncode == 0 and pr.stdout.strip() == out.strip(): ran += 1
                else: bad += 1
        wall = time.perf_counter() - t_wall; mem = E.memory_bytes(); test = {"code_wordings_found": ok, "code_wordings_asked": len(code_data.CODE_TESTS), "python_snippets_run_and_matched": ran, "python_snippets_failed": bad, "misses": miss}
        doc = {"name": "coding lessons", "pages": None, "pdf_bytes": None, "words_pdf": r["words"], "words_learned": r["words"], "sentences": r["lessons"], "tokens": r["words"], "unit": "words", "learn_s": r["seconds"], "tokens_per_s": r["words"] / max(r["seconds"], 1e-9), "words_per_s": r["words"] / max(r["seconds"], 1e-9),
               "extract_s": 0.0, "tokenise_s": 0.0, "elapsed_s": wall, "memory_added": mem - J["memory_before"], "mode": "full speed", "lessons": r["lessons"], "wordings": r["wordings"], "replies": r["lessons"], "reading_paragraphs": 0}
        DOCS.append(doc); json.dump(DOCS, open(DOCS_FILE, "w", encoding="utf-8"), indent=1); J["doc"] = doc; J["test"] = test; J["state"] = "done"; log_run("lessons", "coding lessons", doc, test)
    except Exception as e:
        import traceback; traceback.print_exc(); J["state"] = "error"; J["error"] = repr(e)


def state():
    docs = DOCS; rs = E.reply_stats(); last = [d for d in docs if d.get("words_per_s") and d.get("pages")]
    return {"name": E.name, "job": {k: v for k, v in JOB.items()}, "web": {k: v for k, v in WEB.items() if k != "results"}, "docs": docs, "counts": E.counts(), "memory_bytes": E.memory_bytes(), "disk_bytes": E.disk_bytes(), "ram": ram(),
            "reply": rs, "settings": SETTINGS, "all_sources": [n for n, _ in web_search.PROVIDERS], "unit": E.units, "examples": EXAMPLES.get(ARGS.engine), "can_code": ARGS.engine == 4, "last_doc_speed": (last[-1]["words_per_s"] if last else None)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, b, ctype, code=200):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def _json(self, o, code=200): self._send(json.dumps(o, ensure_ascii=False, default=float).encode("utf-8"), "application/json; charset=utf-8", code)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path == "/": self._send(open(os.path.join(HERE, "si_app.html"), "rb").read(), "text/html; charset=utf-8")
            elif u.path.startswith("/download/") and u.path.endswith(".zip"):
                f = os.path.join(ROOT, "model", "code_projects", os.path.basename(u.path))
                if not os.path.isfile(f): return self._json({"error": "not found"}, 404)
                b = open(f, "rb").read(); self.send_response(200); self.send_header("Content-Type", "application/zip"); self.send_header("Content-Length", str(len(b))); self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(f)}"'); self.end_headers(); self.wfile.write(b)
            elif u.path == "/api/state": self._json(state())
            elif u.path == "/api/web": self._json({k: v for k, v in WEB.items()})
            elif u.path == "/api/items": self._json(E.items(q.get("q", [""])[0]))
            else: self._json({"error": "not found"}, 404)
        except Exception as e: self._json({"error": repr(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path).path
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            busy = JOB.get("state") in ("reading", "preparing", "adding", "checking")
            if u == "/api/ask": return self._json(ask(body["q"].strip()))
            if u == "/api/remember": return self._json(remember(body))
            if u == "/api/upload":
                import base64
                if busy: return self._json({"error": "I am busy reading something else."}, 409)
                raw = base64.b64decode(body["data"]); nm = os.path.basename(body.get("name") or "file")
                try: r = run_upload(nm, raw)
                except Exception as e: r = {"kind": "told", "text": f"I could not read {nm} ({type(e).__name__})."}
                if r.get("kind") == "learned" and body.get("question"): r["answer"] = E.chat(body["question"])
                return self._json(r)
            if u == "/api/reset":
                PENDING.update(topic=None, question=None)
                if hasattr(E, "ai") and hasattr(E.ai, "reset_chat"): E.ai.reset_chat()
                return self._json({"ok": True})
            if u == "/api/book": return self._json(learn_from(("gutenberg", str(body["id"])), "", ""))
            if u == "/api/delete": n = E.delete(body["id"]); log_run("delete", body["id"], {"removed": n}); return self._json({"ok": True, "removed": n})
            if u == "/api/ingest":
                path = body.get("path", "").strip().strip('"')
                if not path.lower().endswith(".pdf") or not os.path.isfile(path): return self._json({"error": "That is not a PDF file on this PC."}, 400)
                if busy: return self._json({"error": "I am busy reading something else."}, 409)
                threading.Thread(target=run_pdf, args=(path, bool(body.get("watch", True))), daemon=True).start(); return self._json({"ok": True})
            if u == "/api/lessons":
                if busy: return self._json({"error": "I am busy reading something else."}, 409)
                threading.Thread(target=run_lessons, args=(bool(body.get("watch", True)),), daemon=True).start(); return self._json({"ok": True})
            if u == "/api/codelessons" and ARGS.engine == 4:
                if busy: return self._json({"error": "I am busy reading something else."}, 409)
                threading.Thread(target=run_codelessons, daemon=True).start(); return self._json({"ok": True})
            if u == "/api/check":
                t = run_tests(); log_run("check", "quick check", {"wordings": t["taught_wordings_tested"]}, t); return self._json(t)
            if u == "/api/stop": JOB["stop"] = True; return self._json({"ok": True})
            if u == "/api/settings":
                if "sources" in body: SETTINGS["sources"] = [s for s in body["sources"] if s in [n for n, _ in web_search.PROVIDERS]]
                if "notice_mb" in body: SETTINGS["notice_mb"] = max(0.1, float(body["notice_mb"]))
                save_settings(); return self._json({"ok": True})
            self._json({"error": "not found"}, 404)
        except Exception as e:
            import traceback; traceback.print_exc(); self._json({"error": repr(e)}, 500)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"READY  {E.name}  http://127.0.0.1:{PORT}   store {E.folder}   {E.counts()}", flush=True)
    srv.serve_forever()
