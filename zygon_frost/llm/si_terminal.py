"""ZYGON solo terminal.   python si_terminal.py   ->   http://127.0.0.1:8820

The solo ZYGON: exponent identities + angle wheels + hooks. No 3B model, no embedding table, no matrices.
It starts as the SI that learned bitcoin.pdf on its own (copied from model/si_pdf) and can learn more PDFs live; every number is measured.
"""
import collections, json, os, re, shutil, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import pypdf
from gguf import GGUF
from tokenizer import Tokenizer
import angle_ai as A
import si_lessons

sys.stdout.reconfigure(encoding="utf-8")
PORT = 8820
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
SOLO, SEED = os.path.join(ROOT, "model", "si_solo"), os.path.join(ROOT, "model", "si_pdf")
DOCS_FILE = os.path.join(SOLO, "docs.json"); RUNS_FILE = os.path.join(SOLO, "runs.jsonl")
DEFAULT_PDF = os.path.join(ROOT, "model", "si_live", "source", "rfc2616.pdf")
try:
    import scan
except Exception:
    scan = None

if not os.path.exists(os.path.join(SOLO, "learned.jsonl")):
    shutil.copytree(SEED, SOLO, dirs_exist_ok=True)                    # the SI that learned bitcoin.pdf alone
TOK = Tokenizer(GGUF().meta)
AI = A.AngleAI(TOK, SOLO)
LOCK = threading.Lock()


def seed_docs():
    if os.path.exists(DOCS_FILE):
        return json.load(open(DOCS_FILE, encoding="utf-8"))
    docs = []
    rj = os.path.join(SEED, "results.json")
    if os.path.exists(rj):
        R = json.load(open(rj, encoding="utf-8")); s, l = R["source"], R["learn"]
        words = len(open(os.path.join(ROOT, "model", "exp_store", "bitcoin.txt"), encoding="utf-8").read().split())
        docs.append({"name": "bitcoin.pdf", "pages": s["pages"], "pdf_bytes": s["pdf_bytes"], "words_pdf": words, "words_learned": None, "sentences": s["sentences"], "tokens": s["tokens"],
                     "learn_s": l["seconds"], "tokens_per_s": l["tokens_per_s"], "sentences_per_s": l["sentences_per_s"], "extract_s": R["extract_s"], "memory_added": R["memory"]["estimated_total_bytes"], "mode": "solo run (full speed)"})
        json.dump(docs, open(DOCS_FILE, "w", encoding="utf-8"), indent=1)
    return docs


DOCS = seed_docs()
JOB = {"state": "idle", "name": "", "stop": False}
CURVE = []


def snapshot():
    m = AI.memory(); rp = AI.agg.get("reply"); rt = AI.agg.get("reply_tokens")
    return {"memory_bytes": m["total"], "memory_parts": m["parts"], "wheels": m["wheels"], "sectors": m["sectors"], "hooks": m["hooks"], "sentences": m["paths"], "lessons": len(AI.groups), "identities": len(AI.cell_of),
            "process_ram": ram(), "disk_bytes": AI.disk_bytes(), "replies_so_far": rp[0] if rp else 0, "reply_mean_ms": (rp[1] / rp[0]) if rp else None, "reply_tokens_per_s": (rt[1] / (rp[1] / 1000)) if rp and rt and rp[1] else None}


def log_run(kind, name, metrics, test=None):
    """every teach or test run is appended here with all its numbers (the research log)"""
    rec = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "name": name, "metrics": metrics, "test": test, "after": snapshot()}
    with open(RUNS_FILE, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
    return rec


# ---------------------------------------------------------------- reading a PDF
def clean_pages(pages):
    """drop headers/footers (lines that repeat on many pages once digits are masked), join broken lines"""
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
    for i, p in enumerate(pages_clean):
        starts.append((off, i + 1)); full += re.sub(r"\s*\n\s*", " ", p) + " "; off = len(full)
    out, last = [], 0
    bounds = [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z“\"\[(])", full)] + [len(full)]
    page_at = lambda o: next(pg for st, pg in reversed(starts) if st <= o)
    for b in bounds:
        s = full[last:b].strip(); pg = page_at(last); last = b
        for part in (re.split(r"(?<=;)\s+", s) if len(s.split()) > 90 else [s]):
            letters = sum(c.isalpha() or c == " " for c in part)
            if len(part.split()) >= 6 and letters >= 0.7 * len(part): out.append({"text": part, "page": pg})
    return out


def run_ingest(path, watch):
    J = JOB; name = os.path.basename(path); CURVE.clear()
    J.clear(); J.update(state="reading", name=name, stop=False, pdf_bytes=os.path.getsize(path), pages_total=0, pages_read=0, words_pdf=0, words_learned=0, tokens=0, tokens_total=0, sentences=0, sentences_total=0,
                        extract_s=0.0, tokenise_s=0.0, learn_s=0.0, wall_s=0.0, feed=[], watch=watch, memory_before=AI.memory()["total"], started=time.time())
    try:
        rd = pypdf.PdfReader(path); n = len(rd.pages); J["pages_total"] = n; pages = []
        for i, pg in enumerate(rd.pages):
            if J["stop"]: raise InterruptedError
            t = time.perf_counter(); txt = pg.extract_text() or ""; J["extract_s"] += time.perf_counter() - t; pages.append(txt); J["pages_read"] = i + 1; J["words_pdf"] += len(txt.split())
        J["state"] = "preparing"; sents = sentences_of(clean_pages(pages)); J["sentences_total"] = len(sents)
        t = time.perf_counter(); ids_list = [TOK.encode(s["text"]) for s in sents]; J["tokenise_s"] = time.perf_counter() - t; J["tokens_total"] = sum(len(i) for i in ids_list)
        J["words_learned_total"] = sum(len(s["text"].split()) for s in sents); delay = (35.0 / max(1, len(sents))) if watch else 0.0
        J["state"] = "learning"; t_wall = time.perf_counter(); nxt = t_wall
        for i, (s, ids) in enumerate(zip(sents, ids_list)):
            if J["stop"]: raise InterruptedError
            t = time.perf_counter(); AI.learn_qa(s["text"], ids, f"{name} p.{s['page']}", ended=True, pull=False); J["learn_s"] += time.perf_counter() - t
            J["sentences"] = i + 1; J["tokens"] += len(ids); J["words_learned"] += len(s["text"].split()); J["wall_s"] = time.perf_counter() - t_wall
            J["feed"] = (J["feed"] + [{"i": i + 1, "page": s["page"], "text": s["text"][:140], "tokens": len(ids)}])[-14:]
            if i % max(1, len(sents) // 200) == 0 or i + 1 == len(sents): CURVE.append([J["tokens"], AI.memory()["total"], round(J["learn_s"], 4)])
            if delay: time.sleep(delay)
        wall = time.perf_counter() - t_wall; J["wall_s"] = wall
        mem = AI.memory()["total"]
        doc = {"name": name, "pages": n, "pdf_bytes": J["pdf_bytes"], "words_pdf": J["words_pdf"], "words_learned": J["words_learned"], "sentences": J["sentences"], "tokens": J["tokens"], "learn_s": J["learn_s"],
               "tokens_per_s": J["tokens"] / max(J["learn_s"], 1e-9), "sentences_per_s": J["sentences"] / max(J["learn_s"], 1e-9), "extract_s": J["extract_s"], "tokenise_s": J["tokenise_s"], "elapsed_s": wall,
               "memory_added": mem - J["memory_before"], "mode": "watch (slowed on purpose)" if watch else "full speed"}
        DOCS.append(doc); json.dump(DOCS, open(DOCS_FILE, "w", encoding="utf-8"), indent=1); J["state"] = "done"; J["doc"] = doc; log_run("pdf", name, doc)
    except InterruptedError:
        J["state"] = "stopped"
    except Exception as e:
        import traceback; traceback.print_exc(); J["state"] = "error"; J["error"] = repr(e)



def run_tests():
    """wordings that were never taught: does the right lesson answer? and are document questions left to the documents?"""
    ok, fails, ms = 0, [], []
    for q, want in si_lessons.TESTS:
        t = time.perf_counter(); r = AI.converse(q); ms.append(1000 * (time.perf_counter() - t))
        if r and r["cat"] == want: ok += 1
        else: fails.append({"q": q, "wanted": want, "got": (r["cat"] if r else None)})
    doc_q = ["what is the longest chain", "what is a cache", "what does a 404 status mean", "how does proof of work stop double spending", "what is a proxy server", "who is satoshi nakamoto"]
    leak = [q for q in doc_q if AI.converse(q)]
    none_q = ["blorp frazzle wumpus", "what is the capital of spain", "who won the cricket world cup in 2011", "translate this into french", "how tall is mount everest"]
    wrong = [q for q in none_q if AI.converse(q)]
    return {"taught_wordings_tested": len(si_lessons.TESTS), "matched_right_topic": ok, "accuracy": ok / max(1, len(si_lessons.TESTS)), "failures": fails, "mean_ms": sum(ms) / len(ms), "max_ms": max(ms),
            "document_questions_wrongly_answered_as_chat": leak, "unknown_questions_wrongly_answered": wrong, "document_questions_tested": len(doc_q), "unknown_questions_tested": len(none_q)}


def run_lessons(watch, force=False):
    J = JOB; J.clear()
    J.update(state="learning", name="english lessons", stop=False, pdf_bytes=None, pages_total=0, pages_read=0, words_pdf=0, words_learned=0, tokens=0, tokens_total=0, sentences=0, sentences_total=len(si_lessons.LESSONS) + len(si_lessons.READING),
             extract_s=0.0, tokenise_s=0.0, learn_s=0.0, wall_s=0.0, feed=[], watch=watch, memory_before=AI.memory()["total"], started=time.time(), lessons=True, wordings=0, replies=0)
    CURVE.clear()
    try:
        n = J["sentences_total"]; delay = (20.0 / n) if watch else 0.0; t_wall = time.perf_counter(); i = 0
        for cat, utts, reps in si_lessons.LESSONS:
            if J["stop"]: raise InterruptedError
            t = time.perf_counter(); AI.learn_lesson(cat, utts, reps, "lesson: " + cat); J["learn_s"] += time.perf_counter() - t
            i += 1; J["sentences"] = i; J["wordings"] += len(utts); J["replies"] += len(reps); J["words_learned"] += sum(len(u.split()) for u in utts) + sum(len(r.split()) for r in reps); J["words_pdf"] = J["words_learned"]
            J["tokens"] += sum(len(TOK.encode(r)) + 1 for r in reps); J["wall_s"] = time.perf_counter() - t_wall
            J["feed"] = (J["feed"] + [{"i": i, "page": "lesson", "text": f"{utts[0]}  ->  {reps[0][:80]}", "tokens": 0}])[-14:]
            if i % 4 == 0: CURVE.append([J["tokens"], AI.memory()["total"], round(J["learn_s"], 4)])
            if delay: time.sleep(delay)
        for par in si_lessons.READING:
            if J["stop"]: raise InterruptedError
            t = time.perf_counter(); k = AI.learn_text(par, "reading"); J["learn_s"] += time.perf_counter() - t
            i += 1; J["sentences"] = i; J["tokens"] += k; J["words_learned"] += len(par.split()); J["words_pdf"] = J["words_learned"]; J["wall_s"] = time.perf_counter() - t_wall
            J["feed"] = (J["feed"] + [{"i": i, "page": "reading", "text": par[:100], "tokens": k}])[-14:]
            if delay: time.sleep(delay)
        wall = time.perf_counter() - t_wall; mem = AI.memory()["total"]
        J["state"] = "testing"; test = run_tests(); J["test"] = test
        doc = {"name": "english lessons", "pages": None, "pdf_bytes": None, "words_pdf": J["words_learned"], "words_learned": J["words_learned"], "sentences": len(si_lessons.LESSONS) + len(si_lessons.READING), "tokens": J["tokens"], "learn_s": J["learn_s"],
               "tokens_per_s": J["tokens"] / max(J["learn_s"], 1e-9), "sentences_per_s": J["sentences"] / max(J["learn_s"], 1e-9), "extract_s": 0.0, "tokenise_s": 0.0, "elapsed_s": wall, "memory_added": mem - J["memory_before"],
               "mode": "watch (slowed on purpose)" if watch else "full speed", "lessons": len(si_lessons.LESSONS), "wordings": J["wordings"], "replies": J["replies"], "reading_paragraphs": len(si_lessons.READING)}
        DOCS.append(doc); json.dump(DOCS, open(DOCS_FILE, "w", encoding="utf-8"), indent=1); J["doc"] = doc; J["state"] = "done"; log_run("lessons", "english lessons", doc, test)
    except InterruptedError:
        J["state"] = "stopped"
    except Exception as e:
        import traceback; traceback.print_exc(); J["state"] = "error"; J["error"] = repr(e)


def answer(q):
    r = AI.converse(q)
    if r: return r
    r = AI.compose(q)
    if r.get("found"): r["kind"] = "document"; return r
    r["kind"] = "none"; return r


# ---------------------------------------------------------------- state
def ram():
    try:
        return scan.mem_info()["process_working_set"] if scan else None
    except Exception:
        return None


def state():
    m = AI.memory(); rp = AI.agg.get("reply"); rt = AI.agg.get("reply_tokens"); rn = AI.agg.get("reply_none")
    reply = {"n": rp[0] if rp else 0, "not_found": rn[0] if rn else 0, "mean_ms": (rp[1] / rp[0]) if rp else None, "fastest_ms": rp[2] if rp else None, "slowest_ms": rp[3] if rp else None,
             "tokens_per_s": (rt[1] / (rp[1] / 1000)) if rp and rt and rp[1] else None, "mean_tokens": (rt[1] / rt[0]) if rt else None}
    docs = DOCS
    return {"job": {k: v for k, v in JOB.items()}, "curve": CURVE[-300:], "docs": docs, "reply": reply,
            "total": {"pages": sum((d["pages"] or 0) for d in docs), "words": sum(d.get("words_pdf") or 0 for d in docs), "tokens": sum(d["tokens"] for d in docs), "sentences": sum(d["sentences"] for d in docs), "learn_s": sum(d["learn_s"] for d in docs)},
            "lessons": len(AI.groups), "memory": {"bytes": m["total"], "parts": m["parts"], "wheels": m["wheels"], "sectors": m["sectors"], "hooks": m["hooks"], "sentences": m["paths"], "identities": len(AI.cell_of), "disk_bytes": AI.disk_bytes(), "process_ram": ram(),
                       "bytes_per_token": m["total"] / max(1, sum(d["tokens"] for d in docs))}, "default_pdf": DEFAULT_PDF if os.path.exists(DEFAULT_PDF) else ""}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, b, ctype, code=200):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def _json(self, o, code=200):
        self._send(json.dumps(o, ensure_ascii=False, default=float).encode("utf-8"), "application/json; charset=utf-8", code)

    def do_GET(self):
        u = urlparse(self.path).path
        try:
            if u == "/": self._send(open(os.path.join(HERE, "si_terminal.html"), "rb").read(), "text/html; charset=utf-8")
            elif u == "/cube_view.js": self._send(open(os.path.join(HERE, "cube_view.js"), "rb").read(), "application/javascript; charset=utf-8")
            elif u == "/api/state": self._json(state())
            elif u == "/api/cube": self._json(AI.cube())
            else: self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": repr(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path).path
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if u == "/api/ask": return self._json(answer(body["q"].strip()))
            if u == "/api/lessons":
                if JOB.get("state") in ("reading", "preparing", "learning", "testing"): return self._json({"error": "already learning"}, 409)
                if AI.groups and not body.get("force"): return self._json({"error": "the lessons are already taught (%d lessons)" % len(AI.groups)}, 409)
                threading.Thread(target=run_lessons, args=(bool(body.get("watch", True)),), daemon=True).start(); return self._json({"ok": True})
            if u == "/api/test":
                t = run_tests(); log_run("test", "english self-test", {"tests": t["taught_wordings_tested"]}, t); return self._json(t)
            if u == "/api/ingest":
                path = body.get("path", "").strip().strip('"')
                if not path.lower().endswith(".pdf") or not os.path.isfile(path): return self._json({"error": "not a PDF file on this PC: " + path}, 400)
                if JOB.get("state") in ("reading", "preparing", "learning", "testing"): return self._json({"error": "already learning a PDF"}, 409)
                threading.Thread(target=run_ingest, args=(path, bool(body.get("watch", True))), daemon=True).start(); return self._json({"ok": True})
            if u == "/api/stop": JOB["stop"] = True; return self._json({"ok": True})
            self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": repr(e)}, 500)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"READY  http://127.0.0.1:{PORT}   (solo ZYGON: {len(AI.paths)} sentences, {len(AI.cell_of)} token identities)", flush=True)
    srv.serve_forever()
