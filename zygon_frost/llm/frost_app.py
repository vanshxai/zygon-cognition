"""Frost 1.0 app: chat window + left sidebar (new chats, chat history, reports).   python frost_app.py  ->  http://127.0.0.1:8810

Stdlib http.server + numpy engine only. The model is loaded on demand (button in the sidebar) because it needs ~13 GB of RAM.
Chats are stored as JSON files in <project>/chats. Reports are the saved .html files already in the project.
"""
import gc, glob, json, os, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import numpy as np
import scan
import path_locker as PL
import locker_report
import angle_ai
from engine import Model, chatml

sys.stdout.reconfigure(encoding="utf-8")
PORT = 8810
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CHATS = os.path.join(ROOT, "chats")
os.makedirs(CHATS, exist_ok=True)
SYSTEM = "You are Frost 1.0, a helpful assistant."
MAX_CTX = 6000                                   # tokens kept in the KV cache (oldest turns are dropped beyond this)
NEED_RAM = 13 * 2 ** 30

LK = PL.Locker()
SK = dict(temp=0.7, top_k=20, top_p=0.8)
ST = {"system": SYSTEM, "model": None, "loading": False, "error": "", "cache": None, "cached": [], "stop": False, "busy": False}
LOCK = threading.Lock()


# ------------------------------------------------------------------ model
def load_model(patch=True):
    mi = scan.mem_info()
    if mi["system_available"] < NEED_RAM:
        ST["error"] = (f"Not enough free RAM to load the model: {mi['system_available'] / 2**30:.1f} GiB free, about 13 GiB needed. "
                       "Stop whatever else holds the model (training run, the lab server on port 8800) and try again.")
        return
    ST.update(loading=True, error="")
    try:
        ST["model"] = Model(verbose=False, patch=patch)
        ST["cache"], ST["cached"] = None, []
    except Exception as e:
        ST["error"] = "load failed: " + repr(e)
    ST["loading"] = False


def unload_model():
    ST.update(model=None, cache=None, cached=[])
    gc.collect()


def state():
    mi = scan.mem_info(); m = ST["model"]
    return {"loaded": m is not None, "loading": ST["loading"], "error": ST["error"], "busy": ST["busy"],
            "patched": list(m.patched) if m is not None else [], "ram_free": mi["system_available"], "ram_total": mi["system_total"],
            "process_ram": mi["process_working_set"], "system": ST["system"], "frost_patch_present": os.path.isdir(os.path.join(ROOT, "model", "frost_1.0"))}


# ------------------------------------------------------------------ chats
def chat_path(cid):
    return os.path.join(CHATS, "".join(c for c in cid if c.isalnum() or c in "-_") + ".json")


def load_chat(cid):
    return json.load(open(chat_path(cid), encoding="utf-8"))


def save_chat(c):
    c["updated"] = time.time()
    json.dump(c, open(chat_path(c["id"]), "w", encoding="utf-8"), ensure_ascii=False)


def list_chats():
    out = []
    for f in glob.glob(os.path.join(CHATS, "*.json")):
        try:
            c = json.load(open(f, encoding="utf-8")); out.append({"id": c["id"], "title": c["title"], "updated": c["updated"], "n": len(c["messages"])})
        except Exception:
            pass
    return sorted(out, key=lambda c: -c["updated"])


def new_chat():
    cid = time.strftime("%Y%m%d-%H%M%S") + "-" + str(int(time.time() * 1000) % 1000)
    c = {"id": cid, "title": "New chat", "created": time.time(), "updated": time.time(), "messages": []}
    save_chat(c)
    return c


# ------------------------------------------------------------------ reports
def report_files():
    """key -> absolute path. Only these files can ever be served."""
    files = {}
    p = os.path.join(ROOT, "capabilities_report.html")
    if os.path.exists(p): files["capabilities_report.html"] = p
    for f in glob.glob(os.path.join(ROOT, "runs", "*.html")):
        files["runs/" + os.path.basename(f)] = f
    p = os.path.join(HERE, "viz.html")
    if os.path.exists(p): files["llm/viz.html"] = p
    for fn in ("software_index.html", "software_neuron_map.html", "neuron_capabilities_report.html", "capital_of_france_board.html", "capital_of_france_paper.html", "the_layer1_paper.html", "exp_store_3d.html", "angle_student_report.html", "si_pdf_report.html"):
        p = os.path.join(ROOT, fn)
        if os.path.exists(p): files[fn] = p
    p = os.path.join(ROOT, "neuron_size_map.html")
    if os.path.exists(p): files["neuron_size_map.html"] = p
    p = os.path.join(ROOT, "text_weights_scan_report.html")
    if os.path.exists(p): files["text_weights_scan_report.html"] = p
    return files


def list_reports():
    fs = report_files(); groups = []
    def item(k, title):
        st = os.stat(fs[k]); return {"key": k, "title": title, "size": st.st_size, "mtime": st.st_mtime}
    if "capabilities_report.html" in fs:
        groups.append({"group": "Capabilities", "items": [item("capabilities_report.html", "Frost 1.0 capabilities report")]})
    runs = sorted([k for k in fs if k.startswith("runs/")], key=lambda k: -os.stat(fs[k]).st_mtime)
    if runs:
        groups.append({"group": "Saved runs", "items": [item(k, os.path.splitext(k[5:])[0]) for k in runs]})
    if "text_weights_scan_report.html" in fs:
        its = [item("text_weights_scan_report.html", "Text-reply neurons (text vs code scan)")]
        if "neuron_size_map.html" in fs:
            its.insert(0, item("neuron_size_map.html", "3D map: neurons vs prompt size (10 prompts)"))
        if "neuron_capabilities_report.html" in fs:
            its.insert(0, item("neuron_capabilities_report.html", "What the other 98% of neurons do (census + switch-off test)"))
        if "software_neuron_map.html" in fs:
            its.insert(0, item("software_neuron_map.html", "3D map: coding neurons in 10 working programs"))
        groups.append({"group": "Weight scans", "items": its})
    if "software_index.html" in fs:
        groups.append({"group": "Software (10 programs)", "items": [item("software_index.html", "The 10 .exe programs: what, where, how tested")]})
    groups.append({"group": "Neuron path locker", "items": [{"key": "live:path_locker", "title": f"Locked and discovered paths ({len(LK.rec)})", "size": 0, "mtime": time.time()},
                                                          {"key": "live:code_paths", "title": "Code function index (common vs unique + neurons)", "size": 0, "mtime": time.time()}]})
    mb = [item(k, t) for k, t in (("capital_of_france_paper.html", "Worked by hand on paper (LaTeX): \"What's the capital of France?\""), ("the_layer1_paper.html", "Layer 1 for the word \"the\", plain-English labels + its reply"), ("capital_of_france_board.html", "Animated board (canvas version)")) if k in fs]
    if mb:
        groups.insert(0, {"group": "Math", "items": mb})
    if "exp_store_3d.html" in fs:
        groups.insert(0, {"group": "Exponent store experiment", "items": [item("exp_store_3d.html", "3D view + results: bitcoin.pdf learned as e^x / angles")]})
    ai_items = [{"key": "live:angle_ai", "title": "Dashboard + chat (live)", "size": 0, "mtime": time.time()}]
    if "angle_student_report.html" in fs: ai_items.append(item("angle_student_report.html", "Experiment: student learning from the 3B model (results)"))
    if "si_pdf_report.html" in fs: ai_items.append(item("si_pdf_report.html", "PDF learning report: time, speed, memory, quality, exponent cube"))
    groups.insert(0, {"group": "ZYGON", "items": ai_items})
    if "llm/viz.html" in fs:
        groups.append({"group": "Traces", "items": [item("llm/viz.html", "Every matrix: \"What's the capital of France?\"")]})
    return groups


# ------------------------------------------------------------------ ZYGON (exponent identities + angle wheels + hooks), taught by the 3B model
AI = {"obj": None}
AI_LOCK = threading.Lock()


def get_ai():
    with AI_LOCK:
        if AI["obj"] is None:
            from gguf import GGUF
            from tokenizer import Tokenizer
            ai = angle_ai.AngleAI(Tokenizer(GGUF().meta))
            seed = os.path.join(ROOT, "model", "exp_store", "teacher.jsonl")
            if not ai.paths and os.path.exists(seed):                      # first start: begin with the 3B model's recorded answers
                for ln in open(seed, encoding="utf-8"):
                    if ln.strip():
                        r = json.loads(ln); ai.learn_qa(r["prompt"], r["reply_ids"], "3B model (seed)", ended=r["finished"])
            AI["obj"] = ai
        return AI["obj"]


def teacher_answer(q, max_new=80):
    m = ST["model"]
    ids = m.tok.encode(chatml([("system", ST["system"] or SYSTEM), ("user", q)]))
    cache = m.new_cache(len(ids) + max_new + 8); lg = m.forward(ids, cache); out, fin = [], False
    end = m.tok.id_of["<|im_end|>"]
    for _ in range(max_new):
        t = int(lg.argmax())
        if t in (end, m.tok.eos): fin = True; break
        out.append(t); lg = m.forward([t], cache)
    return m.tok.decode(out), fin


def angle_ask(q):
    ai = get_ai(); t0 = time.time(); r = ai.ask(q)
    if not r.get("needs_teacher"):
        return r
    if ST["model"] is None:
        ai.stats["none"] += 1; ai.series.append("none"); ai._agg("ask:none", (time.time() - t0) * 1000)
        return {"source": "none", "reply": "", "draft": r.get("draft", ""), "reason": r.get("reason", ""), "message": "This is new to the AI and the 3B model is not loaded, so nothing was learned. Press Load Frost 1.0 in the Frost sidebar, then ask again."}
    if not LOCK.acquire(timeout=600):
        return {"source": "none", "reply": "", "message": "The 3B model is busy with another answer."}
    try:
        ST["busy"] = True
        text, fin = teacher_answer(q)
    finally:
        ST["busy"] = False; LOCK.release()
    out = ai.learned_from_teacher(q, text, t0, ended=fin); out["reason"] = r.get("reason", "")
    return out


def angle_teach(body):
    ai = get_ai(); text, label = body.get("text", ""), "pasted text"
    path = body.get("path", "").strip().strip('"')
    if path:
        if not os.path.isfile(path): return {"error": "file not found: " + path}
        label = os.path.basename(path)
        if path.lower().endswith(".pdf"):
            import pypdf
            text = "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(path).pages)
        else:
            text = open(path, encoding="utf-8", errors="replace").read()
    text = text[:3_000_000]
    if not text.strip(): return {"error": "nothing to learn"}
    return {"ok": True, "tokens": ai.learn_text(text, label), "label": label}


# ------------------------------------------------------------------ chat generation (streams NDJSON lines)
def build_ids(m, history):
    while True:
        msgs = ([("system", ST["system"])] if ST["system"] else []) + [(x["role"], x["content"]) for x in history]
        ids = m.tok.encode(chatml(msgs))
        if len(ids) <= MAX_CTX - 400 or len(history) <= 1:
            return ids, history
        history = history[2:] if len(history) > 2 else history[1:]       # drop the oldest exchange


def generate(chat, write, max_new=600):
    """returns (reply text, path info).  1) locked path for this ask -> replay from disk, no layer runs;
    2) otherwise run the model, following any similar stored path as a draft (branch + rejoin); the result is stored as a candidate path."""
    m = ST["model"]
    history = chat["messages"]
    ask = history[-1]["content"]; prev = history[-2]["content"] if len(history) >= 2 and history[-2]["role"] == "assistant" else ""
    key, ch = PL.ask_key(ask), PL.ctx_hash(prev)
    t00 = time.time()
    hit = LK.find_exact(key, ch)
    if hit:
        r = LK.rec[hit]; text = r["reply"]
        for i in range(0, len(text), 60):
            write({"t": "tok", "s": text[i:i + 60]})
        LK.touch(hit, replayed=r["n_tokens"])
        ms = (time.time() - t00) * 1000
        write({"t": "path", "id": hit, "mode": "locked", "status": "locked", "ms": round(ms, 1), "tokens": r["n_tokens"]})
        write({"t": "done", "tokens": r["n_tokens"], "tok_s": 0, "locked": True, "ms": round(ms, 1)})
        return text, {"id": hit, "mode": "locked"}
    ids, used = build_ids(m, history)
    if len(used) < len(history):
        write({"t": "note", "s": f"(older messages were left out to fit the {MAX_CTX}-token memory)"})
    vec = PL.ask_vector(m, ask)
    near = LK.near(key, vec, ch)
    follower = PL.PathFollower(LK.corpus([r for _, r in near]), K=12)
    cache = ST["cache"] or m.new_cache(MAX_CTX)
    ST["cache"] = cache
    cached = ST["cached"]
    common = 0
    for a, b in zip(cached, ids):
        if a != b: break
        common += 1
    common = min(common, len(ids) - 1)                          # always feed at least one token
    cache.len = common                                          # positions < common are identical: reuse their keys/values
    t0 = time.time()
    for i in range(common, len(ids), 512):
        logits = m.forward(ids[i:i + 512], cache)
    write({"t": "info", "prompt_tokens": len(ids), "reused": common, "prefill_s": round(time.time() - t0, 1), "similar_paths": len(near)})
    out_ids, flags, dec, text = [], [], [], ""
    t1 = time.time()
    for t, on in PL.follow_generate(m, logits, cache, follower, max_new, SK, lambda: ST["stop"]):
        out_ids.append(t); flags.append(on); dec.append(t)
        s = m.tok.decode(dec)
        if not s.endswith(chr(0xFFFD)):
            dec = []; text += s
            write({"t": "tok", "s": s})
    ST["cached"] = (ids + out_ids)[:cache.len]
    n, el = len(out_ids), max(time.time() - t1, 1e-6)
    br, i = [], 0
    while i < n:                                                 # fresh (branch) stretches = tokens that were not the next token of a followed path
        if not flags[i]:
            j = i
            while j < n and not flags[j]: j += 1
            br.append({"start": i, "end": j}); i = j
        else:
            i += 1
    on_path = int(sum(flags))
    mode = "branch" if near and on_path >= 0.3 * max(n, 1) else "new"
    stats = {"tokens": n, "on_path": on_path, "passes": follower.passes, "seconds": round(el, 1), "similar": [[round(sv, 2), r] for sv, r in near]}
    rid = LK.add(ask, ch, out_ids, text, vec, parent=near[0][1] if mode == "branch" else None, branches=br if mode == "branch" else [{"start": 0, "end": n}], stats=stats, prompt_ids=ids)
    LK.stats["branch_runs" if mode == "branch" else "new_runs"] += 1; LK.stats["tokens_drafted_ok"] += follower.accepted; LK.save_index()
    write({"t": "path", "id": rid, "mode": mode, "status": "candidate", "on_path": on_path, "tokens": n, "passes": follower.passes, "branches": len(br)})
    write({"t": "done", "tokens": n, "tok_s": round(n / el, 2), "passes": follower.passes, "on_path": on_path})
    return text, {"id": rid, "mode": mode}


def fingerprint_job(rid):
    with LOCK:                                                   # waits until Frost has finished any answer
        m = ST["model"]; r = LK.rec.get(rid)
        if m is None or r is None or r.get("fingerprint"): return
        ST["busy"] = True
        try:
            LK.fingerprint(m, LK.prompt(rid), rid)
        except Exception:
            import traceback; traceback.print_exc()
        finally:
            ST["busy"] = False


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def _file(self, path, ctype="text/html; charset=utf-8"):
        b = open(path, "rb").read()
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._file(os.path.join(HERE, "frost_ui.html"))
            elif u.path == "/api/state":
                self._json(state())
            elif u.path == "/api/chats":
                self._json(list_chats())
            elif u.path == "/api/chat":
                self._json(load_chat(q["id"][0]))
            elif u.path == "/api/reports":
                self._json(list_reports())
            elif u.path == "/api/locker/status":
                ids = [x for x in q.get("ids", [""])[0].split(",") if x]
                self._json({i: {"status": LK.rec[i]["status"], "hits": LK.rec[i]["hits"], "n": LK.rec[i]["n_tokens"], "fp": LK.rec[i]["fingerprint"]} for i in ids if i in LK.rec})
            elif u.path == "/angle":
                self._file(os.path.join(HERE, "angle_ui.html"))
            elif u.path == "/api/angle/state":
                self._json(get_ai().state())
            elif u.path == "/cube_view.js":
                self._file(os.path.join(HERE, "cube_view.js"), "application/javascript; charset=utf-8")
            elif u.path == "/api/angle/cube":
                self._json(get_ai().cube())
            elif u.path == "/api/angle/wheel":
                self._json(get_ai().wheel_view(q.get("text", [""])[0]))
            elif u.path == "/api/angle/ident":
                self._json(get_ai().identities(q.get("text", [""])[0]))
            elif u.path == "/report":
                key = unquote(q.get("k", [""])[0]); fs = report_files()
                if key in ("live:path_locker", "live:code_paths"):
                    b = (locker_report.paths_page(LK) if key == "live:path_locker" else locker_report.code_page(LK)).encode("utf-8")
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
                    self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
                elif key == "live:angle_ai": self._file(os.path.join(HERE, "angle_ui.html"))
                elif key in fs: self._file(fs[key])
                else: self._json({"error": "not found"}, 404)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": repr(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path).path
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            if u == "/api/chats":
                return self._json(new_chat())
            if u == "/api/chats/delete":
                p = chat_path(body["id"])
                if os.path.exists(p): os.remove(p)
                return self._json({"ok": True})
            if u == "/api/model/load":
                if ST["model"] is None and not ST["loading"]:
                    ST["error"] = ""
                    threading.Thread(target=load_model, args=(bool(body.get("patch", True)),), daemon=True).start()
                return self._json({"ok": True})
            if u == "/api/model/unload":
                if ST["busy"]: return self._json({"error": "busy"}, 409)
                unload_model(); return self._json({"ok": True})
            if u == "/api/system":
                ST["system"] = body.get("text", "").strip(); return self._json({"ok": True, "system": ST["system"]})
            if u == "/api/locker/approve":
                ok = LK.approve(body["id"])
                if ok and ST["model"] is not None: threading.Thread(target=fingerprint_job, args=(body["id"],), daemon=True).start()
                return self._json({"ok": ok})
            if u == "/api/locker/unlock":
                LK.unlock(body["id"]); return self._json({"ok": True})
            if u == "/api/locker/delete":
                LK.delete(body["id"]); return self._json({"ok": True})
            if u == "/api/angle/ask":
                return self._json(angle_ask(body["q"].strip()))
            if u == "/api/angle/reject":
                get_ai().reject(int(body["id"])); return self._json({"ok": True})
            if u == "/api/angle/teach":
                return self._json(angle_teach(body))
            if u == "/api/angle/settings":
                ai = get_ai()
                if "memory_threshold_mb" in body: ai.settings["memory_threshold_mb"] = max(0.01, float(body["memory_threshold_mb"]))
                if "confidence" in body: ai.settings["confidence"] = min(1.0, max(0.0, float(body["confidence"])))
                ai.settings["alert_active"] = False; ai.check_memory(); return self._json({"ok": True, "settings": ai.settings})
            if u == "/api/stop":
                ST["stop"] = True; return self._json({"ok": True})
            if u == "/api/send":
                if ST["model"] is None:
                    return self._json({"error": "Model is not loaded. Press 'Load Frost 1.0' in the sidebar."}, 409)
                if not LOCK.acquire(blocking=False):
                    return self._json({"error": "Frost is busy (answering another message or tracing a locked path)."}, 409)
                try:
                    ST["busy"], ST["stop"] = True, False
                    chat = load_chat(body["chat_id"])
                    chat["messages"].append({"role": "user", "content": body["text"]})
                    if chat["title"] == "New chat":
                        chat["title"] = body["text"].strip().replace("\n", " ")[:42]
                    save_chat(chat)
                    self.send_response(200); self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                    self.send_header("Cache-Control", "no-store"); self.send_header("Connection", "close"); self.end_headers()

                    def write(o):
                        self.wfile.write((json.dumps(o, ensure_ascii=False) + "\n").encode("utf-8")); self.wfile.flush()
                    try:
                        reply, pinfo = generate(chat, write, int(body.get('max_new', 600)))
                        chat["messages"].append({"role": "assistant", "content": reply, "path": pinfo}); save_chat(chat)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        try: write({"t": "error", "s": repr(e)})
                        except Exception: pass
                finally:
                    ST["busy"] = False; LOCK.release()
                return
            self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": repr(e)}, 500)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"READY  http://127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()
