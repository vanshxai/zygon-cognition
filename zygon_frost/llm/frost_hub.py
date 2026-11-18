"""Frost dashboard: every Frost, old and new, in one window.   python llm/frost_hub.py  ->  http://127.0.0.1:8800
One address. The live apps start on first click as hidden background processes (no extra terminals) and stop when this one stops.
Reports and papers are served from here too. Nothing is copied: each entry points at the real file or program."""
import atexit, json, os, socket, subprocess, sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = sys.executable; SI = os.path.join(HERE, "si_app.py"); FORMS = os.path.join(HERE, "frost_forms.py")
DOC = "https://claude.ai/artifact/H2SW2sGv829WnPjSWig9yP"


def si(engine, port): return [PY, SI, "--engine", str(engine), "--port", str(port)]


# group, id, name, kind (app | page | link), what it needs, one line shown on hover
CATALOG = [
    ("Frost", "door", "ZYGON chat", "app", {"port": 8890, "cmd": [PY, os.path.join(HERE, "zygon_door.py")]}, "One screen: invoices, warehouse layout, HVAC sizing and questions."),
    ("Frost", "lang", "Frost Lang", "app", {"port": 8840, "cmd": si(3, 8840)}, "Reads documents and books, answers in their own sentences, keeps the chat thread."),
    ("Frost", "code", "Frost Code", "app", {"port": 8850, "cmd": si(4, 8850)}, "Coding lessons and full-stack web apps you can download."),
    ("Frost", "math", "Frost Math", "app", {"port": 8860, "cmd": [PY, FORMS, "--kind", "math", "--port", "8860"]}, "333 formulas, exact answers, every step shown."),
    ("Frost", "engg", "Frost Engg", "app", {"port": 8870, "cmd": [PY, FORMS, "--kind", "engg", "--port", "8870"]}, "Parts and sizes of structures from photos or video."),
    ("Tools", "round", "Round table", "app", {"port": 8880, "cmd": [PY, os.path.join(os.path.expanduser("~"), "Desktop", "AI chat", "chat_view.py"), "8880"]}, "Short chat between the CEO, ZYGON and FaulterSI."),
    ("Earlier experiments", "si2", "ZYGON 2.0", "app", {"port": 8830, "cmd": si(2, 8830)}, "Letters on one long spiral thread."),
    ("Earlier experiments", "si1", "ZYGON 1.0", "app", {"port": 8820, "cmd": si(1, 8820)}, "Word identities on angle wheels."),
    ("Earlier experiments", "f1", "Frost 1.0 chat", "app", {"port": 8810, "cmd": [PY, os.path.join(HERE, "frost_app.py")]}, "The 3B-model chat with saved chats and reports. Needs about 13 GB of memory once loaded."),
    ("Earlier experiments", "store3d", "Exponent store in 3D", "page", {"file": "exp_store_3d.html"}, "The first exponent memory, tested on one PDF."),
    ("Earlier experiments", "stud", "ZYGON student", "page", {"file": "angle_student_report.html"}, "Report of the SI learning from a teacher."),
    ("Earlier experiments", "pdf", "ZYGON learns a PDF", "page", {"file": "si_pdf_report.html"}, "Report of one PDF learned and tested."),
    ("Papers and reports", "paper", "Frost Lang: how it learns", "page", {"file": os.path.join("llm", "frost_paper.html")}, "One-page paper with the measured results."),
    ("Papers and reports", "doc", "SI research notes", "link", {"url": DOC}, "The running research document with every metric (opens in a new tab)."),
    ("Papers and reports", "layer1", "Layer 1 for 'the'", "page", {"file": "the_layer1_paper.html"}, "Frost 1.0 paper on one word's first layer."),
    ("Papers and reports", "byhand", "Worked by hand", "page", {"file": "capital_of_france_paper.html"}, "Every number of one answer, written out."),
    ("Papers and reports", "board", "Math board", "page", {"file": "capital_of_france_board.html"}, "Animated board of one prompt through the model."),
    ("Papers and reports", "caps", "Frost 1.0 capabilities", "page", {"file": "capabilities_report.html"}, "What Frost 1.0 could do."),
    ("Papers and reports", "neurons", "What the other neurons do", "page", {"file": "neuron_capabilities_report.html"}, "Neuron capabilities report."),
    ("Papers and reports", "wscan", "Text-reply weight scan", "page", {"file": "text_weights_scan_report.html"}, "Which weights carry text replies."),
    ("3D maps", "exploded", "3B model, exploded", "page", {"file": os.path.join("llm", "model_exploded.html")}, "One block of the 3B model pulled apart, drawn from its real weights."),
    ("3D maps", "sizemap", "Neurons vs prompt size", "page", {"file": "neuron_size_map.html"}, "3D map (large file, loads slowly)."),
    ("3D maps", "swmap", "Coding neurons in 10 programs", "page", {"file": "software_neuron_map.html"}, "3D map of coding neurons."),
    ("3D maps", "swindex", "The 10 programs", "page", {"file": "software_index.html"}, "Index of the ten programs."),
    ("3D maps", "qwenrun", "Qwen 3B run report", "page", {"file": "ai_run_report.html"}, "Run report of the 3B model (large file)."),
    ("3D maps", "viz", "Inside Qwen, every matrix", "page", {"file": os.path.join("llm", "viz.html")}, "One question through every matrix (large file)."),
]
PROCS = {}
_G, _TOK, _LRU = None, None, []


def _tensor(layer, part):
    """one real weight matrix of the 3B model, widened to float32 (the last few are kept in memory)"""
    global _G, _TOK
    sys.dont_write_bytecode = True
    if HERE not in sys.path: sys.path.insert(0, HERE)
    import gguf
    if _G is None:
        _G = gguf.GGUF(os.path.join(ROOT, "model", "original", "qwen2.5-3b-instruct-q4_k_m.original.gguf"))
        from tokenizer import Tokenizer
        _TOK = Tokenizer(_G.meta)
    name = "token_embd.weight" if part == "embed" else f"blk.{int(layer)}.{part}.weight"
    if name not in _G.tensors: raise KeyError(name)
    for k, a in _LRU:
        if k == name: return a, _G.tensors[name].type_name
    a = gguf.np_tensor(_G, name); _LRU.append((name, a))
    while sum(x[1].nbytes for x in _LRU) > 1.8e9 and len(_LRU) > 1: _LRU.pop(0)
    return a, _G.tensors[name].type_name


def weights_api(q):
    """a rectangle of a matrix: the exact stored numbers if it is small, otherwise squeezed into at most oh x ow blocks (average size and average value of each block)"""
    import numpy as np
    g = lambda k, d=None: (q.get(k) or [d])[0]
    A, typ = _tensor(g("layer", 0), g("part")); R, C = A.shape
    r0, r1 = max(0, int(g("r0", 0))), min(R, int(g("r1", R))); c0, c1 = max(0, int(g("c0", 0))), min(C, int(g("c1", C))); oh, ow = int(g("oh", 130)), int(g("ow", 200))
    X = A[r0:r1, c0:c1]; rows, cols = X.shape; out = {"shape": [R, C], "type": typ, "r0": r0, "r1": r1, "c0": c0, "c1": c1, "std": float(A[::13, ::7].std())}
    if rows <= oh and cols <= ow:
        out.update(mode="exact", values=np.round(X.astype(np.float64), 6).tolist())
        if g("part") == "embed": out["row_labels"] = [_TOK.decode([i]) for i in range(r0, r1)]
    else:
        h, w = min(rows, oh), min(cols, ow); ri = np.linspace(0, rows, h + 1).astype(int); ci = np.linspace(0, cols, w + 1).astype(int); n = np.outer(np.diff(ri), np.diff(ci))
        ab = np.add.reduceat(np.add.reduceat(np.abs(X), ri[:-1], axis=0), ci[:-1], axis=1) / n; mn = np.add.reduceat(np.add.reduceat(X, ri[:-1], axis=0), ci[:-1], axis=1) / n
        if g("part") == "embed" and rows <= oh: out["row_labels"] = [_TOK.decode([i]) for i in range(r0, r1)]
        out.update(mode="pool", oh=h, ow=w, abs=np.round(ab.astype(np.float64), 6).tolist(), mean=np.round(mn.astype(np.float64), 6).tolist(), row_edges=(ri + r0).tolist(), col_edges=(ci + c0).tolist())
    return out


def up(port):
    with socket.socket() as s: s.settimeout(0.3); return s.connect_ex(("127.0.0.1", port)) == 0


def path_of(rec): return os.path.join(ROOT, rec["file"])


def start(rec):
    if up(rec["port"]): return
    PROCS[rec["port"]] = subprocess.Popen(rec["cmd"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def stop_all():
    for p in PROCS.values():
        try: p.terminate()
        except Exception: pass
    PROCS.clear()


atexit.register(stop_all)


def catalog():
    out = []
    for g, i, n, k, rec, note in CATALOG:
        if k == "page" and not os.path.exists(path_of(rec)): continue
        e = {"group": g, "id": i, "name": n, "kind": k, "note": note}
        if k == "app": e.update(port=rec["port"], up=up(rec["port"]))
        if k == "link": e["url"] = rec["url"]
        if k == "page": e["size_kb"] = os.path.getsize(path_of(rec)) // 1024
        out.append(e)
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, b, ct):
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        p, _, q = self.path.partition("?")
        if p == "/api/catalog": return self.send(json.dumps(catalog()).encode(), "application/json")
        if p == "/api/start":
            rec = next((r for g, i, n, k, r, _ in CATALOG if i == q.split("id=")[-1] and k == "app"), None)
            if rec: start(rec)
            return self.send(json.dumps({"up": bool(rec and up(rec["port"]))}).encode(), "application/json")
        if p == "/api/weights":
            from urllib.parse import parse_qs
            try: return self.send(json.dumps(weights_api(parse_qs(q))).encode(), "application/json")
            except Exception as e: return self.send(json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")
        if p in ("/api/piece", "/api/near"):
            from urllib.parse import parse_qs
            try:
                import numpy as np
                qq = parse_qs(q); A, _t = _tensor(0, "embed")
                if p == "/api/piece":
                    t = qq["q"][0]; ids = []
                    for v in (t, " " + t, t.capitalize(), " " + t.capitalize()):
                        for i in _TOK.encode(v):
                            if i not in ids: ids.append(i)
                    return self.send(json.dumps({"ids": ids[:12], "labels": [_TOK.decode([i]) for i in ids[:12]]}).encode(), "application/json")
                i = int(qq["id"][0]); k = int((qq.get("k") or [12])[0]); v = A[i]; sc = A @ v / (np.linalg.norm(A, axis=1) * np.linalg.norm(v) + 1e-9); o = np.argsort(-sc)[1:k + 1]
                return self.send(json.dumps({"id": i, "label": _TOK.decode([i]), "near": [[int(j), _TOK.decode([int(j)]), round(float(sc[j]), 3)] for j in o]}).encode(), "application/json")
            except Exception as e: return self.send(json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")
        if p == "/api/stopall": stop_all(); return self.send(b'{"ok":true}', "application/json")
        if p.startswith("/doc/"):
            rec = next((r for g, i, n, k, r, _ in CATALOG if i == p[5:] and k == "page"), None)
            if rec and os.path.exists(path_of(rec)): return self.send(open(path_of(rec), "rb").read(), "text/html; charset=utf-8")
            return self.send(b"not found", "text/plain")
        self.send(open(os.path.join(HERE, "hub.html"), "rb").read(), "text/html; charset=utf-8")


if __name__ == "__main__":
    print("READY  Frost dashboard  http://127.0.0.1:8800", flush=True)
    try: ThreadingHTTPServer(("127.0.0.1", 8800), H).serve_forever()
    finally: stop_all()
