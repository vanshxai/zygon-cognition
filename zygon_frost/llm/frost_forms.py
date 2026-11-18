"""Form apps for Frost Math and Frost Engg.   python frost_forms.py --kind math  (port 8860)   |   --kind engg  (port 8870)
Plain forms: the person fills boxes, the page shows the answer and the full working. Text is only used to look things up."""
import argparse, json, os, sys, tempfile, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--kind", default="math"); ap.add_argument("--port", type=int, default=0); A = ap.parse_args()
PORT = A.port or (8860 if A.kind == "math" else 8870)
if A.kind == "math":
    import frostmath; E = frostmath.Math()
    if not E.F: E.learn()
else:
    import frostengg; E = frostengg.Engg()


def api(path, q, body):
    g = lambda k, d=None: (q.get(k) or [d])[0]
    if A.kind == "math":
        if path == "/api/status": return {**E.meta, **E.link_status()}
        if path == "/api/domains": return E.domains()
        if path == "/api/titles": return {n: E.card(n)["title"] for n in E.F}
        if path == "/api/vars": return sorted(E.by_var)
        if path == "/api/find": return E.find(g("q", ""), 12)
        if path == "/api/formula":
            c = E.card(g("name")); return {**c, "constants": E.suggest_constants(c["vars"]), "units": E.var_units(c["vars"])}
        if path == "/api/solve_one": return E.solve_one(body["name"], body.get("values", {}), body["target"], body.get("choice"))
        if path == "/api/solve_chain": return E.solve_chain(body.get("values", {}), body["target"], body.get("force"), body.get("choices"))
    else:
        if path == "/api/status": return {**E.link_status(), "structures": len(E.S), "examples": frostengg.EXAMPLES, "materials": frostengg.MATERIALS}
        if path == "/api/list": return E.listing()
        if path == "/api/get":
            r = E.S[int(g("id"))]; return {k: r[k] for k in ("id", "name", "source", "st", "lines", "bbox", "truth", "learned_at")}
        if path == "/api/measure": return E.measure(int(g("id")), float(g("scale", 1)), g("unit", "m"), float(g("density")) if g("density") else None)
        if path == "/api/similar": return E.similar(int(g("id")))
        if path == "/api/find": return E.find(g("q", ""))
        if path == "/api/forget": E.forget(int(g("id"))); return {"ok": True}
        if path == "/api/example":
            kind = g("kind"); t0 = time.perf_counter(); P, truth = frostengg.example(kind); r = E.learn_cloud(kind, P, "built-in example (known sizes)", truth); sc = frostengg.score(r["st"], truth)
            E.log("example", {"name": kind, "points": len(P), "seconds": round(time.perf_counter() - t0, 2), **{k: v for k, v in sc.items() if k != "errors"}}); return {"id": r["id"], "score": sc}
        if path == "/api/upload":
            name, fn, raw = g("name", "structure"), g("file", ""), body
            if fn.lower().endswith(".json"): r = E.learn_files(name, raw.decode("utf-8", "replace"), None, "uploaded structure file")
            else:
                import numpy as np
                with tempfile.NamedTemporaryFile("wb", suffix=".ply", delete=False) as f: f.write(raw); tmp = f.name
                try: P = frostengg.ST.load_ply(tmp)
                finally: os.unlink(tmp)
                r = E.learn_cloud(name, P, "uploaded point cloud " + fn)
            E.log("upload", {"name": name, "file": fn, "parts": len(r["st"]["primitives"]), "seconds": r.get("seconds")}); return {"id": r["id"]}
    raise KeyError(path)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, b, ct):
        self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def handle_(self, body):
        u = urlparse(self.path)
        if not u.path.startswith("/api/"): return self.send(200, open(os.path.join(HERE, A.kind + "_app.html"), "rb").read(), "text/html; charset=utf-8")
        try:
            raw = body
            if body and self.headers.get("Content-Type", "").startswith("application/json"): raw = json.loads(body)
            out = api(u.path, parse_qs(u.query), raw)
            self.send(200, json.dumps(out, default=str).encode(), "application/json")
        except Exception as e: self.send(200, json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

    def do_GET(self): self.handle_(b"")
    def do_POST(self): self.handle_(self.rfile.read(int(self.headers.get("Content-Length", 0))))


if __name__ == "__main__":
    print("READY  Frost", A.kind, f"http://127.0.0.1:{PORT}", flush=True); ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
