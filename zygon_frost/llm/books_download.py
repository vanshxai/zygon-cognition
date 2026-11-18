"""Downloads the 500 catalogued books (plain text, Project Gutenberg's own cache) with curl. 6 at a time. Writes model/books/text/<id>.txt and download.json"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..")); D = os.path.join(ROOT, "model", "books", "text"); os.makedirs(D, exist_ok=True)
books = json.load(open(os.path.join(ROOT, "model", "books", "catalogue_science.json"), encoding="utf-8"))["books"]
def get(b):
    f = os.path.join(D, f"{b['id']}.txt")
    if os.path.exists(f) and os.path.getsize(f) > 1000: return b["id"], os.path.getsize(f), None
    for _ in range(3):
        r = subprocess.run(["curl", "-sSL", "-m", "90", "-A", "FrostLang/1.0 (personal research)", "-o", f, "-w", "%{http_code}", f"https://www.gutenberg.org/cache/epub/{b['id']}/pg{b['id']}.txt"], capture_output=True, text=True)
        if r.stdout.strip() == "200" and os.path.getsize(f) > 1000: return b["id"], os.path.getsize(f), None
        time.sleep(2)
    if os.path.exists(f): os.remove(f)
    return b["id"], 0, r.stdout.strip() or r.stderr[-80:]
t0 = time.time(); res = []
with ThreadPoolExecutor(6) as ex:
    for i, r in enumerate(ex.map(get, books), 1):
        res.append(r)
        if i % 25 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
ok = [r for r in res if r[1]]; bad = [r for r in res if not r[1]]
json.dump({"seconds": time.time() - t0, "ok": len(ok), "failed": bad, "bytes": sum(r[1] for r in ok)}, open(os.path.join(ROOT, "model", "books", "download.json"), "w"))
print("done", len(ok), "ok", len(bad), "failed", f"{time.time() - t0:.0f}s", sum(r[1] for r in ok), "bytes")
