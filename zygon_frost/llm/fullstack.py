"""Frost Code full-stack builder.
 parse_spec(text)  -> what to build, read from a plain description (title, record name, fields and their kinds, colour, layout, features)
 make_files(spec)  -> {path: text}: server.py (REST backend, JSON file store), index.html (UI), README.md, run.bat, run.sh, Dockerfile, tests/test_api.py
 build_zip(spec)   -> writes the project folder and a .zip next to it (limit 10 MB)
The pieces are written in advance (backend routes, field widgets, palettes, layouts); the description chooses and fills them. Python standard library only."""
import json, os, re, zipfile

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
PROJECTS = os.path.join(ROOT, "model", "code_projects")
LIMIT = 10 * 1024 * 1024
COLORS = {"red": "#c0392b", "blue": "#2563eb", "green": "#2e8b57", "purple": "#7c3aed", "orange": "#e67e22", "pink": "#db2777", "teal": "#0a8f9d", "yellow": "#c99a06", "brown": "#8b5e3c", "grey": "#5f6b73", "gray": "#5f6b73", "black": "#222831", "dark": "#222831"}
FONTS = {"modern": "system-ui,'Segoe UI',Roboto,sans-serif", "classic": "Georgia,'Times New Roman',serif", "mono": "Consolas,'Courier New',monospace"}
KINDS = [  # (kind, words in a field name)
    ("number", r"price|cost|amount|total|qty|quantity|count|age|score|salary|budget|weight|height|stock|number|rating|rate|year|hours|minutes|points|sets|reps|miles|km|calories|distance"),
    ("date", r"date|deadline|due|birthday|day|when|start|end"),
    ("email", r"e-?mail"), ("url", r"url|link|website|site"), ("tel", r"phone|mobile|tel"),
    ("check", r"done|completed|finished|paid|active|available|checked|published|read"),
    ("long", r"notes?|description|details|comments?|message|bio|summary|body|content|address"),
    ("color", r"colou?r"),
]
FEATURE_WORDS = r"search|sort|login|log in|sign in|password|dark mode|dark theme|theme|charts?|export|download|csv|stats|statistics|dashboard|table|cards?|grid|colou?rs?|responsive|mobile|nice|simple|modern|classic"


def kind_of(name):
    for k, rx in KINDS:
        if re.search(rx, name.lower()): return k
    return "text"


def singular(w):
    w = w.lower()
    if w.endswith("ies"): return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes"): return w[:-2]
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def parse_spec(text):
    low = text.lower(); spec = {"request": text}
    m = re.search(r"(?:called|named|titled)\s+[\"']?([A-Za-z0-9 ,'!&-]{2,40}?)[\"']?(?:\s+(?:with|in|using|that|and|where)\b|[.!?]?$)", text, re.I); title = m.group(1).strip() if m else None
    sub = re.search(r"\b(?:for|to (?:track|manage|keep|store|list)|of|that (?:tracks|manages|stores|lists))\s+(?:my |our |the |a |an )?([a-z][a-z ]{2,30}?)(?=\s+(?:with|that|in|and|where|using|having)\b|[.,!?]|$)", low)
    noun = None
    if sub: noun = sub.group(1).split()[-1]
    else:
        m2 = re.search(r"(?:a|an)\s+(?:[a-z-]+\s+)*?([a-z]+)\s+(?:web ?app|app|application|website|web ?site|tracker|manager|catalog|catalogue|portal|dashboard)\b", low)
        if m2 and m2.group(1) not in ("web", "full", "stack", "fullstack", "simple", "small", "new", "nice", "modern"): noun = m2.group(1)
    noun = singular(noun) if noun else "entry"
    fields = []
    for m3 in re.finditer(r"\b(?:with|has|have|having|fields?|columns?|including|tracks?|stores?|containing)\s+(?:a |an |the |these |fields?:? |columns?:? )?([a-z0-9][a-z0-9 ,/&-]{1,120}?)(?=\s+(?:in|using|on|where|so|plus|for)\b|[.!?;]|$)", low):
        part = m3.group(1)
        if re.match(r"(?:" + FEATURE_WORDS + r")\b", part) and "," not in part: continue
        for tok in re.split(r",|\band\b|&|/", part):
            tok = re.sub(r"^(?:a|an|the|its|their|each|some)\s+", "", tok.strip()); tok = re.sub(r"\s+(?:field|column)s?$", "", tok)
            if not tok or len(tok.split()) > 3 or re.fullmatch(r"(?:" + FEATURE_WORDS + r")(?: \w+)?", tok) or re.search(r"\b(?:app|website|site|page|login|search)\b", tok): continue
            if tok not in fields: fields.append(tok)
        if fields: break
    if not fields: fields = ["name", "notes"]
    fields = fields[:8]
    spec.update(noun=noun, fields=[{"label": f.title(), "key": re.sub(r"[^a-z0-9]+", "_", f).strip("_"), "kind": kind_of(f)} for f in fields])
    spec["title"] = title or (noun.title() + " manager" if noun != "entry" else "My records")
    spec["color"] = next((COLORS[c] for c in COLORS if re.search(r"\b" + c + r"\b", low)), "#0a6c80")
    spec["dark"] = bool(re.search(r"dark (?:mode|theme)|\bblack\b|\bdark\b", low))
    if spec["dark"] and spec["color"] == "#222831": spec["color"] = "#2f9fb3"
    spec["layout"] = "table" if re.search(r"\btable|spreadsheet|rows\b", low) else "cards"
    spec["font"] = "classic" if re.search(r"classic|serif|elegant", low) else "mono" if re.search(r"mono|hacker|terminal|code look", low) else "modern"
    spec["export"] = bool(re.search(r"export|download|csv|spreadsheet", low)); spec["login"] = bool(re.search(r"log ?in|sign ?in|password|private|protected", low))
    spec["stats"] = any(f["kind"] == "number" for f in spec["fields"]) or bool(re.search(r"stats|statistics|summary|dashboard|total", low))
    spec["slug"] = re.sub(r"[^a-z0-9]+", "-", spec["title"].lower()).strip("-") or "app"
    return spec


SERVER = r'''"""@TITLE@ - backend. Run: python server.py [port]   (standard library only)
Routes: GET /api/items?q=&sort=   POST /api/items   PUT /api/items/<id>   DELETE /api/items/<id>   GET /api/stats   GET /api/export.csv   POST /api/login"""
import csv, hashlib, io, json, os, secrets, sys, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data.json")
FIELDS = @FIELDS@
PASSWORD = os.environ.get("APP_PASSWORD", @PASSWORD@)       # empty = no login
LOCK = threading.Lock(); TOKENS = set()
items = json.load(open(DATA, encoding="utf-8")) if os.path.exists(DATA) else []


def save():
    with open(DATA, "w", encoding="utf-8") as f: json.dump(items, f)


def clean(body):
    out = {}
    for f in FIELDS:
        v = body.get(f["key"], "")
        if f["kind"] == "number":
            try: v = float(v) if str(v).strip() != "" else None
            except ValueError: raise ValueError(f["label"] + " must be a number")
        elif f["kind"] == "check": v = bool(v)
        else: v = str(v).strip()[:2000]
        out[f["key"]] = v
    if not any(out[f["key"]] not in ("", None, False) for f in FIELDS): raise ValueError("Fill in at least one field")
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="application/json", extra=None):
        b = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def authed(self):
        return not PASSWORD or self.headers.get("X-Token") in TOKENS

    def body(self):
        n = int(self.headers.get("Content-Length") or 0); return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path in ("/", "/index.html"): return self.send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html")
        if not u.path.startswith("/api/"): return self.send(404, {"error": "not found"})
        if u.path == "/api/config": return self.send(200, {"login": bool(PASSWORD)})
        if not self.authed(): return self.send(401, {"error": "login needed"})
        if u.path == "/api/items":
            res = list(items); s = (q.get("q", [""])[0]).lower()
            if s: res = [i for i in res if s in " ".join(str(i.get(f["key"], "")) for f in FIELDS).lower()]
            k = q.get("sort", [""])[0]
            if k: res.sort(key=lambda i: (i.get(k) is None, str(i.get(k, "")).lower() if not isinstance(i.get(k), (int, float)) else i.get(k)))
            return self.send(200, res)
        if u.path == "/api/stats":
            st = {"count": len(items)}
            for f in FIELDS:
                if f["kind"] == "number":
                    v = [i[f["key"]] for i in items if isinstance(i.get(f["key"]), (int, float))]
                    st[f["key"]] = {"sum": sum(v), "avg": (sum(v) / len(v)) if v else 0, "min": min(v) if v else 0, "max": max(v) if v else 0}
            return self.send(200, st)
        if u.path == "/api/export.csv":
            out = io.StringIO(); w = csv.writer(out); w.writerow([f["label"] for f in FIELDS])
            for i in items: w.writerow([i.get(f["key"], "") for f in FIELDS])
            return self.send(200, out.getvalue(), "text/csv", {"Content-Disposition": 'attachment; filename="export.csv"'})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path).path
        try:
            b = self.body()
            if u == "/api/login":
                if PASSWORD and secrets.compare_digest(str(b.get("password", "")), PASSWORD): t = secrets.token_hex(16); TOKENS.add(t); return self.send(200, {"token": t})
                return self.send(401, {"error": "wrong password"})
            if not self.authed(): return self.send(401, {"error": "login needed"})
            if u == "/api/items":
                with LOCK:
                    it = clean(b); it["id"] = (max([i["id"] for i in items]) + 1) if items else 1; items.append(it); save()
                return self.send(201, it)
        except ValueError as e: return self.send(400, {"error": str(e)})
        self.send(404, {"error": "not found"})

    def do_PUT(self):
        m = re.match(r"/api/items/(\d+)$", urlparse(self.path).path) if False else None
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) != 3 or parts[:2] != ["api", "items"] or not parts[2].isdigit(): return self.send(404, {"error": "not found"})
        if not self.authed(): return self.send(401, {"error": "login needed"})
        try:
            b = self.body()
            with LOCK:
                for i in items:
                    if i["id"] == int(parts[2]): i.update(clean(b)); save(); return self.send(200, i)
        except ValueError as e: return self.send(400, {"error": str(e)})
        self.send(404, {"error": "no such item"})

    def do_DELETE(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) != 3 or parts[:2] != ["api", "items"] or not parts[2].isdigit(): return self.send(404, {"error": "not found"})
        if not self.authed(): return self.send(401, {"error": "login needed"})
        with LOCK:
            n = len(items); items[:] = [i for i in items if i["id"] != int(parts[2])]; save()
        self.send(200 if len(items) < n else 404, {"ok": len(items) < n})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"@TITLE@ running at http://127.0.0.1:{port}/", flush=True); ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''

PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@TITLE@</title>
<style>
:root{--acc:@COLOR@;--bg:@BG@;--card:@CARD@;--ink:@INK@;--mute:@MUTE@;--line:@LINE@;font-family:@FONT@}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);line-height:1.5}
header{display:flex;align-items:center;gap:12px;padding:14px 22px;background:var(--acc);color:#fff}header h1{font-size:20px;margin:0;flex:1}
main{max-width:920px;margin:0 auto;padding:18px 16px 60px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 14px}.stat b{display:block;font-size:20px}.stat span{color:var(--mute);font-size:12px}
form,.tools{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin-bottom:14px;display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
label{font-size:12px;color:var(--mute);display:block}input,textarea,select{width:100%;font:inherit;padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}textarea{min-height:60px}
input[type=checkbox]{width:auto}button{font:inherit;border:1px solid var(--acc);background:var(--acc);color:#fff;border-radius:8px;padding:7px 14px;cursor:pointer}button.alt{background:transparent;color:var(--acc)}button.x{background:transparent;color:#b3442f;border-color:transparent;padding:2px 8px}
.tools{grid-template-columns:2fr 1fr auto}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}.card .k{color:var(--mute);font-size:12px}.card .row{display:flex;gap:6px;margin-top:8px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
#msg{color:#b3442f;min-height:20px;font-size:13px}.empty{color:var(--mute);text-align:center;padding:30px}
#login{max-width:340px;margin:12vh auto;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;display:none}
@media(max-width:560px){.tools{grid-template-columns:1fr}}
</style></head><body>
<header><h1>@TITLE@</h1>@EXPORTBTN@</header>
<div id="login"><h3 style="margin-top:0">Please log in</h3><input id="pw" type="password" placeholder="Password"><p><button id="go">Log in</button></p><div id="lmsg" style="color:#b3442f"></div></div>
<main id="app" style="display:none">
<div class="stats" id="stats"></div>
<form id="f">@INPUTS@<div style="align-self:end"><button id="save">Add</button> <button type="button" class="alt" id="cancel" style="display:none">Cancel</button></div></form><div id="msg"></div>
<div class="tools"><input id="q" placeholder="Search"><select id="sort"><option value="">Newest last</option>@SORTS@</select><span></span></div>
<div id="list"></div></main>
<script>
const FIELDS = @FIELDS@, LAYOUT = '@LAYOUT@', TITLE = @TITLEJS@;
let token = '', editing = null; const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
async function api(path, method = 'GET', body) {
  const r = await fetch(path, {method, headers: {'Content-Type': 'application/json', 'X-Token': token}, body: body ? JSON.stringify(body) : undefined});
  if (r.status === 401) { showLogin(); throw new Error('login'); } const j = await r.json(); if (!r.ok) throw new Error(j.error || 'Something went wrong'); return j;
}
function showLogin() { $('#app').style.display = 'none'; $('#login').style.display = 'block'; }
function show(v) { if (v === true) return 'Yes'; if (v === false) return ''; return esc(v); }
function row(i) {
  const act = `<button class="alt" onclick="edit(${i.id})">Edit</button><button class="x" onclick="del(${i.id})">Delete</button>`;
  if (LAYOUT === 'table') return `<tr>${FIELDS.map(f => `<td>${show(i[f.key])}</td>`).join('')}<td style="white-space:nowrap">${act}</td></tr>`;
  return `<div class="card">${FIELDS.map((f, n) => n === 0 ? `<b>${show(i[f.key])}</b>` : `<div><span class="k">${esc(f.label)}</span> ${show(i[f.key])}</div>`).join('')}<div class="row">${act}</div></div>`;
}
let cache = [];
async function load() {
  cache = await api('/api/items?q=' + encodeURIComponent($('#q').value) + '&sort=' + $('#sort').value);
  $('#list').innerHTML = !cache.length ? '<div class="empty">Nothing here yet. Add the first one above.</div>' : LAYOUT === 'table' ? `<table><tr>${FIELDS.map(f => `<th>${esc(f.label)}</th>`).join('')}<th></th></tr>${cache.map(row).join('')}</table>` : `<div class="cards">${cache.map(row).join('')}</div>`;
  const s = await api('/api/stats'); let h = `<div class="stat"><b>${s.count}</b><span>total</span></div>`;
  for (const f of FIELDS) if (f.kind === 'number' && s[f.key]) h += `<div class="stat"><b>${+s[f.key].sum.toFixed(2)}</b><span>${esc(f.label)} total</span></div><div class="stat"><b>${+s[f.key].avg.toFixed(2)}</b><span>${esc(f.label)} average</span></div>`;
  $('#stats').innerHTML = h;
}
function val(f) { const e = $('#fld_' + f.key); return f.kind === 'check' ? e.checked : e.value; }
$('#f').onsubmit = async e => {
  e.preventDefault(); $('#msg').textContent = ''; const b = {}; FIELDS.forEach(f => b[f.key] = val(f));
  try { editing ? await api('/api/items/' + editing, 'PUT', b) : await api('/api/items', 'POST', b); reset(); await load(); } catch (x) { if (x.message !== 'login') $('#msg').textContent = x.message; }
};
function reset() { editing = null; $('#f').reset(); $('#save').textContent = 'Add'; $('#cancel').style.display = 'none'; }
window.edit = id => { const i = cache.find(x => x.id === id); editing = id; FIELDS.forEach(f => { const e = $('#fld_' + f.key); f.kind === 'check' ? e.checked = !!i[f.key] : e.value = i[f.key] ?? ''; }); $('#save').textContent = 'Save'; $('#cancel').style.display = ''; scrollTo(0, 0); };
window.del = async id => { if (confirm('Delete this?')) { await api('/api/items/' + id, 'DELETE'); load(); } };
$('#cancel').onclick = reset; $('#q').oninput = load; $('#sort').onchange = load;
$('#go').onclick = async () => { try { const r = await fetch('/api/login', {method: 'POST', body: JSON.stringify({password: $('#pw').value})}); const j = await r.json(); if (!r.ok) throw new Error(j.error); token = j.token; start(); } catch (x) { $('#lmsg').textContent = x.message; } };
async function start() { $('#login').style.display = 'none'; $('#app').style.display = 'block'; try { await load(); } catch (x) {} }
fetch('/api/config').then(r => r.json()).then(c => c.login ? showLogin() : start());
</script></body></html>'''

TEST = r'''"""Smoke test: starts the server on a free port, adds, edits, searches, exports and deletes one record.  Run: python -m unittest discover tests"""
import json, os, subprocess, sys, time, unittest, urllib.request, socket, tempfile, shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIRST = @FIRST@


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); cls.port = s.getsockname()[1]; s.close(); cls.tmp = tempfile.mkdtemp()
        for f in ("server.py", "index.html"): shutil.copy(os.path.join(ROOT, f), cls.tmp)
        env = dict(os.environ, APP_PASSWORD="")
        cls.p = subprocess.Popen([sys.executable, os.path.join(cls.tmp, "server.py"), str(cls.port)], env=env, stdout=subprocess.DEVNULL, cwd=cls.tmp); cls.base = f"http://127.0.0.1:{cls.port}"
        for _ in range(50):
            try: urllib.request.urlopen(cls.base + "/", timeout=1); break
            except Exception: time.sleep(0.1)

    @classmethod
    def tearDownClass(cls): cls.p.terminate(); shutil.rmtree(cls.tmp, ignore_errors=True)

    def call(self, path, method="GET", body=None):
        r = urllib.request.urlopen(urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None, method=method), timeout=5); return r.status, r.read().decode()

    def test_flow(self):
        code, b = self.call("/api/items", "POST", {FIRST: "hello"}); self.assertEqual(code, 201); i = json.loads(b)["id"]
        self.assertEqual(self.call("/api/items?q=hello")[0], 200); self.assertIn("hello", self.call("/api/items?q=hello")[1])
        self.assertEqual(self.call(f"/api/items/{i}", "PUT", {FIRST: "changed"})[0], 200); self.assertIn("changed", self.call("/api/items")[1])
        self.assertEqual(self.call("/api/export.csv")[0], 200); self.assertEqual(self.call("/api/stats")[0], 200)
        self.assertEqual(self.call(f"/api/items/{i}", "DELETE")[0], 200); self.assertEqual(json.loads(self.call("/api/items")[1]), [])

    def test_page(self): self.assertIn("<title>", self.call("/")[1])


if __name__ == "__main__": unittest.main()
'''


def make_files(spec):
    f = spec["fields"]; dark = spec["dark"]; pal = dict(BG="#14191c", CARD="#1d2428", INK="#e8eef0", MUTE="#8fa0a6", LINE="#2b353a") if dark else dict(BG="#f6f8f9", CARD="#ffffff", INK="#1c2b30", MUTE="#6b7d83", LINE="#dfe6e8")
    def inp(x):
        k, i = x["kind"], "fld_" + x["key"]
        if k == "long": w = f'<textarea id="{i}"></textarea>'
        elif k == "check": w = f'<input id="{i}" type="checkbox">'
        else: w = f'<input id="{i}" type="{ {"number": "number", "date": "date", "email": "email", "url": "url", "tel": "tel", "color": "color"}.get(k, "text") }"' + (' step="any"' if k == "number" else "") + ">"
        return f'<div><label for="{i}">{x["label"]}</label>{w}</div>'
    sorts = "".join(f'<option value="{x["key"]}">Sort by {x["label"]}</option>' for x in f)
    page = PAGE
    for k, v in {**pal, "@COLOR@": spec["color"], "@FONT@": FONTS[spec["font"]], "@TITLE@": spec["title"], "@INPUTS@": "".join(inp(x) for x in f), "@SORTS@": sorts, "@FIELDS@": json.dumps(f), "@LAYOUT@": spec["layout"], "@TITLEJS@": json.dumps(spec["title"]),
                 "@EXPORTBTN@": '<a href="/api/export.csv" style="color:#fff;font-size:14px">Download CSV</a>' if spec["export"] else ""}.items():
        page = page.replace(k if k.startswith("@") else "@" + k + "@", str(v))
    server = SERVER.replace("@TITLE@", spec["title"]).replace("@FIELDS@", json.dumps(f)).replace("@PASSWORD@", '"changeme"' if spec["login"] else '""')
    run_note = "Log in with the password `changeme` (change it with the APP_PASSWORD environment variable)." if spec["login"] else "No login."
    readme = f"""# {spec['title']}

A small full-stack web app made by Frost Code from this request:

> {spec['request']}

- Records: **{spec['noun']}** with {', '.join(x['label'] + ' (' + x['kind'] + ')' for x in f)}
- Backend: `server.py` (Python standard library, JSON file `data.json`)
- Frontend: `index.html` (one file, no build step, {spec['layout']} layout{', dark' if dark else ''})
- {run_note}

## Run
    python server.py 8000        (or double-click run.bat on Windows)
Then open http://127.0.0.1:8000/

## Test
    python -m unittest discover tests

## Deploy
`Dockerfile` is included: `docker build -t app . && docker run -p 8000:8000 app`. For a public host set `APP_PASSWORD` and put it behind HTTPS.
The server binds to 127.0.0.1; change the address in `server.py` (last line) to `0.0.0.0` when deploying in a container.
"""
    docker = 'FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\nEXPOSE 8000\nCMD ["python", "server.py", "8000"]\n'
    server_c = server.replace('ThreadingHTTPServer(("127.0.0.1", port), H)', 'ThreadingHTTPServer((os.environ.get("HOST", "127.0.0.1"), port), H)')
    docker = docker.replace("EXPOSE 8000", "ENV HOST=0.0.0.0\nEXPOSE 8000")
    return {"server.py": server_c, "index.html": page, "README.md": readme, "run.bat": "@echo off\r\npython server.py 8000\r\n", "run.sh": "#!/bin/sh\npython3 server.py 8000\n", "Dockerfile": docker, "tests/test_api.py": TEST.replace("@FIRST@", json.dumps(f[0]["key"]))}


def build_zip(spec):
    files = make_files(spec); os.makedirs(PROJECTS, exist_ok=True); slug = spec["slug"]; d = os.path.join(PROJECTS, slug); n = 1
    while os.path.exists(d): n += 1; d = os.path.join(PROJECTS, f"{slug}-{n}")
    for name, text in files.items():
        p = os.path.join(d, name); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w", encoding="utf-8", newline="").write(text)
    total = sum(len(t.encode()) for t in files.values())
    if total > LIMIT: return {"ok": False, "why": "over the 10 MB limit"}
    z = d + ".zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in files: zf.write(os.path.join(d, name), os.path.basename(d) + "/" + name)
    return {"ok": True, "name": os.path.basename(d), "dir": d, "zip": z, "zip_bytes": os.path.getsize(z), "bytes": total, "files": {k: len(v.encode()) for k, v in files.items()}}
