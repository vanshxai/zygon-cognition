"""Frost Code tools: recipes for small web apps and the functions that build them.
 build(recipe, title, color) -> creates a folder, writes the files, checks the size limit (10 MB), starts the server, checks that it answers
 stop(name | None) -> stops one app or all of them
Everything uses only the Python standard library, so a built app is a few KB."""
import json, os, re, socket, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
PROJECTS = os.path.join(ROOT, "model", "code_projects")
LIMIT = 10 * 1024 * 1024
RUNNING = {}
COLORS = {"red": "#c0392b", "blue": "#2563eb", "green": "#2e8b57", "purple": "#7c3aed", "orange": "#e67e22", "pink": "#db2777", "teal": "#0a8f9d", "dark": "#222831", "black": "#222831", "yellow": "#d4a017", "brown": "#8d5524", "gray": "#6b7280", "grey": "#6b7280"}

CRUD_SERVER = r'''import json, os, sys, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data.json")
FIELDS = @FIELDS@
items = json.load(open(DATA, encoding="utf-8")) if os.path.exists(DATA) else []


def save():
    json.dump(items, open(DATA, "w", encoding="utf-8"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/items":
            return self.send_json(items)
        page = open(os.path.join(HERE, "index.html"), "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/items":
            item = {f: str(data.get(f, "")).strip()[:500] for f in FIELDS}
            if not item[FIELDS[0]]:
                return self.send_json({"error": "%s is required" % FIELDS[0]}, 400)
            item.update(id=int(time.time() * 1000), done=False)
            items.append(item)
            save()
            return self.send_json(item, 201)
        if self.path == "/api/toggle":
            for it in items:
                if it["id"] == data.get("id"):
                    it["done"] = not it["done"]
            save()
            return self.send_json({"ok": True})
        if self.path == "/api/delete":
            items[:] = [it for it in items if it["id"] != data.get("id")]
            save()
            return self.send_json({"ok": True})
        self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    print("running on http://127.0.0.1:%d" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''

STATIC_SERVER = r'''import os, sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("running on http://127.0.0.1:%d" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''

CSS = "body{font:16px/1.5 system-ui,sans-serif;max-width:560px;margin:40px auto;padding:0 16px;color:#1c2b30}h1{color:@COLOR@}input,textarea,button{font:inherit;padding:8px 10px;border:1px solid #cfd8dc;border-radius:8px}button{background:@COLOR@;color:#fff;border:0;cursor:pointer}form{display:grid;gap:8px;margin:16px 0}li{display:flex;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid #eceff1}ul{list-style:none;padding:0}li span{flex:1}.done span{text-decoration:line-through;color:#90a4ae}.del{background:none;color:#b0bec5;padding:2px 8px}"

CRUD_PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@TITLE@</title><style>''' + CSS + '''</style></head><body>
<h1>@TITLE@</h1>
<form id="f">@INPUTS@<button>@BUTTON@</button></form>
<ul id="list"></ul>
<script>
const FIELDS = @FIELDS@, TOGGLE = @TOGGLE@;
async function load() {
  const items = await (await fetch('/api/items')).json();
  const ul = document.getElementById('list'); ul.innerHTML = '';
  items.forEach(it => {
    const li = document.createElement('li'); if (it.done) li.className = 'done';
    const text = FIELDS.map(f => it[f]).filter(Boolean).join(' - ');
    li.innerHTML = (TOGGLE ? '<input type="checkbox" ' + (it.done ? 'checked' : '') + '>' : '') + '<span></span><button class="del">x</button>';
    li.querySelector('span').textContent = text;
    if (TOGGLE) li.querySelector('input').onchange = async () => { await post('/api/toggle', {id: it.id}); load(); };
    li.querySelector('.del').onclick = async () => { await post('/api/delete', {id: it.id}); load(); };
    ul.appendChild(li);
  });
}
function post(url, body) { return fetch(url, {method: 'POST', body: JSON.stringify(body)}); }
document.getElementById('f').onsubmit = async e => {
  e.preventDefault(); const body = {};
  FIELDS.forEach(f => { body[f] = document.getElementById(f).value; });
  const r = await post('/api/items', body);
  if (r.ok) { FIELDS.forEach(f => { document.getElementById(f).value = ''; }); load(); }
};
load();
</script></body></html>'''

COUNTER_PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@TITLE@</title><style>''' + CSS + '''.big{font-size:72px;text-align:center;margin:20px 0}.row{display:flex;gap:10px;justify-content:center}</style></head><body>
<h1>@TITLE@</h1><div class="big" id="n">0</div>
<div class="row"><button onclick="add(-1)">-1</button><button onclick="add(1)">+1</button><button onclick="set(0)">Reset</button></div>
<script>
let n = Number(localStorage.getItem('count') || 0);
function set(v) { n = v; localStorage.setItem('count', n); document.getElementById('n').textContent = n; }
function add(d) { set(n + d); }
set(n);
</script></body></html>'''

CALC_PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@TITLE@</title><style>''' + CSS + '''#screen{font-size:32px;text-align:right;padding:12px;background:#f3f6f7;border-radius:8px;min-height:56px}.pad{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}.pad button{padding:16px;font-size:20px}</style></head><body>
<h1>@TITLE@</h1><div id="screen">0</div><div class="pad" id="pad"></div>
<script>
const keys = ['7','8','9','/','4','5','6','*','1','2','3','-','0','.','=','+','C'];
let expr = '';
const screen = document.getElementById('screen'), pad = document.getElementById('pad');
keys.forEach(k => { const b = document.createElement('button'); b.textContent = k; b.onclick = () => press(k); pad.appendChild(b); });
function press(k) {
  if (k === 'C') expr = '';
  else if (k === '=') { try { expr = String(Function('"use strict";return (' + expr + ')')()); } catch (e) { expr = ''; screen.textContent = 'error'; return; } }
  else expr += k;
  screen.textContent = expr || '0';
}
</script></body></html>'''

STOPWATCH_PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@TITLE@</title><style>''' + CSS + '''.big{font-size:64px;text-align:center;margin:20px 0;font-variant-numeric:tabular-nums}.row{display:flex;gap:10px;justify-content:center}</style></head><body>
<h1>@TITLE@</h1><div class="big" id="t">00:00.0</div>
<div class="row"><button id="go">Start</button><button id="rs">Reset</button></div>
<script>
let start = 0, elapsed = 0, timer = null;
const t = document.getElementById('t'), go = document.getElementById('go');
function show(ms) { const s = ms / 1000; t.textContent = String(Math.floor(s / 60)).padStart(2, '0') + ':' + (s % 60).toFixed(1).padStart(4, '0'); }
go.onclick = () => {
  if (timer) { clearInterval(timer); timer = null; elapsed += Date.now() - start; go.textContent = 'Start'; }
  else { start = Date.now(); timer = setInterval(() => show(elapsed + Date.now() - start), 100); go.textContent = 'Stop'; }
};
document.getElementById('rs').onclick = () => { clearInterval(timer); timer = null; elapsed = 0; show(0); go.textContent = 'Start'; };
</script></body></html>'''

RECIPES = {
    "todo": {"label": "to-do list", "server": "crud", "fields": ["text"], "toggle": True, "button": "Add task", "default_title": "My tasks", "placeholders": {"text": "What needs doing?"}},
    "notes": {"label": "notes app", "server": "crud", "fields": ["title", "text"], "toggle": False, "button": "Save note", "default_title": "My notes", "placeholders": {"title": "Title", "text": "Write a note"}},
    "guestbook": {"label": "guestbook", "server": "crud", "fields": ["name", "message"], "toggle": False, "button": "Sign", "default_title": "Guestbook", "placeholders": {"name": "Your name", "message": "Your message"}},
    "counter": {"label": "counter", "server": "static", "page": COUNTER_PAGE, "default_title": "Counter"},
    "calculator": {"label": "calculator", "server": "static", "page": CALC_PAGE, "default_title": "Calculator"},
    "stopwatch": {"label": "stopwatch", "server": "static", "page": STOPWATCH_PAGE, "default_title": "Stopwatch"},
}
KEYWORDS = {"todo": r"to-?do|task|checklist|chore", "notes": r"notes?\b|notebook|memo", "guestbook": r"guest ?book|feedback|comments?", "counter": r"counter|clicker|tally", "calculator": r"calculator|calc\b", "stopwatch": r"stop ?watch|timer"}


def free_port(start=8901):
    for p in range(start, start + 100):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0: return p
    raise RuntimeError("no free port")


def files_for(recipe, title, color):
    r = RECIPES[recipe]; css = {"@COLOR@": color, "@TITLE@": title}
    def sub(t):
        for k, v in css.items(): t = t.replace(k, v)
        return t
    if r["server"] == "crud":
        def one(f):
            ph = r["placeholders"][f]; req = " required" if f == r["fields"][0] else ""
            return f'<textarea id="{f}" placeholder="{ph}"{req}></textarea>' if (f == "text" and len(r["fields"]) > 1) else f'<input id="{f}" placeholder="{ph}"{req}>'
        inputs = "".join(one(f) for f in r["fields"])
        page = sub(CRUD_PAGE.replace("@INPUTS@", inputs).replace("@BUTTON@", r["button"]).replace("@FIELDS@", json.dumps(r["fields"])).replace("@TOGGLE@", "true" if r["toggle"] else "false"))
        return {"server.py": CRUD_SERVER.replace("@FIELDS@", json.dumps(r["fields"])), "index.html": page}
    return {"server.py": STATIC_SERVER, "index.html": sub(r["page"])}


def parse(request):
    """what recipe, title and colour does the request ask for? (None if it is not a request to build an app)"""
    low = request.lower()
    if not re.search(r"\b(build|make|create|generate|write|develop|set up)\b", low) or not re.search(r"\b(app|application|website|web ?site|web ?page|webpage|site)\b", low): return None
    recipe = next((k for k, rx in KEYWORDS.items() if re.search(rx, low)), None)
    color = next((COLORS[c] for c in COLORS if re.search(r"\b" + c + r"\b", low)), None)
    m = re.search(r"(?:called|named|titled)\s+[\"']?([A-Za-z0-9 ,'!&-]{2,40}?)[\"']?(?:\s+(?:with|in|using|that|and)\b|[.!?]?$)", request, re.I)
    return {"recipe": recipe, "title": m.group(1).strip() if m else None, "color": color}


def build(recipe, title=None, color=None):
    t0 = time.time(); r = RECIPES[recipe]; title = title or r["default_title"]; color = color or "#0a6c80"
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or recipe; d = os.path.join(PROJECTS, slug); n = 1
    while os.path.exists(d): n += 1; d = os.path.join(PROJECTS, f"{slug}-{n}")
    os.makedirs(d); files = files_for(recipe, title, color)
    for name, text in files.items(): open(os.path.join(d, name), "w", encoding="utf-8").write(text)
    total = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
    if total > LIMIT: return {"ok": False, "why": f"the app would be {total / 1048576:.1f} MB, over the 10 MB limit"}
    port = free_port(); proc = subprocess.Popen([sys.executable, os.path.join(d, "server.py"), str(port)], cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    url = f"http://127.0.0.1:{port}/"; ok, t1 = False, time.time()
    for _ in range(30):
        try:
            if urllib.request.urlopen(url, timeout=2).status == 200: ok = True; break
        except Exception: time.sleep(0.2)
    name = os.path.basename(d); RUNNING[name] = {"proc": proc, "url": url, "dir": d}
    return {"ok": ok, "name": name, "dir": d, "url": url, "files": {f: len(t.encode()) for f, t in files.items()}, "bytes": total, "start_seconds": round(time.time() - t1, 2), "total_seconds": round(time.time() - t0, 2), "recipe": recipe, "title": title, "color": color, "code": files["server.py"] if r["server"] == "crud" else files["index.html"]}


def stop(name=None):
    names = [name] if name else list(RUNNING); n = 0
    for k in names:
        e = RUNNING.pop(k, None)
        if e: e["proc"].terminate(); n += 1
    return n


def check_app(info):
    """open the built app like a browser would: the page, then add an item (for the apps that store things)"""
    out = {"page_ok": False, "api_ok": None}
    try:
        out["page_ok"] = urllib.request.urlopen(info["url"], timeout=5).status == 200 and info["title"].split()[0] in urllib.request.urlopen(info["url"], timeout=5).read().decode("utf-8")
        if RECIPES[info["recipe"]]["server"] == "crud":
            f0 = RECIPES[info["recipe"]]["fields"]; body = json.dumps({f: "test " + f for f in f0}).encode()
            r = urllib.request.urlopen(urllib.request.Request(info["url"] + "api/items", data=body, method="POST"), timeout=5); items = json.loads(urllib.request.urlopen(info["url"] + "api/items", timeout=5).read())
            out["api_ok"] = r.status == 201 and any(it.get(f0[0]) == "test " + f0[0] for it in items)
    except Exception as e: out["error"] = repr(e)
    return out
