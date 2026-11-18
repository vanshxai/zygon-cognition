"""ZYGON single-screen chat (the front door).   python zygon_door.py  ->  http://127.0.0.1:8890
One screen, never scrolls: the latest question or result plus a "sheet" of what it knows. Corrections overwrite; a hidden log keeps every change.
Jobs (invoice, warehouse, HVAC sizing) are exact guided forms (Fractions, no rounding until the end). Everything else goes to Frost Lang (8840).
All offline on this PC, no AI tokens."""
import json, os, re, sys, time, urllib.request
from fractions import Fraction
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); STORE = os.path.join(os.path.dirname(HERE), "model", "zygon_door.json")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8890
LANG = "http://127.0.0.1:8840/api/ask"
NUM = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")


def F(x): return Fraction(str(x).replace(",", ""))
def money(v): return f"{float(v):,.2f}"
def num(v): return f"{float(v):,.2f}".rstrip("0").rstrip(".")


# ---- jobs: slots = (key, question, words that point at it, default or None); calc returns [(label, value)] rows, last row is the answer
def inv(a):
    sup = a["systems"] * a["unit_price"]; lab = a["hours"] * a["rate"]; sub = sup + lab + a["materials"] + a["travel"]; dis = sub * a["discount"] / 100; net = sub - dis; tax = net * a["tax"] / 100
    return [("Equipment (systems x price)", money(sup)), ("Labour (hours x rate)", money(lab)), ("Materials", money(a["materials"])), ("Travel / call-out", money(a["travel"])), ("Subtotal", money(sub)), ("Discount", "-" + money(dis)), ("Tax", money(tax)), ("TOTAL DUE", money(net + tax))]


def wh(a):
    area = a["length"] * a["width"]; fp = a["pl"] * a["pw"]; lv = int((a["height"] - Fraction(1, 2)) // a["lvh"]); per = int(area * (1 - a["aisle"] / 100) // fp)
    return [("Floor area (m2)", num(area)), ("Volume (m3)", num(area * a["height"])), ("Pallet footprint (m2)", num(fp)), ("Levels that fit", str(lv)), ("Pallets per level", str(per)), ("PALLET POSITIONS", str(per * lv)),
            ("Assumption", "aisles take " + num(a["aisle"]) + "% of the floor; 0.5 m clear space under the roof")]


def hv(a):
    b = a["btuh"]; return [("Tons", num(b / 12000)), ("Airflow, standard air (CFM) = BTU/h / (1.08 x dT)", num(b / (Fraction(108, 100) * a["dt"]))), ("Electrical input for the same heat (W) = BTU/h / 3.412", num(b / Fraction(3412, 1000))),
                            ("Assumption", "sensible heat, standard air at sea level")]


JOBS = {
    "invoice": ("Service invoice", ("invoice", "bill", "quote", "quotation"), inv, [
        ("systems", "How many HVAC systems did you install?", ("systems", "units", "installed"), None), ("unit_price", "Price of one system (equipment)?", ("price", "equipment"), None),
        ("hours", "How many labour hours in total?", ("hours", "hrs", "labour", "labor"), None), ("rate", "Hourly labour rate?", ("rate", "hourly"), None),
        ("materials", "Total cost of materials used (pipes, wire, refrigerant ...)?", ("material", "materials", "pipes"), Fraction(0)), ("travel", "Travel or call-out charge?", ("travel", "callout", "call-out"), Fraction(0)),
        ("discount", "Discount in percent?", ("discount",), Fraction(0)), ("tax", "Tax in percent?", ("tax", "gst", "vat"), Fraction(0))]),
    "warehouse": ("Warehouse layout", ("warehouse", "racking", "storage space", "pallet"), wh, [
        ("length", "Building length in metres?", ("length", "long"), None), ("width", "Building width in metres?", ("width", "wide"), None), ("height", "Clear height under the roof in metres?", ("height", "high", "roof"), None),
        ("pl", "Pallet length in metres?", ("pallet length", "pl"), Fraction(12, 10)), ("pw", "Pallet width in metres?", ("pallet width", "pw"), Fraction(1)),
        ("lvh", "Height of one rack level in metres (pallet + beam)?", ("level", "lvh"), Fraction(18, 10)), ("aisle", "Share of the floor taken by aisles, in percent?", ("aisle",), Fraction(35))]),
    "hvac_size": ("HVAC sizing", ("size", "sizing", "tonnage", "cfm", "btu", "airflow", "tons"), hv, [
        ("btuh", "Cooling or heating load in BTU/h?", ("btu", "load"), None), ("dt", "Air temperature difference in degrees F (typical 18-22)?", ("difference", "delta", "dt"), Fraction(20))]),
}


def load():
    try: return json.load(open(STORE, encoding="utf-8"))
    except Exception: return {"chats": {}}
def save(d):
    os.makedirs(os.path.dirname(STORE), exist_ok=True); json.dump(d, open(STORE, "w", encoding="utf-8"))


_R = _M = _L = None
FLOOR = 0.45
ITEM = re.compile(r"^(.+?)\s+(\d[\d,]*\.?\d*)\s*(?:x|@|at)?\s*(\d[\d,]*\.?\d*)$", re.I)


def router():
    global _R
    if _R is None: import qkv_router; _R = qkv_router.Router()
    return _R
def math():
    global _M, _L
    if _M is None:
        import frostmath; _M = frostmath.Math(); _L = {}
        for l in open(os.path.join(os.path.dirname(HERE), "model", "math_solo", "labels_draft.jsonl"), encoding="utf-8"):
            r = json.loads(l); _L[r["name"]] = r
    return _M


def info(c):
    """title, slots, calc for the active job. A formula job builds its slots from the formula's variables."""
    j = c["job"]
    if j in JOBS: return JOBS[j][0], JOBS[j][3], JOBS[j][2]
    m = math(); name = j.split(":")[1]; card = m.card(name); lab = (_L.get(name) or {}).get("vars", {}); out = card["output"]
    slots = [(v, f"Value of {lab.get(v, v)} ({v})? Units are fine, e.g. 12 kW", tuple(w for w in re.findall(r"[a-z]+", lab.get(v, "").lower()) + [v.lower()] if len(w) > 1), None) for v in card["vars"] if v != out]
    def calc(a):
        r = m.solve_one(name, {k: str(v) for k, v in a.items()}, out)
        if not r.get("ok"): return [("Could not solve", str(r.get("error", r)))]
        t = r["trace"][0]; v = r["value"]
        rows = [("Formula", t["equation"]), ("Substituted", t["substituted"]), (lab.get(out, out) + " (" + out + ")", f"{v['decimal']} {v.get('unit', '')}".strip())]
        rows += [("Also in " + x["unit"], x["value"]) for x in r.get("also_in", [])[:2]]
        rows += [("Put-back check", "passed" if t["check"]["passed"] else "FAILED"), ("TOTAL" if False else "Source", t["source"][:60])]
        return rows
    return (_L.get(name) or {}).get("title", name)[:60], slots, calc


def view(c):
    out = {"id": c["id"], "title": c["title"], "job": c.get("job"), "sheet": [], "ask": None, "rows": None, "text": c.get("text")}
    if c.get("job"):
        title, slots, calc = info(c)
        out["sheet"] = [{"key": k, "label": q.split("?")[0], "value": (num(F(c["a"][k])) if k in c["a"] else "")} for k, q, _, _ in slots]
        items = c.get("items") or []
        for s in out["sheet"]:
            if s["key"] == "materials" and items and not s["value"]: s["value"] = f"{len(items)} items, {num(sum(F(x[1]) * F(x[2]) for x in items))} so far"
        miss = next((s for s in slots if s[0] not in c["a"]), None)
        if miss and miss[0] == "materials" and c["job"] == "invoice": out["ask"] = "Materials used: type each as name qty price (e.g. copper pipe 10 4.5). Press Enter when done." + (" Items so far: " + "; ".join(f"{x[0]} {x[1]}x{x[2]}" for x in items) if items else "")
        elif miss: out["ask"] = miss[1] + (f"  (press Enter for {num(miss[3])})" if miss[3] is not None else "")
        else: out["rows"] = calc({k: F(v) for k, v in c["a"].items()})
    return out


def say(d, cid, text):
    c = d["chats"].setdefault(cid, {"id": cid, "title": "New chat", "job": None, "a": {}, "log": [], "text": None}); t = text.strip(); low = t.lower()
    if low in ("cancel", "start over", "reset", "new job"): c.update(job=None, a={}, items=[], text=None); return view(c)
    if not c["job"]:
        if not t: return view(c)
        name, score = router().route(t, 1)[0]
        if score >= FLOOR and name.startswith("job:"): c.update(job=name[4:], title=JOBS[name[4:]][0], a={}, items=[], text=None)
        elif score >= FLOOR and name.startswith("formula:"): c.update(job=name, a={}, items=[], text=None); c["title"] = info(c)[0]
        elif score >= FLOOR and name == "engg": c["text"] = "That is a job for Frost Engg: open http://localhost:8870, drop in the point cloud or photo set, and it lists the planes and cylinders with sizes."; c["log"].append([time.time(), "engg", t]); return view(c)
        else:
            c["text"] = lang(t); c["title"] = c["title"] if c["title"] != "New chat" else t[:30]; c["log"].append([time.time(), "lang", t]); return view(c)
    title, slots, calc = info(c); nums = NUM.findall(t)
    miss = next((s for s in slots if s[0] not in c["a"]), None)
    if c["job"] == "invoice" and miss and miss[0] == "materials":                   # material list: one line per item
        m = ITEM.match(t)
        if m: c.setdefault("items", []).append([m.group(1), str(F(m.group(2))), str(F(m.group(3)))]); c["log"].append([time.time(), "item", t]); return view(c)
        if not t or low in ("done", "none", "no"): c["a"]["materials"] = str(sum((F(x[1]) * F(x[2]) for x in c.get("items", [])), Fraction(0))); return view(c)
    hit = max(((len(w), k) for k, _, ws, _ in slots for w in ws if w in low), default=None)
    if hit and nums: key = hit[1]
    elif miss and nums: key = miss[0]
    elif miss and miss[3] is not None and not t: key = miss[0]
    else: return view(c)
    if not t: val = str(miss[3])
    elif c["job"].startswith("formula:"):
        try: val = str(math().read_value(t, key, c["job"].split(":")[1])[0])
        except Exception: val = nums[0]
    else: val = nums[0]
    old = c["a"].get(key); c["a"][key] = str(F(val)); c["log"].append([time.time(), key, str(old), c["a"][key]]); return view(c)


def lang(q):
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(LANG, json.dumps({"q": q}).encode(), {"Content-Type": "application/json"}), timeout=20)); return r.get("text") or "I do not know that yet."
    except Exception: return "The language engine is not running. Start ZYGON Lang from the dashboard (port 8840)."


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, b, ct="application/json"): self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        d = load()
        if self.path.startswith("/api/chats"): return self.send(json.dumps([{"id": c["id"], "title": c["title"]} for c in d["chats"].values()]).encode())
        if self.path.startswith("/api/open"): c = d["chats"].get(self.path.split("id=")[-1]); return self.send(json.dumps(view(c) if c else {}).encode())
        self.send(open(os.path.join(HERE, "zygon_door.html"), "rb").read(), "text/html; charset=utf-8")
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"); d = load(); r = say(d, b["chat"], b.get("text", "")); save(d); self.send(json.dumps(r).encode())


if __name__ == "__main__":
    print(f"READY  ZYGON chat http://127.0.0.1:{PORT}", flush=True); ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
